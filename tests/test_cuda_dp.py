import jax.numpy as jnp
import numpy as np
import pytest

from erslens.cuda_dp import constants, reference_step
from erslens.jaxsim import params_of, static_track
from erslens.optimal import _interp, _transition, grid_with, race_car, start_at_slowest_point
from erslens.params import load_car_params
from erslens.track import demo_circuit


@pytest.fixture(scope="module")
def setup():
    rc = race_car(load_car_params("configs/car_2026_q_low.yaml")).with_(superclip_max_w=250e3)
    tr = start_at_slowest_point(demo_circuit(), rc)
    g = grid_with(rc, 0.5, 41)
    st = static_track(tr, rc)
    p, mass = params_of(rc), rc.mass_kg + rc.fuel_kg
    unlimited = rc.with_(harvest_per_lap_j=np.inf)
    V, S, U = (jnp.asarray(a) for a in (g.v_ms, g.soc_j, g.cmd_w))
    k = constants(p, unlimited, st.ds, mass)
    table = np.add.outer(np.linspace(0.0, 30.0, len(g.v_ms)), np.cos(np.linspace(0.0, 7.0, len(g.soc_j))))
    return g, st, p, unlimited, mass, V, S, U, k, table


@pytest.mark.parametrize("point,cap", [(0, np.inf), (50, np.inf), (200, np.inf), (400, np.inf), (700, np.inf),
                                       (200, 250e3), (700, 250e3)])
def test_reference_step_matches_the_xla_step(setup, point, cap):
    g, st, p, unlimited, mass, V, S, U, k, table = setup
    target, straight = st.env[point], st.straight[point]
    v_n, s_n, e_in, dt = _transition(p, unlimited, st.ds, mass, V, S, U, target, straight, cap)
    xla = np.asarray((dt + 2e-7 * e_in + _interp(jnp.asarray(table), g, v_n, s_n)).min(axis=1))
    ref = reference_step(table, g.v_ms, g.soc_j, g.cmd_w, float(target), bool(straight), 2e-7, k, cap)
    assert np.abs(ref - xla).max() < 1e-9


def test_kernel_source_run_on_the_cpu_matches_the_reference(setup, tmp_path):
    """Compiles the CUDA kernel's own code as C++ (thread indices stubbed) and runs every thread."""
    import ctypes
    import shutil
    import subprocess

    from erslens.cuda_dp import KERNEL

    if shutil.which("g++") is None:
        pytest.skip("no C++ compiler")
    stub = ("#include <cmath>\n#include <algorithm>\nusing std::fmin; using std::fmax; using std::sqrt; using std::floor;\n"
            "using std::min; using std::max;\n#define __global__\n#define __restrict__\n"
            "struct D { int x, y; }; static D blockIdx, blockDim, threadIdx;\n")
    harness = """
extern "C" void run_all(const double* vn, double* out, const double* cmd, const double* vg, const double* sg,
    int nv, int ns, int nu, double v0, double dv, double dsoc, double target, int straight, double price, const double* k, double cap) {
  blockDim.x = 1; threadIdx.x = 0;
  for (int iv = 0; iv < nv; ++iv) for (int is = 0; is < ns; ++is) {
    blockIdx.y = iv; blockIdx.x = is;
    dp_step(vn, out, cmd, vg, sg, nv, ns, nu, v0, dv, dsoc, target, straight, price, k, cap);
  }
}
"""
    src, lib_path = tmp_path / "k.cpp", tmp_path / "k.so"
    src.write_text(stub + KERNEL + harness)
    subprocess.run(["g++", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(lib_path), str(src)], check=True)
    lib = ctypes.CDLL(str(lib_path))
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    g, st, p, unlimited, mass, V, S, U, k, table = setup
    vg, sg, cm = (np.ascontiguousarray(a, dtype=np.float64) for a in (g.v_ms, g.soc_j, g.cmd_w))
    table = np.ascontiguousarray(table)
    for point, cap in ((0, np.inf), (200, np.inf), (700, np.inf), (200, 250e3)):
        target, straight = float(st.env[point]), bool(st.straight[point])
        out = np.empty_like(table)
        lib.run_all(ptr(table), ptr(out), ptr(cm), ptr(vg), ptr(sg), len(vg), len(sg), len(cm), ctypes.c_double(vg[0]),
                    ctypes.c_double(vg[1] - vg[0]), ctypes.c_double(sg[1] - sg[0]), ctypes.c_double(target),
                    int(straight), ctypes.c_double(2e-7), ptr(k), ctypes.c_double(cap))
        assert np.array_equal(out, reference_step(table, g.v_ms, g.soc_j, g.cmd_w, target, straight, 2e-7, k, cap))


def test_unknown_backend_is_rejected(monkeypatch):
    from erslens.optimal import dp_backend

    monkeypatch.setenv("ERSLENS_DP_BACKEND", "tpu")
    with pytest.raises(ValueError):
        dp_backend()
    monkeypatch.delenv("ERSLENS_DP_BACKEND")
    assert dp_backend() == "xla"


def test_benchmark_verdicts_follow_the_written_rules():
    import sys

    sys.path.insert(0, "scripts")
    from cuda_benchmark import verdicts

    good = {"max_value_diff_s": 1e-12, "backward_xla_s": 7.5, "backward_cuda_s": 3.0,
            "solve_xla": {"lap_s": 77.5813, "seconds": 58.0}, "solve_cuda": {"lap_s": 77.5815, "seconds": 30.0}}
    slow = {**good, "backward_cuda_s": 5.0, "solve_cuda": {"lap_s": 77.5815, "seconds": 50.0}}
    v = verdicts({"events": {"A": good, "B": slow}})
    assert v["K1"] == (2, 2, True) and v["K2"] == (1, 2, False) and v["K3"] == (1, 2, False)
