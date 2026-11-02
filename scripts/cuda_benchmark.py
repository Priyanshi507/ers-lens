"""The hand-written CUDA backward pass against XLA's, same session, same GPU (research log, 2026-11-02).

For each circuit: one backward pass on the fine grid with each backend (median of --repeats, after
a warm-up), the largest difference between their value tables over every point and state, and a
full solve with each backend (lap time, wall time). Writes a JSON file and prints the K1-K3
verdicts.

    python scripts/cuda_benchmark.py --out results/cuda_benchmark/t4.json     # on a CUDA GPU
"""
import os

# JAX would otherwise reserve 75% of GPU memory and leave none for CuPy.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

EVENTS = ("Australia", "China", "Miami", "Canada")
CAR = "configs/car_2026_q_low.yaml"
SAME_VALUE_S = 1e-9
SAME_LAP_S = 1e-3


def timed(f, sync, repeats):
    sync(f())
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        sync(f())
        runs.append(time.perf_counter() - t)
    return statistics.median(runs), runs


def solve(track, car, backend):
    from erslens.optimal import solve_refined

    os.environ["ERSLENS_DP_BACKEND"] = backend
    t = time.perf_counter()
    sol = solve_refined(track, car)
    return {"seconds": round(time.perf_counter() - t, 2), "lap_s": sol.lap_time_sim_s, "dp_pred_s": sol.lap_time_dp_s}


def run(events, repeats):
    import cupy as cp
    import jax

    from erslens.cuda_dp import CudaBackward, constants
    from erslens.ingest import load_track
    from erslens.jaxsim import params_of, static_track
    from erslens.optimal import MISS_PENALTY_S_PER_J, _Compiled, grid_with, race_car, start_at_slowest_point
    from erslens.params import load_car_params

    car = load_car_params(CAR)
    dev = jax.devices()[0]
    out = {"device": getattr(dev, "device_kind", dev.platform), "jax": jax.__version__, "cupy": cp.__version__,
           "car": CAR, "events": {}}
    for e in events:
        track = load_track(f"results/tracks/{e}.npz")
        rc = race_car(car)
        tr = start_at_slowest_point(track, rc)
        grid = grid_with(rc, 0.125, 321)
        os.environ["ERSLENS_DP_BACKEND"] = "xla"
        xla = _Compiled(tr, rc, grid, True)
        st = static_track(tr, rc)
        unlimited = rc.with_(harvest_per_lap_j=np.inf)
        cuda = CudaBackward(grid.v_ms, grid.soc_j, grid.cmd_w, np.asarray(st.env), np.asarray(st.straight),
                            constants(params_of(rc), unlimited, st.ds, rc.mass_kg + rc.fuel_kg), True,
                            MISS_PENALTY_S_PER_J)
        soc0 = 0.5 * rc.es_capacity_j
        t_xla, runs_xla = timed(lambda: xla.backward(soc0, 0.0)[0], jax.block_until_ready, repeats)
        t_cuda, runs_cuda = timed(lambda: cuda(soc0, 0.0)[0], lambda v: cp.cuda.Device().synchronize(), repeats)
        v_xla = cp.from_dlpack(xla.backward(soc0, 0.0)[0])
        v_cuda = cuda(soc0, 0.0)[0]
        diff = float(cp.abs(v_xla - v_cuda).max())
        sums = (float(v_xla[0].sum()), float(v_cuda[0].sum()))
        del v_xla, v_cuda
        entry = {"backward_xla_s": round(t_xla, 4), "backward_cuda_s": round(t_cuda, 4),
                 "runs_xla_s": [round(r, 4) for r in runs_xla], "runs_cuda_s": [round(r, 4) for r in runs_cuda],
                 "max_value_diff_s": diff, "checksum_xla": sums[0], "checksum_cuda": sums[1]}
        print(f"{e}: backward XLA {t_xla:.3f} s, CUDA {t_cuda:.3f} s ({t_xla / t_cuda:.2f}x), largest value "
              f"difference {diff:.2e} s", flush=True)
        entry["solve_xla"], entry["solve_cuda"] = solve(track, car, "xla"), solve(track, car, "cuda")
        print(f"{e}: full solve XLA {entry['solve_xla']['seconds']:.0f} s lap {entry['solve_xla']['lap_s']:.4f} s, "
              f"CUDA {entry['solve_cuda']['seconds']:.0f} s lap {entry['solve_cuda']['lap_s']:.4f} s", flush=True)
        out["events"][e] = entry
    return out


def verdicts(data: dict) -> dict:
    ev = data["events"].values()
    k1 = [x["max_value_diff_s"] <= SAME_VALUE_S for x in ev]
    k2 = [x["backward_xla_s"] / x["backward_cuda_s"] >= 2.0 for x in ev]
    k3 = [abs(x["solve_xla"]["lap_s"] - x["solve_cuda"]["lap_s"]) <= SAME_LAP_S
          and x["solve_xla"]["seconds"] / x["solve_cuda"]["seconds"] >= 1.5 for x in ev]
    return {k: (sum(v), len(v), all(v)) for k, v in (("K1", k1), ("K2", k2), ("K3", k3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=list(EVENTS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = run(args.events, args.repeats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=1))
    for k, (n, total, held) in verdicts(data).items():
        print(f"{k}: {n} of {total} -> {'held' if held else 'NOT held'}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
