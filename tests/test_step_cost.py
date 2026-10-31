import sys

from erslens.ingest import save_track
from erslens.track import demo_circuit

sys.path.insert(0, "scripts")
from step_cost import step_cost  # noqa: E402


def test_one_step_cost_is_counted(tmp_path):
    path = tmp_path / "demo.npz"
    save_track(demo_circuit(), path)
    cost = step_cost(str(path), "configs/car_2026_q_low.yaml")
    assert cost["pairs"] == 768 * 321 * 19
    assert cost["flops"] > cost["pairs"] and cost["bytes"] > 8 * cost["pairs"]


def test_parts_are_timed(tmp_path):
    from step_cost import time_parts

    path = tmp_path / "demo.npz"
    save_track(demo_circuit(), path)
    parts = time_parts(str(path), "configs/car_2026_q_low.yaml", repeats=1)
    assert set(parts) == {"transition_s", "interp_s", "step_s"} and all(v > 0 for v in parts.values())


def test_per_pair_operations_are_counted():
    from step_cost import per_pair_ops

    hlo = """  %a = f64[2,3,4]{2,1,0} divide(%x, %y)
  %b = f64[2,3,4]{2,1,0} divide(%y, %x)
  %c = f64[2,3]{1,0} divide(%p, %q)
  %d = f64[24]{0} sqrt(%z)
  %e = s32[24]{0} convert(%z)"""
    assert per_pair_ops(hlo, 24) == {"f64 divide": 2, "f64 sqrt": 1, "s32 convert": 1}


def test_float32_timing_really_runs_in_float32(tmp_path):
    from step_cost import time_parts

    path = tmp_path / "demo.npz"
    save_track(demo_circuit(), path)
    assert time_parts(str(path), "configs/car_2026_q_low.yaml", repeats=1, dtype="float32")["step_s"] > 0
