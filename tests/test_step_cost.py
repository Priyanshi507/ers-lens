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
