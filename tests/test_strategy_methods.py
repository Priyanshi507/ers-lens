import pytest

from erslens.gradopt import solve_m2
from erslens.optimal import solve
from erslens.params import CarParams
from erslens.simple import tune
from erslens.track import toy_track

TRACK = toy_track([("straight", 600, None), ("corner", 100, 90),
                   ("straight", 300, None), ("corner", 80, 140)], name="short")
CAR = CarParams()


@pytest.fixture(scope="module")
def results():
    return solve(TRACK, CAR, soc0_fracs=(0.5,)), solve_m2(TRACK, CAR), tune(TRACK, CAR)


def test_every_simple_family_is_tuned_and_energy_neutral(results):
    _, _, tuned = results
    assert set(tuned) == {"flat_out", "straights_only", "clip_end"}
    assert all(t.lap_time_s > 0 for t in tuned.values())


def test_gradient_method_matches_or_beats_dynamic_programming(results):
    m1, (m2, _), _ = results
    assert m2.lap_time_sim_s <= m1.lap_time_sim_s + 0.02
    assert m2.soc_end_j >= m2.soc0_j - 5e3


def test_optimizers_beat_the_best_simple_rule(results):
    m1, (m2, seed), tuned = results
    best_simple = min(t.lap_time_s for t in tuned.values())
    assert seed.lap_time_s == best_simple
    assert max(m1.lap_time_sim_s, m2.lap_time_sim_s) <= best_simple + 0.01
