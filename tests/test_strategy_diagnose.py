import sys

import numpy as np
import pytest

from erslens.optimal import solve
from erslens.params import CarParams
from erslens.simple import tune
from erslens.track import toy_track

sys.path.insert(0, "scripts")
from strategy_diagnose import time_breakdown, zones  # noqa: E402

TRACK = toy_track([("straight", 600, None), ("corner", 100, 90), ("straight", 300, None), ("corner", 80, 140)])


def test_zones_cover_the_lap_without_gaps():
    free = np.array([False, True, True, False, False, True])
    z = zones(free)
    assert z == [(0, 1, False), (1, 3, True), (3, 5, False), (5, 6, True)]


def test_breakdown_reproduces_lap_times_and_sums_to_the_total():
    car = CarParams()
    opt = solve(TRACK, car, soc0_fracs=(0.5,))
    best = min(tune(TRACK, car).values(), key=lambda t: t.lap_time_s)
    rows = {r["what"]: r["gain_s"] for r in time_breakdown(TRACK, car, opt.cmd_w, best.cmd_w)}
    assert rows["total"] == pytest.approx(best.lap_time_s - opt.lap_time_sim_s, abs=1e-6)
    assert rows["all corners"] + rows["all straights"] == pytest.approx(rows["total"], abs=1e-9)
