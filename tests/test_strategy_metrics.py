import numpy as np

from erslens.params import CarParams
from erslens.strategy import clip_metrics, legal_sweep, longest_full_throttle_run, no_taper
from erslens.physics import mguk_cap_kmh as mguk_cap
from erslens.track import Track


def track(limits):
    n = len(limits)
    return Track("t", np.arange(n) * 5.0, np.asarray(limits, float), np.isinf(limits), None)


def test_longest_run_is_found():
    inf = np.inf
    assert longest_full_throttle_run(track([30, inf, inf, 30, inf, inf, inf, 30])) == (4, 7)


def test_clip_ratio_compares_power_at_peak_with_pre_peak_mean():
    inf = np.inf
    t = track([30, inf, inf, inf, inf, 30])
    out = {"v": np.array([30, 60, 70, 80, 75, 30.0]), "deploy": np.array([0, 300, 300, 60, 0, 0.0]) * 1e3,
           "harvest": np.zeros(6), "dt": np.ones(6)}
    m = clip_metrics(t, out)
    assert m["clip_ratio"] == 0.2
    assert m["peak_from_start_m"] == 10.0 and m["run_length_m"] == 20.0


def test_sweep_has_27_distinct_legal_cars():
    cars = legal_sweep(CarParams())
    assert len(cars) == 27 and len({c for _, c in cars}) == 27


def test_no_taper_keeps_the_flat_cap_at_every_speed():
    car = no_taper(CarParams())
    kmh = np.array([100.0, 290.0, 330.0, 360.0])
    assert np.allclose(mguk_cap(kmh, car), car.mguk_power_w)
    assert mguk_cap(np.array([360.0]), CarParams())[0] < CarParams().mguk_power_w
