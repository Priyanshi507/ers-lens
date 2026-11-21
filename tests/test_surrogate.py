import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import surrogate  # noqa: E402
from surrogate import advised, baseline, linear, runs_of, track_features  # noqa: E402


def test_runs_of_finds_each_stretch():
    assert runs_of(np.array([True, True, False, True])) == [(0, 2), (3, 4)]


def test_features_of_a_real_track_are_sane():
    f = track_features(__import__("pathlib").Path("results/tracks/Italian Grand Prix.npz"))
    assert 5.0 < f["lap_km"] < 6.0 and 0.3 < f["free_share"] < 1.0 and f["longest_km"] > 0.5


def frame(n=6):
    rows = []
    for e in range(n):
        for mj in (5.0, 7.0, 9.0):
            rows.append({"event": f"c{e}", "lap_km": 5 + e / 10, "free_share": 0.5 + e / 20, "longest_km": 1.0,
                         "long_straights": 3, "slow_corners": 4, "limit_mj": mj,
                         "superclip_s": max(0.0, (mj - 5.0) * (1 + e / 5))})
    return pd.DataFrame(rows)


def test_linear_and_baseline_on_a_worked_case():
    d = frame()
    train, test = d[d.event != "c0"], d[d.event == "c0"]
    assert np.allclose(baseline(train, test), [0.0, 3.2, 6.4])
    assert np.abs(linear(d, d) - d.superclip_s).max() < 2.0


def test_advised_takes_the_largest_limit_within_the_target():
    p = pd.DataFrame({"event": "a", "limit_mj": [5.0, 7.0, 9.0], "network": [1.0, 3.9, 6.0]})
    assert advised(p, "network")["a"] == 7.0


def test_the_network_ensemble_runs_and_gives_positive_spread(monkeypatch):
    monkeypatch.setattr(surrogate, "SEEDS", (0,))
    monkeypatch.setattr(surrogate, "NET", {"hidden": (8,), "epochs": 20, "batch": 4, "lr": 3e-3})
    d = frame()
    mean, sd = surrogate.ensemble(d, d)
    assert mean.shape == (len(d),) and (sd > 0).all()
