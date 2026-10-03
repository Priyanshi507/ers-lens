import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from laptime_validate import run  # noqa: E402

from erslens.laptime import clip_time_loss
from erslens.params import CarParams


def test_time_loss_matches_simulated_truth():
    df = run(8, seed=5)
    err = (df.est_s - df.true_s).abs()
    assert err.median() < 0.06
    assert np.corrcoef(df.est_s, df.true_s)[0, 1] > 0.95


def test_no_clipping_means_no_time_loss():
    t = np.arange(0, 12, 0.25)
    kmh = np.minimum(220 + 10 * t, 330.0)
    lap = pd.DataFrame({"time_s": t, "distance_m": np.cumsum(kmh / 3.6 * 0.25), "speed_kmh": kmh,
                        "throttle": 100.0, "brake": False})
    r = clip_time_loss(lap, 0, 1e6, 800.0, CarParams())
    assert r is not None and not r["clipping"] and r["t_loss_s"] == 0.0


def test_single_throttle_blip_does_not_hide_clipping():
    from erslens.throttle import full_throttle_mask

    thr = np.array([100, 100, 95, 100, 100, 85, 100], dtype=float)
    brake = np.zeros(7, dtype=bool)
    mask = full_throttle_mask(thr, brake)
    assert mask[2]          # 95% between full-throttle samples is noise
    assert not mask[5]      # 85% is a real lift
