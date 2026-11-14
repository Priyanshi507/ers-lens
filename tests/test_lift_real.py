import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from derate_analysis import coast_before_brake, lap_coast  # noqa: E402


def window(throttle, brake, step_m=20.0, speed_ms=80.0):
    n = len(throttle)
    d = np.arange(n) * step_m
    return pd.DataFrame({"throttle": throttle, "brake": brake, "distance_m": d, "time_s": d / speed_ms})


def test_coast_is_throttle_closed_to_brake_on():
    w = window([100, 100, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1])
    assert coast_before_brake(w, 1) == {"coast_before_brake_m": 60.0, "coast_before_brake_s": 0.75}


def test_no_coast_when_the_brake_comes_first_and_unknown_without_a_brake():
    assert coast_before_brake(window([100, 100, 0, 0], [0, 0, 1, 1]), 1)["coast_before_brake_m"] == 0.0
    assert coast_before_brake(window([100, 100, 50, 0], [0, 0, 0, 1]), 1)["coast_before_brake_m"] == 0.0
    assert np.isnan(coast_before_brake(window([100, 100, 0, 0], [0, 0, 0, 0]), 1)["coast_before_brake_m"])


def test_lap_coast_sums_every_coast_into_a_brake_of_60_m_or_more():
    thr = [100, 0, 0, 0, 0, 0, 100, 100, 0, 0, 0, 100, 100, 0, 0]
    brk = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # 80 m, then 40 m, then a lift with no brake
    assert lap_coast(window(thr, brk)) == {"lap_coast_m": 80.0, "lap_coast_events": 1}
