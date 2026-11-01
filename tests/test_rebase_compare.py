import sys

import pandas as pd

sys.path.insert(0, "scripts")
from rebase_compare import compare  # noqa: E402


def summary(rows):
    return pd.DataFrame(rows, columns=["event", "straight_m", "clean_laps", "clip_share", "clip_frac_mean"])


def test_verdicts_follow_the_written_rules():
    old = summary([["Bahrain Grand Prix", 700, 500, 0.30, 0.1], ["Italian Grand Prix", 1240, 500, 0.6, 0.2],
                   ["Monaco Grand Prix", 510, 900, 0.0, 0.0], ["China", 1320, 250, 0.93, 0.36]])
    new = summary([["Bahrain Grand Prix", 1100, 400, 0.85, 0.3], ["Italian Grand Prix", 1250, 500, 0.65, 0.2],
                   ["Monaco Grand Prix", 600, 10, 0.0, 0.0], ["China", 1320, 250, 0.93, 0.36]])
    v = compare(old, new)
    assert v["R1"] == (1, 1, True)
    assert v["R2"] == (1, 2, True)
    assert v["R3"][1] == 3
