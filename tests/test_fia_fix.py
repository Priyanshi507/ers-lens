import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fia_fix import superclip_metrics  # noqa: E402
from fia_fix_verdicts import verdicts  # noqa: E402


def test_superclip_metrics_measure_energy_time_and_longest_stretch():
    out = {"superclip": np.array([0, 250e3, 250e3, 0, 100e3, 0.0]), "dt": np.full(6, 0.5)}
    m = superclip_metrics(out, 5.0, 250e3)
    assert m["superclip_mj"] == 0.3 and m["superclip_s"] == 1.5 and m["longest_superclip_s"] == 1.0
    assert m["superclip_m"] == 15.0 and m["peak_superclip_kw"] == 250.0
    assert abs(m["superclip_at_cap_share"] - 2 / 3) < 1e-12


def run(event, car, cap, lap, longest, mj, converged=True):
    return {"event": event, "car": car, "cap_kw": cap, "lap_s": lap, "converged": converged,
            "longest_superclip_s": longest, "superclip_mj": mj, "superclip_s": 1.0}


def test_failed_solves_count_against_the_predictions():
    rows = []
    for i in range(12):
        ok = i != 0
        rows += [run(f"e{i}", "c", 250.0, 80.0, 2.0, 3.0), run(f"e{i}", "c", 350.0, 79.9, 1.5, 3.5, ok)]
    v = verdicts(pd.DataFrame(rows))
    assert v["F1"] == (11, 12, False) and v["F2"] == (11, 12, True) and v["F3"] == (11, 12, True)
