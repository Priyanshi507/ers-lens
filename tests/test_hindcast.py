import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from hindcast_2026 import table  # noqa: E402


def test_hindcast_pairs_every_used_race_with_a_profile():
    t = table(Path("results/fia_fix_calendar_c"), Path("results/multi_rebased_mid/R/race_summary.csv"))
    assert len(t) == 16
    assert set(t.cap_kw[t.event.isin(["China", "Japan"])]) == {250}
    assert (t.model_drop_kmh >= 0).all()


def test_model_drop_stops_where_full_throttle_ends():
    import numpy as np
    from fia_fix_figures import longest_free_run
    from hindcast_2026 import BRAKES, drop_from_profile, full_throttle

    path = Path("results/fia_fix_calendar_c/profiles/Monaco Grand Prix_original_350.npz")
    z = np.load(path)
    i, j = longest_free_run(z["v_limit"])
    assert not full_throttle(z, BRAKES)[i:j][-1]          # the run ends under braking
    assert z["superclip"][i:j].max() == 0.0 and drop_from_profile(path) == 0.0
