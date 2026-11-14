import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from hindcast_2026 import table  # noqa: E402


def test_hindcast_pairs_every_used_race_with_a_profile():
    t = table(Path("results/fia_fix_calendar_c"), Path("results/multi_rebased_mid/R/race_summary.csv"))
    assert len(t) == 16
    assert set(t.cap_kw[t.event.isin(["China", "Japan"])]) == {250}
    assert (t.model_drop_kmh >= 0).all()
