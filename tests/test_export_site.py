import json
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "scripts")
from export_site import lap_records, lap_trace, race_id, summarise_races, team_table  # noqa: E402


def per_race(**overrides):
    row = {"event": "Canadian Grand Prix", "date": "2026-05-24", "clean_laps": 700,
           "measurable_share": 0.9, "laps": 630, "failure_share": 0.05, "clipping_share": 0.95,
           "t_loss_median_s": 0.25, "t_loss_q25_s": 0.165, "t_loss_q75_s": 0.338, "valid": True}
    return {**row, **overrides}


def test_reported_and_unreported_races():
    races = summarise_races(pd.DataFrame([
        per_race(),
        per_race(event="Austrian Grand Prix", date="2026-06-28", measurable_share=0.3, valid=False),
        per_race(event="Australian Grand Prix", date="2026-03-08", laps=np.nan, failure_share=np.nan,
                 measurable_share=0.0, valid=False),
    ]))
    assert [r["id"] for r in races] == ["australian", "canadian", "austrian"]
    canada = races[1]
    assert canada["reported"] and canada["reason"] is None and canada["loss_median_s"] == 0.25
    assert canada["period"] == "after" and races[0]["period"] == "before"
    for r in (races[0], races[2]):
        assert not r["reported"] and "measure" in r["reason"] and r["loss_median_s"] is None
    json.dumps(races)


def test_invalid_race_without_a_known_reason_fails_loudly():
    with pytest.raises(ValueError):
        summarise_races(pd.DataFrame([per_race(valid=False)]))


def per_lap():
    rows = []
    for team, base in (("Fast", 0.2), ("Slow", 0.4)):
        for lap in range(1, 11):
            rows.append({"event": "X", "driver": team[:3].upper(), "team": team, "lap": lap,
                         "measurable": True, "clipping": True, "t_loss_s": base + 0.001 * lap,
                         "t_low_s": base - 0.05, "t_high_s": base + 0.05, "swing_kw": 200.0})
    rows.append({**rows[0], "lap": 50, "t_loss_s": -0.5})                        # model failure
    rows.append({**rows[0], "lap": 51, "clipping": False, "t_loss_s": 0.0})       # no clipping
    rows.append({**rows[0], "lap": 52, "measurable": False, "clipping": np.nan,
                 "t_loss_s": np.nan, "swing_kw": np.nan})                         # not measurable
    return pd.DataFrame(rows)


def test_team_table_uses_only_measured_clipping_laps():
    teams, left_out = team_table(per_lap())
    assert [t["team"] for t in teams] == ["Fast", "Slow"] and left_out == []
    assert all(t["laps"] == 10 for t in teams)
    assert teams[0]["vs_race_s"] < 0 < teams[1]["vs_race_s"]


def test_teams_with_too_few_laps_are_named_not_charted():
    laps = per_lap()
    few = laps.iloc[:3].assign(team="Rare", driver="RAR", t_loss_s=0.9)
    teams, left_out = team_table(pd.concat([laps, few], ignore_index=True))
    assert "Rare" not in [t["team"] for t in teams]
    assert left_out == [{"team": "Rare", "laps": 3}]


def test_lap_records_drop_failures_and_unmeasurable_laps():
    recs = lap_records(per_lap(), traces={("FAS", 1): {"d": [0], "v": [300], "flat": [0, 0], "peak": 0}},
                       max_power_kw=350)
    laps = {(r["driver"], r["lap"]) for r in recs}
    assert ("FAS", 50) not in laps and ("FAS", 52) not in laps and ("FAS", 51) in laps
    assert "trace" in next(r for r in recs if (r["driver"], r["lap"]) == ("FAS", 1))
    json.dumps(recs)


def test_interval_includes_simulation_error_even_when_setups_agree():
    from export_site import SIM_ERROR_90_S

    laps = per_lap().assign(t_low_s=0.25, t_high_s=0.25, t_loss_s=0.25)
    rec = lap_records(laps, traces={}, max_power_kw=350)[0]
    assert rec["loss_low_s"] == round(0.25 - SIM_ERROR_90_S, 3) and rec["loss_high_s"] == round(0.25 + SIM_ERROR_90_S, 3)


def test_impossible_power_drop_is_withheld():
    laps = per_lap().assign(swing_kw=478.0)
    rec = lap_records(laps, traces={}, max_power_kw=350)[0]
    assert rec["power_drop_kw"] is None and rec["power_drop_implausible"]


def test_lap_trace_marks_full_throttle_run_containing_the_speed_peak():
    d = np.arange(0, 1300, 25.0)
    v = np.minimum(200 + 0.12 * d, 330)
    throttle = np.where(d < 1100, 100.0, 0.0)
    lap = pd.DataFrame({"distance_m": d + 1000, "speed_kmh": v, "throttle": throttle,
                        "brake": d >= 1150})
    t = lap_trace(lap, start_m=1000, end_m=2100)
    assert t["d"][0] == 0 and len(t["d"]) == len(t["v"])
    a, b = t["flat"]
    assert a == 0 and a <= t["peak"] <= b and t["d"][b] < 1100
    json.dumps(t)


def test_race_id_is_url_safe():
    assert race_id("São Paulo Grand Prix") == "sao-paulo"
    assert race_id("Miami Grand Prix") == "miami"


def test_committed_site_data_is_consistent():
    """Guards deployment: the published data must match the schema and agree with itself."""
    from pathlib import Path

    from export_site import SCHEMA_VERSION

    data = Path("site/data")
    if not (data / "index.json").exists():
        pytest.skip("no exported site data committed yet")
    index = json.loads((data / "index.json").read_text())
    assert index["schema"] == SCHEMA_VERSION and index["races"]
    assert 0 < index["accuracy"]["typical_lap_s"] <= index["accuracy"]["team_band_s"]
    for race in index["races"]:
        assert (race["reason"] is None) == race["reported"]
        detail = json.loads((data / "races" / f"{race['id']}.json").read_text())
        assert detail["schema"] == SCHEMA_VERSION and detail["loss_median_s"] == race["loss_median_s"]
        assert all(t["laps"] >= detail["min_team_laps"] for t in detail["teams"])
        for lap in detail["laps"]:
            assert lap["loss_low_s"] < lap["loss_s"] < lap["loss_high_s"]
            assert lap["power_drop_kw"] is None or lap["power_drop_kw"] <= 350
            if "trace" in lap:
                t = lap["trace"]
                assert len(t["d"]) == len(t["v"]) and 0 <= t["flat"][0] <= t["peak"] <= t["flat"][1] < len(t["d"])
