import sys

import numpy as np
import pandas as pd
import pytest

from erslens import openf1

sys.path.insert(0, "scripts")
from compare_sources import LAPTIME_CHECKS, MULTI_CHECKS, compare  # noqa: E402

T0 = pd.Timestamp("2026-06-07T13:00:00Z")


def car_table(seconds, speeds, throttle=100.0):
    n = len(seconds)
    return pd.DataFrame({"date": T0 + pd.to_timedelta(np.asarray(seconds, float), unit="s"), "speed": speeds,
                         "throttle": np.full(n, throttle), "rpm": np.full(n, 11000.0),
                         "brake": np.zeros(n), "n_gear": np.full(n, 8)})


def test_filters_put_the_operator_in_the_key():
    url = openf1._url("car_data", [("session_key", 9001), ("date>=", "2026-06-07T13:00:00+00:00")])
    assert url == "https://api.openf1.org/v1/car_data?session_key=9001&date>=2026-06-07T13%3A00%3A00%2B00%3A00"


def test_lap_slice_matches_fastf1_distance_convention():
    car = car_table(np.arange(-1.0, 12.0, 0.25), np.full(52, 180.0))
    df = openf1.slice_lap(car, T0, T0 + pd.Timedelta(10, unit="s"))
    assert df["time_s"].iloc[0] == 0.0 and df["time_s"].iloc[-1] == pytest.approx(10.0)
    # Constant 180 km/h for 10 s is 500 m; FastF1 integrates speed over the time since the previous sample.
    assert df["distance_m"].iloc[-1] == pytest.approx(500.0)
    assert np.all(np.diff(df["distance_m"]) >= 0)


def test_lap_edges_are_interpolated():
    car = car_table([-0.3, 0.2, 0.5, 0.9, 1.4, 1.8, 2.3], [100, 110, 120, 130, 140, 150, 160])
    df = openf1.slice_lap(car, T0, T0 + pd.Timedelta(2.0, unit="s"))
    assert df["speed_kmh"].iloc[0] == pytest.approx(100 + 10 * 0.3 / 0.5)
    assert df["speed_kmh"].iloc[-1] == pytest.approx(150 + 10 * 0.2 / 0.5)


def test_too_few_samples_is_not_a_lap():
    assert openf1.slice_lap(car_table([0.1, 0.5], [100, 110]), T0, T0 + pd.Timedelta(1, unit="s")) is None


def test_lap_end_falls_back_to_the_next_start():
    laps = pd.DataFrame({"lap_number": [1, 2, 3], "lap_duration": [None, 80.0, None],
                         "date_start": [None, "2026-06-07T13:01:00+00:00", "2026-06-07T13:02:20+00:00"]})
    w = openf1.lap_windows(laps)
    assert list(w["lap_number"]) == [2]
    assert w["end"].iloc[0] == pd.Timestamp("2026-06-07T13:02:20Z")


def test_deleted_lap_times_are_skipped_for_the_reference_lap():
    msgs = [{"message": "CAR 1 (VER) TIME 1:10.166 DELETED - TRACK LIMITS AT TURN 4 LAP 12 14:03:22"},
            {"message": "GREEN LIGHT - PIT EXIT OPEN"}]
    deleted = openf1.deleted_laps(msgs)
    assert deleted == {(1, 70.166)}
    laps = pd.DataFrame({"driver_number": [1, 16, 4], "lap_duration": [70.166, 70.3, 69.9],
                         "is_pit_out_lap": [False, False, True]})
    assert int(openf1.fastest_valid_lap(laps, deleted)["driver_number"]) == 16


def test_meeting_is_matched_by_race_date():
    meetings = [{"meeting_key": 1, "date_end": "2026-05-24T22:00:00+00:00", "is_cancelled": False},
                {"meeting_key": 2, "date_end": "2026-06-07T15:00:00+00:00", "is_cancelled": False}]
    assert openf1.match_meeting(meetings, pd.Timestamp("2026-06-07"))["meeting_key"] == 2
    with pytest.raises(LookupError):
        openf1.match_meeting(meetings, pd.Timestamp("2026-08-01"))


def test_source_comparison_flags_disagreement():
    ref = pd.DataFrame({"event": ["A", "B"], "clean_laps": [200, 300], "clip_share": [0.95, 0.90],
                        "clip_frac_mean": [0.35, 0.30], "drop_kmh_median": [15.0, 12.0]})
    new = ref.assign(clip_share=[0.94, 0.80])
    out = compare(ref, new, MULTI_CHECKS)
    assert out.loc[(out.event == "A") & (out.check == "clip_share"), "passed"].item()
    assert not out.loc[(out.event == "B") & (out.check == "clip_share"), "passed"].item()
    lt = pd.DataFrame({"event": ["A"], "measurable_share": [0.9], "valid": [True], "t_loss_median_s": [0.25]})
    assert compare(lt, lt.assign(t_loss_median_s=[0.26]), LAPTIME_CHECKS, {"A"})["passed"].all()
    assert not compare(lt, lt.drop(index=0), LAPTIME_CHECKS, {"A"})["passed"].all()


def test_unknown_source_is_rejected(monkeypatch):
    from erslens.ingest import source

    monkeypatch.setenv("ERSLENS_SOURCE", "f1tv")
    with pytest.raises(ValueError):
        source()
    monkeypatch.delenv("ERSLENS_SOURCE")
    assert source() == "fastf1"


def test_jolpica_lap_times_parse():
    assert openf1.lap_seconds("1:23.046") == 83.046
    assert openf1.lap_seconds("59.912") == 59.912


def test_race_with_an_incomplete_openf1_lap_table_is_refused(monkeypatch):
    openf1.race_laps.cache_clear()
    openf1.KEY_EVENT[777] = (2026, pd.Timestamp("2026-03-08"))
    few = [{"driver_number": 1, "lap_number": 1, "date_start": T0.isoformat(), "lap_duration": None, "is_pit_out_lap": True}]
    monkeypatch.setattr(openf1, "get", lambda *a, **k: few)
    jl = pd.DataFrame({"driver_number": [1] * 10, "lap_number": range(1, 11), "seconds": [80.0] * 10, "pit_in": [False] * 10})
    monkeypatch.setattr(openf1, "jolpica_laps", lambda year, date: jl)
    with pytest.raises(LookupError):
        openf1.race_laps(777)
    monkeypatch.setattr(openf1, "jolpica_laps", lambda year, date: jl.head(1))
    assert len(openf1.race_laps(777)) == 1
    openf1.race_laps.cache_clear()
