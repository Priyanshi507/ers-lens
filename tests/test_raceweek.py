import json
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, "scripts")
import raceweek  # noqa: E402
from raceweek import add_judgement, decide, right, verdict, write_new  # noqa: E402


def session(meeting, name, start, end):
    return {"meeting_name": meeting, "session_name": name, "date_start": start, "date_end": end}


WEEK = [session("Qatar Grand Prix", "Qualifying", "2026-11-28T18:00:00+00:00", "2026-11-28T19:00:00+00:00"),
        session("Qatar Grand Prix", "Race", "2026-11-29T16:00:00+00:00", "2026-11-29T18:00:00+00:00"),
        session("Las Vegas Grand Prix", "Qualifying", "2026-11-21T04:00:00+00:00", "2026-11-21T05:00:00+00:00"),
        session("Las Vegas Grand Prix", "Race", "2026-11-22T04:00:00+00:00", "2026-11-22T06:00:00+00:00")]


def at(text):
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def test_predicts_between_qualifying_and_the_race():
    assert decide(WEEK, at("2026-11-28T20:00:00+00:00"), lambda e: None) == ("predict", "Qatar Grand Prix")


def test_never_predicts_once_the_race_has_started():
    assert decide(WEEK, at("2026-11-29T16:30:00+00:00"), lambda e: None) == ("missed", "Qatar Grand Prix")


def test_judges_three_hours_after_the_race():
    state = lambda e: {"status": "predicted"}
    assert decide(WEEK, at("2026-11-29T20:00:00+00:00"), state) == ("wait", "Qatar Grand Prix")
    assert decide(WEEK, at("2026-11-29T21:00:00+00:00"), state) == ("judge", "Qatar Grand Prix")


def test_does_nothing_when_judged_or_dropped():
    assert decide(WEEK, at("2026-12-01T00:00:00+00:00"), lambda e: {"status": "predicted", "judged_utc": "x"})[0] == "wait"
    assert decide(WEEK, at("2026-12-01T00:00:00+00:00"), lambda e: {"status": "dropped"})[0] == "wait"


def test_waits_before_any_qualifying():
    assert decide(WEEK, at("2026-11-20T00:00:00+00:00"), lambda e: None) == ("wait", None)


def test_verdict_and_judgement_rules():
    assert verdict(3.0) == "clip" and verdict(2.99) == "no clip"
    assert right("clip", 0.5) and not right("clip", 0.49)
    assert right("no clip", 0.18) and not right("no clip", 0.6)


def test_a_prediction_file_is_never_rewritten(tmp_path):
    p = tmp_path / "x.json"
    write_new(p, {"status": "predicted", "call": "clip"})
    with pytest.raises(FileExistsError):
        write_new(p, {"status": "predicted", "call": "no clip"})
    add_judgement(p, {"judged_utc": "t", "right": True})
    with pytest.raises(FileExistsError):
        add_judgement(p, {"judged_utc": "t2", "right": False})
    assert json.loads(p.read_text())["call"] == "clip"


def test_scoreboard_reads_the_committed_predictions():
    table = raceweek.scoreboard(2026)
    mexico = table[table.event == "Mexico City Grand Prix"].iloc[0]
    assert mexico.call == "no clip" and bool(mexico.right)
    assert "Las Vegas Grand Prix" in set(table.event)
