import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
from export_strategy import EVENTS, build, parse_verdicts  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return build()


def test_every_car_and_circuit_is_exported(data):
    assert [c["id"] for c in data["cars"]] == ["original", "low", "high"]
    for e in EVENTS:
        cars = data["circuits"][e]["cars"]
        assert set(cars) == {"original", "low", "high"}
        for entry in cars.values():
            assert set(entry["profiles"]) == {"race", "car"}
            assert len(entry["sweep"]) == 27
            for p in entry["profiles"].values():
                n = len(p["v"])
                assert all(len(p[k]) == n for k in ("deploy", "harvest", "cap", "soc"))
                a, b = p["straight"]
                assert 0 <= a < p["peak"] < b <= n
                assert p["converged_gap_s"] <= data["thresholds"]["converged_s"]


def test_profiles_are_charge_neutral_and_within_the_battery(data):
    for e in EVENTS:
        for entry in data["circuits"][e]["cars"].values():
            for p in entry["profiles"].values():
                assert p["soc_end_mj"] >= p["soc"][0] - 1e-3
                assert min(p["soc"]) >= -1e-3 and max(p["soc"]) <= 4.0 + 1e-3
                assert p["harvested_mj"] <= p["harvest_limit_mj"] + 1e-2


def test_published_verdicts_match_the_evaluated_ones(data):
    cars = {c["id"]: c["verdicts"] for c in data["cars"]}
    assert all(v["S1"]["held"] and v["S1 (corrected rules)"]["held"] for v in cars.values())
    assert not cars["high"]["S2"]["held"] and cars["low"]["S2"]["held"]
    assert not any(v[k]["held"] for v in cars.values() for k in ("S5", "S6", "S7 (exploratory)"))


def test_original_car_reproduces_the_logged_diagnostics(data):
    """The 2026-10-10 diagnostics (results/strategy/diagnostics.md) used the original car's profiles."""
    text = Path("results/strategy/diagnostics.md").read_text()
    for e in EVENTS:
        m = re.search(rf"\| {e} \| \d+ / \d+ m \| \d+ / \d+ m \| (\d+) / (\d+) m \|.*\(n=(\d+)\)", text)
        opt_tail, real_tail, n = map(int, m.groups())
        p = data["circuits"][e]["cars"]["original"]["profiles"]["car"]
        real = data["circuits"][e]["real"]
        assert p["peak_to_brake_m"] == opt_tail
        assert real["laps"] == n
        peaks = real["peak_to_brake_m"]
        assert abs(sorted(peaks)[len(peaks) // 2] - real_tail) <= 5
        cap, zero = re.search(rf"\| {e} \| original \| (\d+)% \| (\d+)% \|", text).groups()
        assert round(p["bang_bang"]["cap"] * 100) == int(cap) and round(p["bang_bang"]["zero"] * 100) == int(zero)


def test_real_laps_are_aligned_at_the_braking_point(data):
    for e in EVENTS:
        real = data["circuits"][e]["real"]
        assert real["to_brake_m"][0] == 0 and real["to_brake_m"] == sorted(real["to_brake_m"])
        assert all(lo <= mid <= hi for lo, mid, hi in zip(real["v_q25"], real["v_median"], real["v_q75"]))


def test_a_missing_verdict_row_fails_loudly():
    with pytest.raises(ValueError):
        parse_verdicts("| S1 | held | Australia 0.00 |")


def test_published_file_is_current(data):
    published = json.loads(Path("site/data/strategy.json").read_text())
    published.pop("generated")
    fresh = json.loads(json.dumps(data))
    fresh.pop("generated")
    assert published == fresh, "run python scripts/export_strategy.py and commit site/data/strategy.json"
