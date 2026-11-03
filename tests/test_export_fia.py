import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
from export_fia import build  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return build(Path("results/fia_fix_calendar_fixed"), Path("results/fia_fix_fixed"))


def test_every_circuit_has_both_caps(data):
    assert len(data["circuits"]) == 18
    for c in data["circuits"]:
        assert set(c["caps"]) == {"250", "350"}
        n = len(c["caps"]["250"]["v_kmh"])
        assert all(len(c["caps"][cap][k]) == n for cap in ("250", "350") for k in ("v_kmh", "superclip_kw"))
        a, b = c["straight"]
        assert 0 <= a < b <= n


def test_verdicts_match_the_research_log(data):
    got = {p["id"]: (p["count"], p["total"], p["held"]) for p in data["predictions"]}
    assert got == {"C1": (17, 18, True), "C2": (14, 18, False), "C3": (18, 18, True), "F1": (10, 12, False),
                   "F2": (12, 12, True), "F3": (12, 12, True), "F4": (4, 4, True)}
    assert data["straight_down"] == {"count": 16, "total": 18}


def test_published_data_is_current(data):
    published = json.loads(Path("site/data/fia.json").read_text())
    published.pop("generated"), data.pop("generated")
    assert published == json.loads(json.dumps(data))


def test_no_circuit_uses_an_assumed_limit(data):
    assert not [c["label"] for c in data["circuits"] if c["assumed_recharge_mj"]]
