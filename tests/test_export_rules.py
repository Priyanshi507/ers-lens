import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
from export_rules import build  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return build()


def test_every_rule_covers_every_circuit_with_proven_solves(data):
    assert len(data["circuits"]) == 18 and len(data["rules"]) == 14
    events = {c["event"] for c in data["circuits"]}
    for r in data["rules"]:
        assert set(r["rows"]) == events
        assert all(v["proven"] for v in r["rows"].values())


def test_rules_reproduce_the_findings(data):
    rules = {r["id"]: r["rows"] for r in data["rules"]}
    less = lambda a, b, k: sum(rules[b][e][k] < rules[a][e][k] for e in rules[a])
    assert less("race_lc_250", "race_lc_350", "coast_mj") == 17          # LC2
    assert sum(rules["race_350"][e]["superclip_mj"] > rules["race_250"][e]["superclip_mj"] for e in rules["race_250"]) == 15  # C1
    assert all(v["coast_mj"] == 0 for v in rules["race_350"].values())
    assert rules["race_350"]["Monaco Grand Prix"]["drop_kmh"] == 0.0


def test_real_laps_are_the_measured_ones(data):
    assert data["real"]["Canada"] == {"race": {"drop_kmh": 16.0, "clip_share": 0.98, "laps": 629, "cap_kw": 350},
                                      "quali": {"drop_kmh": 10.0, "clip_share": 0.53, "laps": 88, "cap_kw": 350}}
    assert "quali" not in data["real"]["China"] and data["real"]["China"]["race"]["cap_kw"] == 250


def test_published_data_is_current(data):
    published = json.loads(Path("site/data/rules.json").read_text())
    published.pop("generated"), data.pop("generated")
    assert published == json.loads(json.dumps(data))
