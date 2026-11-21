import sys

import pandas as pd

sys.path.insert(0, "scripts")
from limit_advisor import LIMITS, advise, fia_limits, verdicts  # noqa: E402


def runs(event, clip_at):
    return pd.DataFrame({"event": event, "limit_mj": LIMITS, "ok": True, "converged": True,
                         "superclip_s": [clip_at(mj) for mj in LIMITS]})


def test_the_advised_limit_is_the_largest_within_the_target():
    r = pd.concat([runs("a", lambda mj: 2.0 * (mj - 5.0)), runs("b", lambda mj: 10.0)])
    a = advise(r)
    assert a["a"] == 7.0 and a["b"] == 5.0


def test_verdicts_on_a_case_worked_by_hand():
    r = pd.concat([runs("a", lambda mj: 2.0 * (mj - 5.0)), runs("b", lambda mj: 1.5 * (mj - 5.0)),
                   runs("c", lambda mj: 0.0)])
    v = verdicts(r, {"a": 6.5, "b": 8.0, "c": 9.0})
    assert v["LA1"] == (3, 3, True)
    assert v["LA2"][:2] == (1.0, 3)
    assert v["LA3"][:2] == (2, 3)


def test_a_failed_solve_breaks_monotonicity():
    r = runs("a", lambda mj: mj)
    r.loc[2, "ok"] = False
    assert verdicts(r, {"a": 6.0})["LA1"][0] == 0


def test_fia_limits_come_from_the_rules_file():
    f = fia_limits("configs/rules_2026_races.yaml")
    assert f["Italian Grand Prix"] == 5.0 and f["Monaco Grand Prix"] == 9.0 and len(f) == 20
