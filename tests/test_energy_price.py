import sys

import pandas as pd

sys.path.insert(0, "scripts")
from energy_price import lower_limit_events, verdicts  # noqa: E402


def frame(price, lap, clip, ok=True):
    return pd.DataFrame({"event": list("abcd"), "cap_kw": 350.0, "converged": ok, "price_s_per_mj": price,
                         "lap_s": lap, "superclip_mj": clip}).set_index("event")


def test_verdicts_on_a_case_worked_by_hand():
    hi = frame([0.4, 0.1, 0.2, 0.3], [80.0, 80.0, 80.0, 80.0], [3.0, 0.5, 1.0, 2.0])
    lo = frame([0.6, 0.2, 0.3, 0.5], [80.25, 80.06, 80.12, 80.5], [3.0, 0.5, 1.0, 2.0])
    v = verdicts(hi, lo, {"a"})
    assert v["EP1"] == (3, 4, True)
    assert v["EP2"][2:] == (1, 1, True)
    assert v["EP3"] == (1.0, True)


def test_a_failed_solve_counts_against_ep1():
    hi = frame([0.4] * 4, [80.0] * 4, [1.0, 2.0, 3.0, 4.0])
    lo = frame([0.6] * 4, [80.25] * 4, [1.0, 2.0, 3.0, 4.0], ok=False)
    assert verdicts(hi, lo, set())["EP1"] == (0, 4, False)


def test_the_lowered_circuits_come_from_the_rules_file():
    assert lower_limit_events("configs/rules_2026_races.yaml") == {
        "Australia", "Canada", "Austria", "British Grand Prix", "Italian Grand Prix", "Las Vegas Grand Prix"}
