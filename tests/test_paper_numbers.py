import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import paper_numbers


def test_sign_test_matches_hand_value():
    s = paper_numbers.sign_test(6, 7)
    assert s["p"] == pytest.approx(16 / 128)


def test_paper_numbers_match_the_record():
    n = paper_numbers.numbers()
    assert (n["C1"]["k"], n["C1"]["n"]) == (15, 18)
    assert (n["LC2"]["k"], n["LC2"]["n"]) == (17, 17)
    assert (n["Y2"]["k"], n["Y2"]["n"]) == (13, 16)
    assert (n["QL4"]["k"], n["QL4"]["n"]) == (6, 7)
    assert round(n["Y1"][0], 2) == 0.58 and n["Y1"][1] < 0.6 < n["Y1"][2]
    assert round(n["QL1"][0], 2) == 0.45
    assert n["forecast"] == {"less": 14, "no forecast": 4}
