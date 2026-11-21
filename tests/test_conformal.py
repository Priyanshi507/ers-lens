import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from conformal import nested, scale, verdicts  # noqa: E402


def test_scale_uses_the_finite_sample_rank():
    assert scale(np.arange(1, 10, dtype=float), 0.9) == 9.0
    assert scale(np.arange(1, 20, dtype=float), 0.9) == 18.0


def fake_fit(train, test):
    return np.full(len(test), train.superclip_s.mean()), np.ones(len(test))


def test_the_held_out_circuit_never_sets_its_own_width():
    d = pd.DataFrame({"event": ["a", "b", "c", "d"], "superclip_s": [0.0, 1.0, 2.0, 100.0]})
    p = nested(d, fake_fit).set_index("event")
    assert p.loc["d", "q"] < 10 and p.loc["a", "q"] > 10


def test_verdicts_on_a_worked_case():
    p = pd.DataFrame({"superclip_s": [0.0, 1.0, 2.0, 10.0], "mean": [0.0] * 4, "half_width": [3.0] * 4})
    assert verdicts(p) == {"CP1": (0.75, False), "CP2": (3.0, True)}
