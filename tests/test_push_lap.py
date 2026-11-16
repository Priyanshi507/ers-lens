import sys

import numpy as np
import pandas as pd

from erslens.optimal import solve
from erslens.params import load_car_params
from erslens.track import demo_circuit

sys.path.insert(0, "scripts")
from quali_test import verdicts  # noqa: E402

CAR = load_car_params("configs/car_2026.yaml").with_(superclip_max_w=350e3)


def test_push_lap_spends_the_battery_and_is_faster():
    neutral = solve(demo_circuit(), CAR, soc0_fracs=(0.5,))
    push = solve(demo_circuit(), CAR, soc0_fracs=(1.0,), terminal=False)
    assert push.soc0_j == CAR.es_capacity_j and push.soc_end_j < 0.5 * push.soc0_j
    assert neutral.soc_end_j >= neutral.soc0_j - 5e3
    assert push.lap_time_sim_s < neutral.lap_time_sim_s
    assert float((np.asarray(push.out["harvest"]) * np.asarray(push.out["dt"])).sum()) * CAR.eta_harvest <= CAR.harvest_per_lap_j * 1.001


def test_verdicts_follow_the_written_bars():
    t = pd.DataFrame({"model_push": [1.0, 5.0, 10.0, 20.0], "model_neutral": [20.0, 10.0, 5.0, 1.0],
                      "model_race": [3.0, 9.0, 8.0, 30.0], "real_quali": [1.0, 6.0, 9.0, 18.0],
                      "real_race": [4.0, 6.5, 5.0, 25.0], "quali_clip_share": [0.2, 0.9, 0.9, 1.0]})
    v = verdicts(t)
    assert all(held for held, _ in v.values())
    assert v["QL4"][1] == "sign right at 3 of 3 races"
    worse = verdicts(t.assign(model_push=t.model_neutral))
    assert not worse["QL1"][0] and not worse["QL3"][0]
