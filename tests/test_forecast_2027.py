import sys

import numpy as np
import pytest

sys.path.insert(0, "scripts")
from forecast_2027 import TAPER_START_KMH, car_2027  # noqa: E402


@pytest.mark.parametrize("curve", ["t280", "t300", "t320"])
def test_deployment_cap_starts_tapering_at_the_curve_speed(curve):
    from erslens.physics import mguk_cap_kmh

    car = car_2027("Miami", curve, 375e3)
    start = TAPER_START_KMH[curve]
    cap = lambda kmh: min(float(mguk_cap_kmh(np.array([kmh]), car)[0]), car.deploy_max_w)
    assert cap(start - 10) == pytest.approx(300e3)
    assert cap(start + 10) == pytest.approx(250e3)


def test_2027_car_keeps_the_event_recharge_limit_and_new_powers():
    car = car_2027("Italian Grand Prix", "none", 350e3)
    assert car.harvest_per_lap_j == 7.0e6 and car.superclip_max_w == 350e3
    assert (car.ice_power_w, car.mguk_power_w, car.deploy_max_w, car.es_capacity_j) == (420e3, 375e3, 300e3, 4.5e6)


def test_forecast_rule_gives_a_direction_only_when_every_curve_agrees(tmp_path):
    from pathlib import Path

    import pandas as pd
    from forecast_2027 import EVENTS, SUPERCLIP_W, baseline_drops, forecast

    base = baseline_drops(Path("results/fia_fix_calendar_h"))
    rows = []
    for k, e in enumerate(EVENTS):
        for m, curve in enumerate(TAPER_START_KMH):
            for sc in SUPERCLIP_W:
                shift = {0: -5.0, 1: 5.0}.get(k, -1.0 if m < 3 else 1.0)  # mixed signs from the third circuit on
                rows.append({"event": e, "curve": curve, "superclip_kw": sc / 1e3, "converged": k != 2 or m != 0,
                             "model_drop_kmh": base[e] + shift})
    half = len(rows) // 2
    for name, part in (("a", rows[:half]), ("b", rows[half:])):
        (tmp_path / name).mkdir()
        pd.DataFrame(part).to_csv(tmp_path / name / "runs.csv", index=False)
    t = forecast([tmp_path / "a", tmp_path / "b"], Path("results/fia_fix_calendar_h"))
    assert t.loc[EVENTS[0], "direction"] == "less" and t.loc[EVENTS[1], "direction"] == "more"
    assert (t.loc[list(EVENTS[2:]), "direction"] == "no forecast").all()
    assert "rank_2027" not in t and len(t) == len(EVENTS)
