import numpy as np
import pandas as pd
import pytest

from erslens.params import CarParams
from erslens.strategy import CONVERGED_S, REFINED_TAG, no_taper, resolve_runs, variant_setup


def run(event, variant, car_id, lap, pred, clip=0.0):
    return {"event": event, "variant": variant, "car_id": car_id, "lap_s": lap, "dp_pred_s": pred,
            "clip_ratio": clip}


def test_converged_rows_are_used_and_unconverged_are_excluded():
    runs = pd.DataFrame([run("A", "sweep", "c1", 70.0, 70.01),
                         run("A", "sweep", "c2", 70.0, 70.0 + CONVERGED_S + 0.01)])
    used, excluded = resolve_runs(runs)
    assert list(used.car_id) == ["c1"]
    assert list(excluded.car_id) == ["c2"]


def test_threshold_is_inclusive():
    used, excluded = resolve_runs(pd.DataFrame([run("A", "nominal", "nominal", 70.0, 70.0 + CONVERGED_S)]))
    assert len(used) == 1 and excluded.empty


def test_converged_refinement_replaces_the_original_without_editing_it():
    runs = pd.DataFrame([run("A", "rules_no_taper", "nominal", 70.0, 70.10, clip=0.55),
                         run("A", "rules_no_taper", "nominal" + REFINED_TAG + " dv=0.125 soc=641", 70.02, 70.03, clip=0.40)])
    used, excluded = resolve_runs(runs)
    assert excluded.empty and len(used) == 1
    r = used.iloc[0]
    assert r.car_id == "nominal" and r.refined and r.clip_ratio == 0.40
    assert len(runs) == 2


def test_case_stays_excluded_when_every_refinement_fails():
    runs = pd.DataFrame([run("A", "sweep", "c1", 70.0, 70.2),
                         run("A", "sweep", "c1" + REFINED_TAG + " dv=0.125 soc=641", 70.0, 70.1)])
    used, excluded = resolve_runs(runs)
    assert used.empty and list(excluded.car_id) == ["c1"]


def test_cases_are_kept_apart_by_event_and_variant():
    runs = pd.DataFrame([run("A", "nominal", "nominal", 70, 70), run("B", "nominal", "nominal", 80, 80),
                         run("A", "no_taper", "nominal", 69, 69)])
    used, _ = resolve_runs(runs)
    assert len(used) == 3


def test_m2_and_simple_rows_pass_through():
    runs = pd.DataFrame([run("A", "m2", "nominal", 70.5, 70.5),
                         {"event": "A", "variant": "simple_flat_out", "car_id": "nominal", "lap_s": 72.0,
                          "dp_pred_s": np.nan, "clip_ratio": np.nan}])
    used, excluded = resolve_runs(runs)
    assert sorted(used.variant) == ["m2", "simple_flat_out"] and excluded.empty


def test_variant_setup_rebuilds_each_case():
    car = CarParams()
    sweep_car, frac = variant_setup("sweep", "i2d0m1" + REFINED_TAG + " dv=0.125 soc=641", "Miami", car)
    assert frac == 0.5 and sweep_car.ice_power_w == pytest.approx(car.ice_power_w * 1.1)
    assert sweep_car.cda_straight == pytest.approx(car.cda_straight * 0.8) and sweep_car.mass_kg == car.mass_kg
    assert variant_setup("soc0_0.3", "nominal", "Miami", car) == (car, 0.3)
    nt, _ = variant_setup("rules_no_taper", "nominal", "Miami", car)
    assert nt.harvest_per_lap_j == 9e6 and nt == no_taper(nt)
    d250, _ = variant_setup("rules_deploy250", "nominal", "Canada", car)
    assert d250.deploy_max_w == 250e3
    with pytest.raises(ValueError):
        variant_setup("m2", "nominal", "Miami", car)
