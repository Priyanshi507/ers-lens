import sys

import numpy as np

sys.path.insert(0, "scripts")
from identifiability_demo import as_telemetry, battery_energy_mj, drive  # noqa: E402

from erslens.energy import electric_power_swing
from erslens.params import CarParams
from erslens.physics import mguk_cap_kmh

CAR = CarParams()
MASS = CAR.mass_kg + CAR.fuel_kg / 2


def p_a(t, v):
    return 0.6 * float(mguk_cap_kmh(v * 3.6, CAR)) if t < 8.0 else -150e3


def test_different_cars_produce_identical_speed_but_different_battery_use():
    a, delta = -40e3, 0.05
    ta, sa, va, pa = drive(CAR.ice_power_w, CAR.cda_straight, p_a, CAR, MASS)
    tb, sb, vb, pb = drive(CAR.ice_power_w + a, CAR.cda_straight + delta,
                           lambda t, v: p_a(t, v) - a + 0.5 * CAR.rho * delta * v ** 3, CAR, MASS)
    assert np.abs(va - vb).max() < 1e-9
    for p, v in ((pa, va), (pb, vb)):
        assert np.all(p <= mguk_cap_kmh(v * 3.6, CAR) + 1.0)
        assert np.all(p >= -CAR.mguk_power_w - 1.0)
    ea = battery_energy_mj(pa, CAR.eta_deploy, CAR.eta_harvest)
    eb = battery_energy_mj(pb, CAR.eta_deploy, CAR.eta_harvest)
    assert abs(eb - ea) > 0.3


def test_matched_speed_estimator_is_invariant_to_the_unidentifiable_part():
    a, delta = -40e3, 0.05
    ta, sa, va, _ = drive(CAR.ice_power_w, CAR.cda_straight, p_a, CAR, MASS)
    tb, sb, vb, _ = drive(CAR.ice_power_w + a, CAR.cda_straight + delta,
                          lambda t, v: p_a(t, v) - a + 0.5 * CAR.rho * delta * v ** 3, CAR, MASS)
    sw_a = electric_power_swing(as_telemetry(ta, sa, va), 0, 1e6, MASS)
    sw_b = electric_power_swing(as_telemetry(tb, sb, vb), 0, 1e6, MASS)
    assert abs(sw_a["swing_w"] - sw_b["swing_w"]) < 1.0
