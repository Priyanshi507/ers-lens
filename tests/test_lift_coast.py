import jax.numpy as jnp
import numpy as np
import pytest

from erslens.jaxsim import COAST_BASE, G, _step, params_of
from erslens.optimal import default_grid, dp_car, solve
from erslens.params import load_car_params
from erslens.track import demo_circuit

CAR = load_car_params("configs/car_2026.yaml")


def one_step(car, cmd, v=85.0, target=200.0, straight=True, soc=2e6):
    carry = (jnp.asarray(v), jnp.asarray(soc), jnp.asarray(0.0))
    (v_next, _, e_in), out = _step(params_of(car), car, 5.0, car.mass_kg, carry, (jnp.asarray(cmd), target, straight))
    return float(v_next), float(e_in), [float(o) for o in out]


def test_coasting_closes_the_aero_and_harvests_with_the_mguk():
    h = 250e3
    v_next, e_in, (_, _, deploy, harvest, dt, superclip, coast) = one_step(CAR, COAST_BASE - h)
    resist = 0.5 * CAR.rho * CAR.cda_corner * 85.0 ** 2 + CAR.crr * CAR.mass_kg * G
    assert v_next == pytest.approx(np.sqrt(85.0 ** 2 - 2.0 * (h / 85.0 + resist) / CAR.mass_kg * 5.0))
    assert (deploy, superclip) == (0.0, 0.0) and coast == harvest == h
    assert e_in == pytest.approx(CAR.eta_harvest * h * dt)


def test_coast_is_capped_by_the_mguk_and_falls_back_to_braking_at_the_envelope():
    _, _, out = one_step(CAR, COAST_BASE - 1e7)
    assert out[6] == CAR.mguk_power_w
    v_next, _, out = one_step(CAR, COAST_BASE - 75e3, target=60.0)
    assert v_next == 60.0 and out[6] == 0.0


def test_existing_commands_are_untouched():
    for cmd in (-350e3, -75e3, 0.0, 200e3):
        _, _, out = one_step(CAR, cmd)
        assert out[6] == 0.0
    assert not (default_grid(CAR).cmd_w <= COAST_BASE).any()
    coast = default_grid(CAR.with_(lift_coast=True)).cmd_w
    assert sorted(COAST_BASE - coast[coast <= COAST_BASE]) == [75e3, 150e3, 250e3, CAR.mguk_power_w]


def test_more_choices_never_make_the_lap_slower():
    car = CAR.with_(superclip_max_w=250e3)
    off, on = (solve(demo_circuit(), car.with_(lift_coast=f), soc0_fracs=(0.5,)) for f in (False, True))
    assert on.lap_time_sim_s <= off.lap_time_sim_s + abs(off.gap_or_nan) + abs(on.gap_or_nan)  # coarse grid: either sign
    assert float(np.asarray(on.out["coast"]).max()) > 0.0
    assert float(np.asarray(off.out["coast"]).max()) == 0.0
    assert dp_car(car).lift_coast is False
