import jax.numpy as jnp
import numpy as np
import pytest

from erslens.jaxsim import params_of, simulate_laps, static_track
from erslens.optimal import race_car, solve, start_at_slowest_point
from erslens.params import CarParams
from erslens.track import toy_track

TRACK = toy_track([("straight", 600, None), ("corner", 100, 90),
                   ("straight", 300, None), ("corner", 80, 140)], name="short")
CAR = CarParams()


def replay_constant(car, cmd_w, soc0_frac):
    rc = race_car(car)
    tr = start_at_slowest_point(TRACK, rc)
    st = static_track(tr, rc)
    out = simulate_laps(params_of(rc), jnp.full(tr.n, cmd_w), st, rc, 1, soc0_frac, float(st.env[-1]))
    return float(out["dt"].sum())


@pytest.fixture(scope="module")
def nominal():
    return solve(TRACK, CAR, soc0_fracs=(0.5,))


def test_strategy_is_energy_neutral_and_respects_harvest_limit(nominal):
    assert nominal.soc_end_j >= nominal.soc0_j - 0.05e6
    assert nominal.harvested_j <= CAR.harvest_per_lap_j * 1.001


def test_dp_prediction_is_close_to_the_replayed_lap(nominal):
    assert abs(nominal.lap_time_dp_s - nominal.lap_time_sim_s) < 0.25


def test_beats_never_deploying(nominal):
    assert nominal.lap_time_sim_s < replay_constant(CAR, 0.0, 0.5) - 0.1


def test_free_energy_means_full_deployment():
    free = CAR.with_(es_capacity_j=1e9, harvest_per_lap_j=1e12)
    sol = solve(TRACK, free, soc0_fracs=(0.5,), terminal=False)
    assert sol.lap_time_sim_s == pytest.approx(replay_constant(free, free.mguk_power_w, 0.5), abs=0.02)


def test_no_energy_means_engine_only_lap_time():
    empty = CAR.with_(es_capacity_j=1.0, harvest_per_lap_j=0.0)
    sol = solve(TRACK, empty, soc0_fracs=(0.5,))
    assert sol.lap_time_sim_s == pytest.approx(replay_constant(empty, 0.0, 0.5), abs=0.02)
