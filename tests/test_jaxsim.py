import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from erslens.jaxsim import lap_times, params_of, simulate_laps, static_track
from erslens.params import CarParams
from erslens.policies import ProfilePolicy
from erslens.simulate import lap_times as np_lap_times
from erslens.simulate import simulate
from erslens.track import demo_circuit

CAR = CarParams()
TRACK = demo_circuit()
CMD = np.where(TRACK.straight_mode, 350e3, -150e3)
ST = static_track(TRACK, CAR)


def test_matches_numpy_simulator():
    ref = simulate(TRACK, CAR, ProfilePolicy(CMD), n_laps=3, soc0_frac=0.5)
    out = simulate_laps(params_of(CAR), jnp.asarray(CMD), ST, CAR, 3, 0.5)
    np.testing.assert_allclose(np.asarray(lap_times(out)), np_lap_times(ref).to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(out["v"]).reshape(-1), ref["speed_kmh"].to_numpy() / 3.6,
                               atol=1e-9)


def test_gradients_match_finite_differences():
    f = jax.jit(lambda p: lap_times(simulate_laps(p, jnp.asarray(CMD), ST, CAR, 2, 0.5)).sum())
    p = params_of(CAR)
    g = jax.grad(f)(p)
    for k in ("cda_straight", "ice_power_w", "eta_deploy"):
        h = 1e-6 * float(p[k])
        fd = (float(f({**p, k: p[k] + h})) - float(f({**p, k: p[k] - h}))) / (2 * h)
        assert float(g[k]) == pytest.approx(fd, rel=1e-4)


def test_batched_strategies():
    batch = jax.vmap(lambda c: lap_times(simulate_laps(params_of(CAR), c, ST, CAR, 1, 0.5)))
    scales = jnp.linspace(0.0, 1.0, 8)
    times = np.asarray(batch(jnp.asarray(CMD)[None, :] * scales[:, None]))[:, 0]
    assert times.shape == (8,)
    # Deploying more on the straights should never make the lap slower here.
    assert np.all(np.diff(times) <= 1e-9)


def test_straight_fit_recovers_known_strategy():
    from erslens.straightfit import DS, _unpack, fit_straight, simulate_straight

    rng = np.random.default_rng(0)
    n = 240
    s = jnp.arange(n) * DS
    errs = []
    for _ in range(5):
        d, sc, h = rng.uniform(0.3, 1.0), rng.uniform(0.3, 0.8), rng.uniform(5e4, 3e5)
        theta = jnp.array([np.log(d / (1 - d)), np.log(sc / (1 - sc)), np.log(np.expm1(h / 1e5))])
        v, _ = simulate_straight(theta, 60.0, s, jnp.zeros(n), 800.0, CAR, CAR.ice_power_w, CAR.cda_straight)
        v_obs = np.round(np.asarray(v) * 3.6 + rng.normal(0, 0.8, n)) / 3.6
        f = fit_straight(v_obs, np.ones(n), np.zeros(n), 800.0, CAR)
        errs.append((abs(f.deploy_frac - d), abs(f.clip_start_m - sc * float(s[-1])), abs(f.harvest_w - h)))
    errs = np.array(errs).mean(axis=0)
    assert errs[0] < 0.03
    assert errs[1] < 15
    assert errs[2] < 15e3
