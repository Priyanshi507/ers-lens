"""Lap-time-optimal energy deployment by gradient descent through the JAX simulator (method M2).

Independent of the dynamic-programming method M1: it never starts from M1's answer, so agreement
between the two is evidence that both are right. The command at every track point is free,
mapped through tanh into +/- the MGU-K power; the simulator applies the speed taper, battery
window and per-lap harvest limit.

Energy neutrality uses an augmented Lagrangian: a learned energy price mu (s/MJ) plus a squared
penalty. A pure penalty only punishes ending short, so surplus energy earns nothing and the
gradient never finds trades such as harvesting in a corner to deploy on the next straight; the
price makes every harvested joule count.

The battery window's hard limits have zero gradient once the battery is empty or full, so the
search runs with them removed (hard_soc=False) and penalizes leaving the window and ending the
lap short instead. The final strategy is replayed with the real limits, and only that replay is
reported.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .jaxsim import params_of, simulate_laps, static_track
from .optimal import Solution, race_car, start_at_slowest_point
from .params import CarParams
from .track import Track

PENALTY_S_PER_MJ2 = 200.0


def soc_end(out: dict, car: CarParams):
    gain = car.eta_harvest * out["harvest"][-1] - out["deploy"][-1] / car.eta_deploy
    return jnp.clip(out["soc"][-1] + gain * out["dt"][-1], 0.0, car.es_capacity_j)


def solve_gradient(track: Track, car: CarParams, soc0_frac: float = 0.5, rounds: int = 15,
                   steps_per_round: int = 300, learning_rate: float = 0.05,
                   init_cmd_w: float | np.ndarray = -50e3) -> Solution:
    """init_cmd_w: a constant, or a command per point of the track as passed in (for example a
    tuned simple policy from simple.tune, which uses the same rotated track)."""
    car = race_car(car)
    track = start_at_slowest_point(track, car)
    st = static_track(track, car)
    p, soc0 = params_of(car), soc0_frac * car.es_capacity_j
    v0 = float(st.env[-1])
    scale = car.mguk_power_w

    def run(theta, hard_soc=True):
        out = simulate_laps(p, scale * jnp.tanh(theta), st, car, 1, soc0_frac, v0, hard_soc)
        return {k: a[0] for k, a in out.items()}

    def shortfall_mj(out):
        end = out["soc"][-1] + (car.eta_harvest * out["harvest"][-1]
                                - out["deploy"][-1] / car.eta_deploy) * out["dt"][-1]
        return (soc0 - end) / 1e6

    def loss(theta, mu):
        out = run(theta, hard_soc=False)
        short = shortfall_mj(out)
        window_mj = (jnp.maximum(0.0, -out["soc"].min())
                     + jnp.maximum(0.0, out["soc"].max() - car.es_capacity_j)) / 1e6
        return (out["dt"].sum() + mu * short
                + PENALTY_S_PER_MJ2 * (jnp.maximum(0.0, short) ** 2 + window_mj ** 2))

    opt = optax.adam(learning_rate)
    init = np.broadcast_to(np.asarray(init_cmd_w, dtype=float), (track.n,))
    theta = jnp.arctanh(jnp.clip(jnp.asarray(init) / scale, -0.995, 0.995))
    state = opt.init(theta)

    @jax.jit
    def inner(theta, state, mu):
        def body(_, carry):
            th, st_ = carry
            upd, st_ = opt.update(jax.grad(loss)(th, mu), st_)
            return optax.apply_updates(th, upd), st_
        theta, state = jax.lax.fori_loop(0, steps_per_round, body, (theta, state))
        return theta, state, shortfall_mj(run(theta, hard_soc=False))

    mu = 0.0
    for _ in range(rounds):
        theta, state, short = inner(theta, state, mu)
        mu = max(0.0, mu + 2.0 * PENALTY_S_PER_MJ2 * float(short))

    out = {k: np.asarray(a) for k, a in jax.jit(run)(theta).items()}
    lap = float(out["dt"].sum())
    return Solution(track, car, np.asarray(scale * jnp.tanh(theta)), soc0, 0.0, lap, lap, out)


def solve_m2(track: Track, car: CarParams, soc0_frac: float = 0.5) -> tuple[Solution, "Tuned"]:
    """Method M2 as used for results: refine the best tuned simple policy by gradient descent.

    Starting from a simple policy, not from M1's answer, keeps M2 independent of M1; constant
    starts get trapped in local optima on realistic circuits.
    """
    from .simple import tune

    seed = min(tune(track, car, soc0_frac).values(), key=lambda t: t.lap_time_s)
    return solve_gradient(track, car, soc0_frac, init_cmd_w=seed.cmd_w), seed
