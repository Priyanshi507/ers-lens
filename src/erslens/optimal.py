"""Lap-time-optimal electric energy deployment by dynamic programming (method M1).

State: speed and battery state of charge at each track point. Decision: the electric power
command (positive deploys, negative harvests from the engine). Transitions use
jaxsim._step itself, so the optimizer and the simulator share one physics model, and every
optimal strategy can be replayed in the simulator to check the optimizer's own lap time.

The lap starts at the slowest point of the circuit, where speed is set by grip, so the lap is
steady-state in speed. Energy neutrality is a hard terminal constraint: the battery must end
the lap at least as charged as it started. The per-lap harvest limit is enforced with a
Lagrange multiplier found by bisection.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from .jaxsim import V_MIN, _step, params_of, simulate_laps, static_track
from .params import CarParams
from .track import Track
from .physics import braking_envelope

MISS_PENALTY_S_PER_J = 1e-3   # ending the lap 1 kJ short costs 1 s: never worth it


@dataclass(frozen=True)
class Grid:
    v_ms: np.ndarray
    soc_j: np.ndarray
    cmd_w: np.ndarray


def default_grid(car: CarParams, n_soc: int = 81) -> Grid:
    deploy = np.arange(0.0, car.mguk_power_w + 1.0, 25e3)
    harvest = -np.array([75e3, 150e3, 250e3, car.mguk_power_w])
    return Grid(np.arange(V_MIN, 101.0, 1.0), np.linspace(0.0, car.es_capacity_j, n_soc),
                np.concatenate([harvest[::-1], deploy]))


@dataclass
class Solution:
    track: Track
    car: CarParams
    cmd_w: np.ndarray
    soc0_j: float
    harvest_price_s_per_j: float
    lap_time_dp_s: float
    lap_time_sim_s: float
    out: dict

    @property
    def harvested_j(self) -> float:
        return float((self.out["harvest"] * self.out["dt"]).sum() * self.car.eta_harvest)

    @property
    def soc_end_j(self) -> float:
        o = self.out
        return float(o["soc"][-1] + (self.car.eta_harvest * o["harvest"][-1]
                                     - o["deploy"][-1] / self.car.eta_deploy) * o["dt"][-1])


def race_car(car: CarParams) -> CarParams:
    """Mid-race fuel load, constant over the lap."""
    return car.with_(fuel_kg=car.fuel_kg / 2.0, fuel_per_lap_kg=0.0)


def start_at_slowest_point(track: Track, car: CarParams) -> Track:
    k = int(np.argmin(braking_envelope(track, car)))
    roll = lambda a: None if a is None else np.roll(a, -k)
    return Track(track.name, track.distance_m, roll(track.v_limit_ms), roll(track.straight_mode),
                 roll(track.elevation_m))


def _transition(p, car, ds, mass, V, S, U, target, straight):
    carry = (V[:, None, None], S[None, :, None], 0.0)
    (v_next, s_next, e_in), (_, _, _, _, dt) = _step(p, car, ds, mass, carry, (U[None, None, :], target, straight))
    return v_next, s_next, e_in, dt


def _interp(value, grid: Grid, v, s):
    iv = (v - grid.v_ms[0]) / (grid.v_ms[1] - grid.v_ms[0])
    i_s = s / (grid.soc_j[1] - grid.soc_j[0])
    shape = v.shape
    return map_coordinates(value, [iv.ravel(), i_s.ravel()], order=1, mode="nearest").reshape(shape)


class _Compiled:
    """Backward DP, forward rollout and simulator replay, compiled once per circuit and car."""

    def __init__(self, track: Track, car: CarParams, grid: Grid, terminal: bool):
        unlimited = car.with_(harvest_per_lap_j=np.inf)
        st = static_track(track, car)
        p, mass = params_of(car), car.mass_kg + car.fuel_kg
        V, S, U = (jnp.asarray(a) for a in (grid.v_ms, grid.soc_j, grid.cmd_w))
        self.v0 = float(st.env[-1])

        def backward(soc0, price):
            short = jnp.maximum(0.0, soc0 - S)[None, :] * jnp.ones((len(V), 1))
            v_end = MISS_PENALTY_S_PER_J * short if terminal else jnp.zeros((len(V), len(S)))

            def back(value_next, x):
                target, straight = x
                v_n, s_n, e_in, dt = _transition(p, unlimited, st.ds, mass, V, S, U, target, straight)
                value = (dt + price * e_in + _interp(value_next, grid, v_n, s_n)).min(axis=-1)
                return value, value

            _, values = jax.lax.scan(back, v_end, (st.env, st.straight), reverse=True)
            return values, v_end

        def forward(values, v_end, soc0, price):
            nxt = jnp.concatenate([values[1:], v_end[None]], axis=0)

            def fwd(state, x):
                v, s = state
                target, straight, value_next = x
                (v_n, s_n, e_in), (_, _, _, _, dt) = _step(p, unlimited, st.ds, mass, (v, s, 0.0),
                                                            (U, target, straight))
                j = jnp.argmin(dt + price * e_in + _interp(value_next, grid, v_n, s_n))
                return (v_n[j], s_n[j]), U[j]

            _, cmd = jax.lax.scan(fwd, (jnp.asarray(self.v0), soc0), (st.env, st.straight, nxt))
            start = _interp(values[0], grid, jnp.asarray([self.v0]), jnp.asarray([soc0]))[0]
            return cmd, start

        def replay(cmd, soc0):
            out = simulate_laps(p, cmd, st, car, 1, soc0 / car.es_capacity_j, self.v0)
            return {k: a[0] for k, a in out.items()}

        self.backward, self.forward, self.replay = jax.jit(backward), jax.jit(forward), jax.jit(replay)


def solve(track: Track, car: CarParams, grid: Grid | None = None, soc0_fracs=(0.25, 0.5, 0.75),
          terminal: bool = True, max_bisect: int = 12) -> Solution:
    """Optimal one-lap strategy for a mid-race car, with the harvest limit enforced.

    lap_time_dp_s is the DP's own prediction of the lap time (its optimal cost minus the harvest
    price term), to be compared with lap_time_sim_s from replaying the strategy in the simulator.
    """
    car = race_car(car)
    track = start_at_slowest_point(track, car)
    grid = grid or default_grid(car)
    c = _Compiled(track, car, grid, terminal)

    def attempt(soc0, price):
        values, v_end = c.backward(soc0, price)
        cmd, start_cost = c.forward(values, v_end, soc0, price)
        out = {k: np.asarray(a) for k, a in c.replay(cmd, soc0).items()}
        harvested = float((out["harvest"] * out["dt"]).sum() * car.eta_harvest)
        return Solution(track, car, np.asarray(cmd), soc0, price,
                        float(start_cost) - price * harvested, float(out["dt"].sum()), out), harvested

    best = None
    for frac in soc0_fracs:
        soc0 = frac * car.es_capacity_j
        sol, harvested = attempt(soc0, 0.0)
        if harvested > car.harvest_per_lap_j:
            lo, hi = 0.0, 1e-6
            while attempt(soc0, hi)[1] > car.harvest_per_lap_j:
                hi *= 4.0
            for _ in range(max_bisect):
                mid = 0.5 * (lo + hi)
                (lo, hi) = (mid, hi) if attempt(soc0, mid)[1] > car.harvest_per_lap_j else (lo, mid)
            sol, _ = attempt(soc0, hi)
        if best is None or sol.lap_time_sim_s < best.lap_time_sim_s:
            best = sol
    return best
