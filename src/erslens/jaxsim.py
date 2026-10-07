"""Differentiable lap simulator in JAX, mirroring simulate.simulate for open-loop commands.

The energy strategy is an array of commands per track point, so it can be fitted or
learned by gradient descent, and car parameters are differentiable inputs.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .params import CarParams
from .physics import G, mguk_cap_kmh, braking_envelope
from .track import Track

jax.config.update("jax_enable_x64", True)

V_MIN = 5.0
DIFF_PARAMS = ("cda_straight", "cda_corner", "ice_power_w", "eta_harvest", "eta_deploy")


@dataclass(frozen=True)
class StaticTrack:
    ds: float
    env: jnp.ndarray          # braking envelope at the next point, m/s
    straight: jnp.ndarray     # bool per point


def static_track(track: Track, car: CarParams) -> StaticTrack:
    env = braking_envelope(track, car)
    return StaticTrack(track.ds, jnp.asarray(np.roll(env, -1)), jnp.asarray(track.straight_mode))


def params_of(car: CarParams) -> dict:
    return {k: jnp.asarray(getattr(car, k), dtype=jnp.float64) for k in DIFF_PARAMS}


def _mguk_cap(v, car: CarParams):
    return mguk_cap_kmh(v * 3.6, car, jnp)


def _step(p: dict, car: CarParams, ds: float, mass, carry, x, hard_soc: bool = True):
    """hard_soc=False removes the battery-window limits so gradients survive an empty or full
    battery; only optimizers use it, and they must penalize leaving the window themselves."""
    v, soc, harvested = carry
    cmd, target, straight = x
    cda = jnp.where(straight, p["cda_straight"], p["cda_corner"])
    v_eff = jnp.maximum(v, V_MIN)
    resist = 0.5 * car.rho * cda * v_eff ** 2 + car.crr * mass * G
    cap_k = _mguk_cap(v_eff, car)
    budget = jnp.maximum(0.0, car.harvest_per_lap_j - harvested)
    space = car.es_capacity_j - soc if hard_soc else jnp.inf
    dt_guess = ds / v_eff
    eh, ed = p["eta_harvest"], p["eta_deploy"]

    deploy = jnp.minimum(jnp.maximum(cmd, 0.0), cap_k)
    if hard_soc:
        deploy = jnp.minimum(deploy, soc * ed / dt_guess)
    ice_h = jnp.where(cmd < 0.0,
                      jnp.minimum(jnp.minimum(jnp.minimum(-cmd, cap_k), p["ice_power_w"]),
                                  jnp.minimum(budget, space) / (eh * dt_guess)),
                      0.0)
    p_wheel = p["ice_power_w"] + deploy - ice_h
    f_drive = jnp.minimum(p_wheel / v_eff, car.traction_max_n)
    a = (f_drive - resist) / mass
    v_acc = jnp.sqrt(jnp.maximum(v * v + 2.0 * a * ds, V_MIN ** 2))

    accel = v_acc <= target
    f_needed = mass * (target ** 2 - v ** 2) / (2.0 * ds) + resist
    partial = (~accel) & (f_needed >= 0.0)
    braking = (~accel) & (f_needed < 0.0)
    v_next = jnp.where(accel, v_acc, target)
    p_need = f_needed * v_eff
    deploy = jnp.where(accel, deploy, 0.0)
    ice_h = jnp.where(accel, ice_h,
                      jnp.where(partial, jnp.minimum(ice_h, jnp.maximum(0.0, p["ice_power_w"] - p_need)), 0.0))
    brake_h = jnp.where(braking,
                        jnp.minimum(jnp.minimum(-p_need, cap_k),
                                    jnp.minimum(budget, space) / (eh * dt_guess)),
                        0.0)

    dt = 2.0 * ds / (v + v_next)
    e_in = eh * (ice_h + brake_h) * dt
    soc_next = soc + e_in - deploy / ed * dt
    if hard_soc:
        soc_next = jnp.clip(soc_next, 0.0, car.es_capacity_j)
    out = (v, soc, deploy, ice_h + brake_h, dt)
    return (v_next, soc_next, harvested + e_in), out


def simulate_laps(p: dict, cmd: jnp.ndarray, st: StaticTrack, car: CarParams, n_laps: int,
                  soc0_frac: float = 0.5, v0: float | None = None, hard_soc: bool = True):
    """Returns per-lap outputs stacked as (n_laps, n_points): v, soc, deploy, harvest, dt."""
    v0 = jnp.minimum(st.env[-1], 80.0) if v0 is None else v0

    def lap(carry, lap_idx):
        v, soc = carry
        mass = car.mass_kg + jnp.maximum(0.0, car.fuel_kg - car.fuel_per_lap_kg * lap_idx)
        step = lambda c, x: _step(p, car, st.ds, mass, c, x, hard_soc)
        (v, soc, _), out = jax.lax.scan(step, (v, soc, 0.0), (cmd, st.env, st.straight))
        return (v, soc), out

    _, outs = jax.lax.scan(lap, (jnp.asarray(v0, jnp.float64), soc0_frac * car.es_capacity_j),
                           jnp.arange(n_laps, dtype=jnp.float64))
    v, soc, deploy, harvest, dt = outs
    return {"v": v, "soc": soc, "deploy": deploy, "harvest": harvest, "dt": dt}


def lap_times(out: dict) -> jnp.ndarray:
    return out["dt"].sum(axis=-1)
