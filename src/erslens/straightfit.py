"""Fit a 3-parameter energy strategy to the speed trace of one full-throttle straight.

On a straight at full throttle the car's acceleration is set by ICE power plus electric
power, minus drag, rolling resistance and gravity. With ICE power and drag pinned to
nominal values, the electric power profile is identifiable. It is modelled as deploying a
fraction D of the MGU-K cap until clipping starts at distance s_c, then harvesting at
power H, joined by a smooth step so the fit is differentiable.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

from .params import CarParams
from .physics import G

jax.config.update("jax_enable_x64", True)

DS = 5.0
STEP_WIDTH_M = 20.0
H_SCALE_W = 1e5


@dataclass
class StraightFit:
    deploy_frac: float      # D
    clip_start_m: float     # s_c, from the start of the window
    harvest_w: float        # H
    rmse_kmh: float
    swing_w: float          # electric power drop at the clip point
    energy_used_kj: float   # net battery energy over the window (deploy minus harvest)


def _cap(v, car: CarParams):
    frac = (car.mguk_taper_end_kmh - v * 3.6) / (car.mguk_taper_end_kmh - car.mguk_taper_start_kmh)
    return car.mguk_power_w * jnp.clip(frac, 0.0, 1.0)


def _unpack(theta, length_m):
    d = jax.nn.sigmoid(theta[0])
    s_c = length_m * jax.nn.sigmoid(theta[1])
    h = H_SCALE_W * jax.nn.softplus(theta[2])
    return d, s_c, h


def simulate_straight(theta, v0, s, grade, mass, car: CarParams, ice_w, cda):
    """Speed at each grid point s (m) given strategy parameters; returns (v, electric power)."""
    d, s_c, h = _unpack(theta, s[-1])
    on = 1.0 - jax.nn.sigmoid((s - s_c) / STEP_WIDTH_M)

    def step(v, x):
        on_k, g_k = x
        p_e = on_k * d * _cap(v, car) - (1.0 - on_k) * h
        f = jnp.minimum((ice_w + p_e) / v, car.traction_max_n)
        a = (f - 0.5 * car.rho * cda * v * v - car.crr * mass * G - mass * G * g_k) / mass
        v_next = jnp.sqrt(jnp.maximum(v * v + 2.0 * a * DS, 25.0))
        return v_next, (v, p_e)

    _, (v, p_e) = jax.lax.scan(step, v0, (on, grade))
    return v, p_e


@jax.jit
def _fit_core(theta0, v_obs, w, grade, mass, ice_w, cda, car_vec):
    car = CarParams(*car_vec)
    s = jnp.arange(v_obs.shape[0]) * DS

    def loss(theta):
        v, _ = simulate_straight(theta, v_obs[0], s, grade, mass, car, ice_w, cda)
        return jnp.sum(w * (v - v_obs) ** 2) / jnp.sum(w)

    res = jax.vmap(lambda t0: minimize(loss, t0, method="BFGS"))(theta0)
    i = jnp.argmin(res.fun)
    return res.x[i], res.fun[i]


STARTS = jnp.asarray([(1.0, 0.0, -1.0), (2.0, 1.0, 0.5), (0.0, -1.0, 1.0)])


def fit_straight(v_obs: np.ndarray, mask: np.ndarray, grade: np.ndarray, mass: float,
                 car: CarParams, ice_w: float | None = None, cda: float | None = None) -> StraightFit:
    """v_obs in m/s on a 5 m grid from the start of the straight; mask marks usable points.

    Compiled once per straight length, so fitting many laps of one circuit is fast.
    """
    ice_w = car.ice_power_w if ice_w is None else ice_w
    cda = car.cda_straight if cda is None else cda
    car_vec = tuple(float(x) for x in car.__dict__.values())
    w = jnp.asarray(mask, float)
    theta, mse = _fit_core(STARTS, jnp.asarray(v_obs, float), w, jnp.asarray(grade, float),
                           float(mass), float(ice_w), float(cda), car_vec)
    s = jnp.arange(len(v_obs)) * DS
    d, s_c, h = (float(x) for x in _unpack(theta, s[-1]))
    v, p_e = simulate_straight(theta, jnp.asarray(v_obs[0], float), s, jnp.asarray(grade, float),
                               mass, car, ice_w, cda)
    dt = DS / v
    used = jnp.sum(jnp.where(p_e > 0, p_e / car.eta_deploy, p_e * car.eta_harvest) * dt * w)
    v_clip = float(jnp.interp(s_c, s, v))
    return StraightFit(d, s_c, h, float(jnp.sqrt(mse)) * 3.6,
                       d * float(_cap(v_clip, car)) + h, float(used) / 1e3)
