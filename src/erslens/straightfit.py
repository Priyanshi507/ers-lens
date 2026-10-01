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
    run_start_m: float = 0.0
    run_len_m: float = 0.0
    clip_range_m: float = 0.0  # spread of clip points fitting within 10% of the best


def _cap(v, car: CarParams):
    frac = (car.mguk_taper_end_kmh - v * 3.6) / (car.mguk_taper_end_kmh - car.mguk_taper_start_kmh)
    return car.mguk_power_w * jnp.clip(frac, 0.0, 1.0)


def _unpack(theta, length_m):
    d = jax.nn.sigmoid(theta[0])
    s_c = length_m * jax.nn.sigmoid(theta[1])
    h = H_SCALE_W * jax.nn.softplus(theta[2])
    return d, s_c, h


def _simulate(d, s_c, h, v0, s, grade, mass, car: CarParams, ice_w, cda):
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


def simulate_straight(theta, v0, s, grade, mass, car: CarParams, ice_w, cda):
    """Speed at each grid point s (m) given strategy parameters; returns (v, electric power)."""
    d, s_c, h = _unpack(theta, s[-1])
    return _simulate(d, s_c, h, v0, s, grade, mass, car, ice_w, cda)


N_CLIP = 19
CLIP_FRACS = jnp.linspace(0.05, 0.95, N_CLIP)
NEAR_BEST = 1.10  # clip points fitting within 10% of the best mse form the uncertainty range


def _simulate_fixed_clip(dh, s_c, v0, s, grade, mass, car, ice_w, cda):
    d = jax.nn.sigmoid(dh[0])
    h = H_SCALE_W * jax.nn.softplus(dh[1])
    return _simulate(d, s_c, h, v0, s, grade, mass, car, ice_w, cda)


@jax.jit
def _fit_core(v_obs, w, grade, run_len, mass, ice_w, cda, car_vec):
    """Profile over candidate clip points; D and H are fitted by BFGS at each one."""
    car = CarParams(*car_vec)
    s = jnp.arange(v_obs.shape[0]) * DS

    def best_at(s_c):
        def loss(dh):
            v, _ = _simulate_fixed_clip(dh, s_c, v_obs[0], s, grade, mass, car, ice_w, cda)
            return jnp.sum(w * (v - v_obs) ** 2) / jnp.sum(w)
        res = minimize(loss, jnp.asarray([1.0, -0.5]), method="BFGS")
        return res.x, res.fun

    coarse = CLIP_FRACS * run_len
    dhs, mses = jax.vmap(best_at)(coarse)
    # Refine on a finer grid around the best coarse point; the profile is smooth there.
    k = jnp.argmin(mses)
    step = run_len * (CLIP_FRACS[1] - CLIP_FRACS[0])
    fine = jnp.clip(coarse[k] + jnp.linspace(-step, step, 9), 0.02 * run_len, 0.98 * run_len)
    dhf, msef = jax.vmap(best_at)(fine)
    return jnp.concatenate([dhs, dhf]), jnp.concatenate([mses, msef]), jnp.concatenate([coarse, fine])


def fit_straight(v_obs: np.ndarray, mask: np.ndarray, grade: np.ndarray, mass: float,
                 car: CarParams, ice_w: float | None = None, cda: float | None = None,
                 min_points: int = 40) -> StraightFit | None:
    """v_obs in m/s on a 5 m grid along the straight; mask marks full-throttle points.

    The fit uses the first unbroken full-throttle run, shifted to start at index 0 and
    zero-weighted padding after it, so every lap of a circuit shares one compilation.
    """
    ice_w = car.ice_power_w if ice_w is None else ice_w
    cda = car.cda_straight if cda is None else cda
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    i0 = idx[0]
    i1 = i0
    while i1 + 1 < len(mask) and mask[i1 + 1]:
        i1 += 1
    n_run = i1 - i0 + 1
    if n_run < min_points:
        return None
    n = len(v_obs)
    v = np.full(n, v_obs[i1]); v[:n_run] = v_obs[i0:i1 + 1]
    g = np.zeros(n); g[:n_run] = grade[i0:i1 + 1]
    w = np.zeros(n); w[:n_run] = 1.0
    run_len = (n_run - 1) * DS
    car_vec = tuple(float(x) for x in car.__dict__.values())
    dhs, mses, s_cs = _fit_core(jnp.asarray(v), jnp.asarray(w), jnp.asarray(g), float(run_len),
                                float(mass), float(ice_w), float(cda), car_vec)
    mses, s_cs = np.asarray(mses), np.asarray(s_cs)
    k = int(np.argmin(mses))
    near = s_cs[mses <= mses[k] * NEAR_BEST]
    s = jnp.arange(n) * DS
    vs, p_e = _simulate_fixed_clip(dhs[k], s_cs[k], jnp.asarray(v[0]), s, jnp.asarray(g),
                                   mass, car, ice_w, cda)
    d = float(jax.nn.sigmoid(dhs[k][0]))
    h = float(H_SCALE_W * jax.nn.softplus(dhs[k][1]))
    dt = DS / vs
    wj = jnp.asarray(w)
    used = jnp.sum(jnp.where(p_e > 0, p_e / car.eta_deploy, p_e * car.eta_harvest) * dt * wj)
    v_clip = float(jnp.interp(s_cs[k], s, vs))
    return StraightFit(d, float(i0 * DS + s_cs[k]), h, float(np.sqrt(mses[k])) * 3.6,
                       d * float(_cap(v_clip, car)) + h, float(used) / 1e3,
                       run_start_m=float(i0 * DS), run_len_m=float(run_len),
                       clip_range_m=float(near.max() - near.min()))
