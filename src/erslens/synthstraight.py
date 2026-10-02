"""Synthetic full-throttle straights for the learnability experiment.

Each sample is a speed trace observed like public telemetry (noisy, whole km/h) and the
battery energy actually used. Cars are either identical or varied; strategies are either
a simple 3-parameter family or flexible random profiles within the regulation limits.
Flexible profiles matter: the non-identifiability argument needs deployment that can
absorb any function of speed, which a 3-parameter family cannot represent.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .params import CarParams
from .physics import G, mguk_cap_kmh

jax.config.update("jax_enable_x64", True)

N_POINTS = 240
DS = 5.0
N_BUMPS = 5
SPEED_NOISE_KMH = 0.8
# Realism filters, applied identically in every regime: one straight cannot use more than
# the 4 MJ battery window, and a car at full throttle on a straight stays fast.
MAX_ABS_ENERGY_MJ = 4.0
MIN_SPEED_KMH = 180.0


def _car_draw(key, car: CarParams, varied: bool):
    if not varied:
        return (jnp.asarray(car.ice_power_w), jnp.asarray(car.cda_straight),
                jnp.asarray(car.mass_kg + car.fuel_kg / 2))
    k1, k2, k3 = jax.random.split(key, 3)
    ice = car.ice_power_w * jax.random.uniform(k1, minval=0.9, maxval=1.1)
    cda = car.cda_straight * jax.random.uniform(k2, minval=0.8, maxval=1.2)
    mass = car.mass_kg + jax.random.uniform(k3, minval=0.0, maxval=car.fuel_kg)
    return ice, cda, mass


def _strategy_draw(key, flexible: bool):
    """Parameters of a command u(s) in [-1, 1]: deploy u * cap(v) if positive, else harvest."""
    if flexible:
        ka, kb, kc, kw = jax.random.split(key, 4)
        return {"amp": 1.5 * jax.random.normal(ka, (N_BUMPS,)),
                "offset": 0.7 * jax.random.normal(kb),
                "centre": jax.random.uniform(kc, (N_BUMPS,), maxval=N_POINTS * DS),
                "width": jax.random.uniform(kw, (N_BUMPS,), minval=60.0, maxval=300.0)}
    kd, ks, kh = jax.random.split(key, 3)
    return {"deploy": jax.random.uniform(kd, minval=0.2, maxval=1.0),
            "clip_m": jax.random.uniform(ks, minval=0.3, maxval=0.9) * N_POINTS * DS,
            "harvest_frac": jax.random.uniform(kh, minval=0.0, maxval=300e3 / 350e3)}


def _command(strategy: dict, s, flexible: bool):
    if flexible:
        bumps = strategy["amp"] * jnp.exp(-0.5 * ((s - strategy["centre"]) / strategy["width"]) ** 2)
        return jnp.tanh(strategy["offset"] + bumps.sum())
    return jnp.where(s < strategy["clip_m"], strategy["deploy"], -strategy["harvest_frac"])


def _drive(car: CarParams, ice, cda, mass, v0, strategy, flexible: bool):
    s_grid = jnp.arange(N_POINTS) * DS

    def step(v, s):
        u = _command(strategy, s, flexible)
        cap = mguk_cap_kmh(v * 3.6, car, jnp)
        p_e = jnp.where(u > 0, u * cap, u * car.mguk_power_w)
        f = jnp.minimum((ice + p_e) / v, car.traction_max_n)
        a = (f - 0.5 * car.rho * cda * v * v - car.crr * mass * G) / mass
        v_next = jnp.sqrt(jnp.maximum(v * v + 2.0 * a * DS, 25.0))
        battery = jnp.where(p_e > 0, p_e / car.eta_deploy, p_e * car.eta_harvest) * DS / v
        return v_next, (v, battery)

    _, (v, battery) = jax.lax.scan(step, v0, s_grid)
    return v, battery.sum() / 1e6


@partial(jax.jit, static_argnames=("n", "varied", "flexible", "car"))
def _generate(key, n: int, varied: bool, flexible: bool, car: CarParams):
    def one(k):
        kc, ks, kv, kn = jax.random.split(k, 4)
        ice, cda, mass = _car_draw(kc, car, varied)
        strategy = _strategy_draw(ks, flexible)
        v0 = jax.random.uniform(kv, minval=55.0, maxval=70.0)
        v, energy_mj = _drive(car, ice, cda, mass, v0, strategy, flexible)
        observed = jnp.round(v * 3.6 + SPEED_NOISE_KMH * jax.random.normal(kn, v.shape))
        return observed, energy_mj

    return jax.vmap(one)(jax.random.split(key, n))


def generate(seed: int, n: int, cars: str, strategies: str, car: CarParams):
    """Returns (speed_kmh[n, 240], battery_energy_mj[n]) for cars in {identical, varied}
    and strategies in {simple, flexible}, keeping only physically plausible samples."""
    if cars not in ("identical", "varied") or strategies not in ("simple", "flexible"):
        raise ValueError(f"unknown regime {cars}/{strategies}")
    xs, ys, batch, round_no = [], [], max(n, 1000), 0
    while sum(len(y) for y in ys) < n:
        key = jax.random.fold_in(jax.random.PRNGKey(seed), round_no)
        x, y = (np.asarray(a) for a in _generate(key, batch, cars == "varied",
                                                    strategies == "flexible", car))
        keep = (np.abs(y) <= MAX_ABS_ENERGY_MJ) & (x.min(axis=1) >= MIN_SPEED_KMH)
        xs.append(x[keep])
        ys.append(y[keep])
        round_no += 1
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]
