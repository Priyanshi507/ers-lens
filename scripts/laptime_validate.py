"""Validate the clipping time-loss estimator on simulated straights where the truth is known."""
import argparse

import jax.numpy as jnp
import numpy as np
import pandas as pd

from erslens.laptime import clip_time_loss
from erslens.params import load_car_params
from erslens.straightfit import DS, _simulate

N_POINTS = 260


def simulate(car, ice, cda, mass, v0, d, clip_m, h):
    s = jnp.arange(N_POINTS) * DS
    v, _ = _simulate(d, clip_m, h, v0, s, jnp.zeros(N_POINTS), mass, car, ice, cda)
    return np.asarray(s), np.asarray(v)


def as_telemetry(s, v, rng, hz=4.0):
    t = np.concatenate([[0.0], np.cumsum(2 * DS / (v[1:] + v[:-1]))])
    tt = np.arange(0, t[-1], 1 / hz)
    kmh = np.round(np.interp(tt, t, v) * 3.6 + rng.normal(0, 0.8, len(tt)))
    return pd.DataFrame({"time_s": tt, "distance_m": np.interp(tt, t, s), "speed_kmh": kmh,
                         "throttle": 100.0, "brake": False})


def time_between(s, v, s0, s1):
    m = (s >= s0) & (s <= s1)
    vv = v[m]
    return float(np.sum(2 * DS / (vv[1:] + vv[:-1])))


def run(n: int, seed: int):
    car = load_car_params("configs/car_2026.yaml")
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        ice = car.ice_power_w * rng.uniform(0.9, 1.1)
        cda = car.cda_straight * rng.uniform(0.8, 1.2)
        mass = car.mass_kg + rng.uniform(0, car.fuel_kg)
        d, clip_frac, h = rng.uniform(0.4, 1.0), rng.uniform(0.35, 0.7), rng.uniform(5e4, 3e5)
        v0 = rng.uniform(58.0, 66.0)
        s, v = simulate(car, ice, cda, mass, v0, d, clip_frac * N_POINTS * DS, h)
        _, v_free = simulate(car, ice, cda, mass, v0, d, 1e9, 0.0)
        if int(np.argmax(v)) > N_POINTS - 10:
            continue
        # The analyst knows only the nominal mass, not this car's fuel load.
        est = clip_time_loss(as_telemetry(s, v, rng), 0, 1e6, car.mass_kg + car.fuel_kg / 2, car)
        if est is None or not est["clipping"]:
            continue
        # Compare over exactly the stretch the estimator analyses.
        a, b = est["span_from_m"], est["span_to_m"]
        true_loss = time_between(s, v, a, b) - time_between(s, v_free, a, b)
        rows.append({"true_s": true_loss, "est_s": est["t_loss_s"], "low_s": est["t_low_s"],
                     "high_s": est["t_high_s"]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    df = run(args.n, args.seed)
    inside = (df.true_s >= df.low_s) & (df.true_s <= df.high_s)
    width = (df.high_s - df.low_s) / df.est_s
    err = df.est_s - df.true_s
    print(f"{len(df)} simulated laps; true time loss {df.true_s.median():.3f} s median "
          f"({df.true_s.min():.3f}-{df.true_s.max():.3f})")
    print(f"Nominal estimate: median abs error {err.abs().median():.3f} s, bias {err.median():+.3f} s, "
          f"correlation {np.corrcoef(df.est_s, df.true_s)[0, 1]:.2f}")
    print(f"Truth inside parameter bounds on {inside.mean():.0%} of laps; median bound width {width.median():.0%} of estimate")
    rel = (err / df.true_s).abs()
    print(f"Error quantiles: abs 90th percentile {err.abs().quantile(0.9):.3f} s; "
          f"relative median {rel.median():.1%}, 90th percentile {rel.quantile(0.9):.1%}")


if __name__ == "__main__":
    main()
