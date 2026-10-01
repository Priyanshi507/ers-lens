"""Validate the electric power swing estimator on simulated cars where the truth is known."""
import argparse

import numpy as np
import pandas as pd

from derate_analysis import longest_flat_out_run
from erslens.energy import electric_power_swing, true_swing
from erslens.params import load_car_params
from erslens.policies import ClipEndOfStraight, StraightsOnly
from erslens.simulate import simulate
from erslens.synth import randomize_car, sensor_model
from erslens.track import demo_circuit


def run(n_cars: int, laps: int, seed: int, mass_error: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = load_car_params("configs/car_2026.yaml")
    track = demo_circuit()
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
    rows = []
    for car_id in range(n_cars):
        car = randomize_car(base, rng)
        clips = car_id % 4 != 0
        policy = (ClipEndOfStraight(clip_from=rng.uniform(0.4, 0.85), clip_w=rng.uniform(50e3, 350e3))
                  if clips else StraightsOnly())
        sim = simulate(track, car, policy, n_laps=laps, soc0_frac=0.9)
        obs = sensor_model(sim, car, rng)
        obs["distance_m"] -= obs["lap"] * track.length_m
        # The analyst knows only the nominal car, not this car's true fuel load.
        mass = (base.mass_kg + base.fuel_kg / 2) * (1 + mass_error)
        for lap_no, lap in obs.groupby("lap"):
            if lap_no == 0:
                continue
            r = electric_power_swing(lap.reset_index(drop=True), start_m, end_m, mass)
            truth = (true_swing(sim, lap_no, start_m, end_m, (r["band_lo_kmh"], r["band_hi_kmh"]))
                     if r else np.nan)
            rows.append({"car": car_id, "clips": clips, "measurable": r is not None,
                         "est_kw": r["swing_w"] / 1e3 if r else np.nan,
                         "true_kw": truth / 1e3})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    ok = df.dropna(subset=["est_kw", "true_kw"])
    err = ok["est_kw"] - ok["true_kw"]
    return {"measurable_clip_laps": f"{df.loc[df.clips, 'measurable'].mean():.0%}",
            "false_positive_laps": int(df.loc[~df.clips, "measurable"].sum()),
            "true_mean_kw": round(ok["true_kw"].mean()), "est_mean_kw": round(ok["est_kw"].mean()),
            "mae_kw": round(err.abs().mean(), 1), "bias_kw": round(err.mean(), 1),
            "corr": round(float(np.corrcoef(ok["est_kw"], ok["true_kw"])[0, 1]), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cars", type=int, default=40)
    p.add_argument("--laps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    for err in (0.0, -0.05, 0.05):
        label = "nominal mass" if err == 0 else f"mass assumed {err:+.0%}"
        print(f"{label:22s}", summarise(run(args.cars, args.laps, args.seed, err)))


if __name__ == "__main__":
    main()
