"""Where the simulated pole lap beats the real one, run by run.

Off full throttle the track caps speed at the real trace, so any time the model gains over the
real lap is gained under full throttle. This splits each full-throttle run below and above
200 km/h: a gain at low speed points at traction or total power, a gain at high speed at drag.
Writes results/strategy/correlation.md.

    python scripts/strategy_correlation.py --events Australia China Miami Canada
"""
import argparse
from pathlib import Path

import numpy as np

from erslens.ingest import _car_data, _session
from erslens.optimal import race_car, solve_refined
from erslens.params import load_car_params
from erslens.physics import braking_envelope
from erslens.strategy import race_rules
from erslens.track import track_from_reference

QUALI_FUEL_KG = 2.0
FULL_THROTTLE = 98.0
SPLIT_KMH = 200.0
MIN_RUN_M = 100.0


def runs(mask: np.ndarray, min_len: int):
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_len:
                yield i, j
            i = j
        else:
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=["Australia", "China", "Miami", "Canada"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--ds", type=float, default=5.0)
    args = ap.parse_args()
    base = load_car_params("configs/car_2026.yaml")
    out = ["Optimum at qualifying fuel vs the real pole lap it was built from. "
           "Gain = real time minus simulated time; positive = model faster.", ""]
    summary = ["| Circuit | Real lap (from trace) | Model lap | Gain at full throttle < 200 km/h | "
               "Gain at full throttle >= 200 km/h | Gain elsewhere | Real top speed | Model top speed |",
               "|---|---|---|---|---|---|---|---|"]
    detail = ["| Circuit | Run start | Length | Real time | Model time | Gain | Real exit speed | Model exit speed |",
              "|---|---|---|---|---|---|---|---|"]

    for event in args.events:
        lap = _session(args.year, event, "Q", "data/cache").laps.pick_quicklaps().pick_fastest()
        df = _car_data(lap)
        track = track_from_reference(event, df["distance_m"].to_numpy(), df["speed_kmh"].to_numpy(),
                                     df["throttle"].to_numpy(), ds=args.ds)
        grid = track.distance_m
        v_real = np.interp(grid, df["distance_m"], df["speed_kmh"]) / 3.6
        flat = np.interp(grid, df["distance_m"], df["throttle"]) >= FULL_THROTTLE

        car = race_rules(base, event).with_(fuel_kg=QUALI_FUEL_KG)
        sol = solve_refined(track, car)
        # The solver starts the lap at the slowest point; undo that roll to line up with the real trace.
        k = int(np.argmin(braking_envelope(track, race_car(car))))
        v_sim = np.roll(np.asarray(sol.out["v"]), k)
        dt_sim = np.roll(np.asarray(sol.out["dt"]), k)
        dt_real = args.ds / np.maximum(v_real, 1.0)
        gain = dt_real - dt_sim

        low = flat & (v_real < SPLIT_KMH / 3.6)
        high = flat & (v_real >= SPLIT_KMH / 3.6)
        summary.append(
            f"| {event} | {dt_real.sum():.3f} s | {dt_sim.sum():.3f} s | {gain[low].sum():+.3f} s | "
            f"{gain[high].sum():+.3f} s | {gain[~flat].sum():+.3f} s | {v_real.max() * 3.6:.0f} km/h | "
            f"{v_sim.max() * 3.6:.0f} km/h |")
        for i, j in runs(flat, int(MIN_RUN_M / args.ds)):
            detail.append(
                f"| {event} | {grid[i]:.0f} m | {grid[j - 1] - grid[i] + args.ds:.0f} m | {dt_real[i:j].sum():.3f} s | "
                f"{dt_sim[i:j].sum():.3f} s | {gain[i:j].sum():+.3f} s | {v_real[j - 1] * 3.6:.0f} km/h | "
                f"{v_sim[j - 1] * 3.6:.0f} km/h |")
        print(summary[-1], flush=True)

    text = "\n".join(out + summary + ["", "Full-throttle runs of at least 100 m:", ""] + detail)
    Path("results/strategy/correlation.md").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
