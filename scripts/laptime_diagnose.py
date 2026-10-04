"""Compare the Phase 2 detector with the lap-time code on the same laps.

Prints, for one circuit, what each sees on individual laps, and for each circuit how well
time loss tracks the detector's clipping duration (a fairer consistency check than power
drop alone, since time lost depends on both the size and the duration of clipping).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from derate_analysis import lap_quality, longest_flat_out_run
from erslens.laptime import MIN_GAIN_KMH, _flat_run, clip_time_loss
from erslens.params import load_car_params


def laps_of(year, event, car):
    from erslens.ingest import load_driver_race, race_drivers, reference_track

    track = reference_track(year, event)
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
    elevation = None if track.elevation_m is None else (track.distance_m, track.elevation_m)
    rows = []
    for drv in race_drivers(year, event):
        race = load_driver_race(year, event, drv)
        if race.empty:
            continue
        durations = race.groupby("lap")["time_s"].max()
        for lap_no, lap in race.groupby("lap"):
            if lap_no == 1 or durations[lap_no] > 1.07 * durations.median():
                continue
            if lap["pit_in"].iloc[0] or lap["pit_out"].iloc[0]:
                continue
            lap = lap.reset_index(drop=True)
            metrics, status = lap_quality(lap, start_m, end_m)
            if status != "ok":
                continue
            run = _flat_run(lap, start_m, end_m)
            mass = car.mass_kg + max(0.0, car.fuel_kg - car.fuel_per_lap_kg * lap_no)
            est = clip_time_loss(lap, start_m, end_m, mass, car, elevation)
            v = run["v"].to_numpy() * 3.6 if run is not None else np.array([np.nan])
            rows.append({
                "driver": drv, "lap": lap_no,
                "det_drop_kmh": metrics["flat_out_drop_kmh"],
                "det_peak_to_brake_m": metrics["peak_before_brake_m"],
                "det_top_kmh": metrics["top_speed_kmh"],
                "lt_run_m": f"{run['distance_m'].iloc[0]:.0f}-{run['distance_m'].iloc[-1]:.0f}" if run is not None else "none",
                "lt_peak_kmh": float(np.nanmax(v)), "lt_end_kmh": float(v[-1]),
                "lt_drop_kmh": float(np.nanmax(v) - v[-1]),
                "lt_result": "none" if est is None else ("clip" if est["clipping"] else "no clip"),
                "t_loss_s": np.nan if est is None else est["t_loss_s"],
                "det_clip_s": metrics["peak_before_brake_m"] / (0.5 * (metrics["top_speed_kmh"] + metrics["end_speed_kmh"]) / 3.6),
            })
    return (start_m, end_m), pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--inspect", default="Austria")
    p.add_argument("--events", nargs="+", default=["Australia", "China", "Miami", "Canada", "Austria"])
    p.add_argument("--n", type=int, default=12)
    args = p.parse_args()
    car = load_car_params("configs/car_2026.yaml")

    (start_m, end_m), df = laps_of(args.year, args.inspect, car)
    print(f"{args.inspect}: straight {start_m:.0f}-{end_m:.0f} m; {len(df)} clean laps")
    print(f"Detector sees >= {MIN_GAIN_KMH:.0f} km/h drop on {np.mean(df.det_drop_kmh >= MIN_GAIN_KMH):.0%}; "
          f"lap-time run sees it on {np.mean(df.lt_drop_kmh >= MIN_GAIN_KMH):.0%}")
    print(df.head(args.n).round(1).to_string(index=False))

    print("\nConsistency: time loss vs detector clipping duration (Spearman, clipping laps)")
    for ev in args.events:
        _, d = laps_of(args.year, ev, car)
        c = d[d.lt_result == "clip"].dropna(subset=["t_loss_s"])
        rho = spearmanr(c.t_loss_s, c.det_clip_s)[0] if len(c) > 10 else np.nan
        print(f"  {ev}: rho = {rho:.2f} on {len(c)} laps")


if __name__ == "__main__":
    main()
