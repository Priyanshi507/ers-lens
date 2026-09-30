"""Detect energy derating on the longest straight: speed falling while throttle stays flat out."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from erslens.ingest import load_track
from erslens.track import Track

FULL_THROTTLE = 98.0


def longest_flat_out_run(track: Track) -> tuple[float, float]:
    flat = ~np.isfinite(track.v_limit_ms)
    best, best_len, i = (0, 0), 0, 0
    while i < track.n:
        if flat[i]:
            j = i
            while j < track.n and flat[j]:
                j += 1
            if j - i > best_len:
                best, best_len = (i, j), j - i
            i = j
        else:
            i += 1
    return float(track.distance_m[best[0]]), float(track.distance_m[best[1] - 1])


def lap_metrics(lap: pd.DataFrame, start_m: float, end_m: float) -> dict | None:
    win = lap[(lap["distance_m"] >= start_m) & (lap["distance_m"] <= end_m + 100)]
    flat = ((win["throttle"] >= FULL_THROTTLE) & ~win["brake"].astype(bool)).to_numpy()
    if flat.sum() < 5:
        return None
    speed = win["speed_kmh"].to_numpy()
    dist = win["distance_m"].to_numpy()
    peak = int(np.argmax(np.where(flat, speed, -np.inf)))
    # Follow only the unbroken flat-out stretch after the peak; later flat-out
    # samples belong to the exit of the next corner.
    last = peak
    while last + 1 < len(flat) and flat[last + 1]:
        last += 1
    return {
        "top_speed_kmh": float(speed[peak]),
        "end_speed_kmh": float(speed[last]),
        "flat_out_drop_kmh": float(speed[peak] - speed[last]),
        "peak_before_brake_m": float(dist[last] - dist[peak]),
    }


def analyse_driver(race: pd.DataFrame, start_m: float, end_m: float) -> pd.DataFrame:
    durations = race.groupby("lap")["time_s"].max()
    clean = durations[durations <= 1.07 * durations.median()].index
    rows = []
    for lap_no, lap in race.groupby("lap"):
        if lap_no == 1 or lap_no not in clean:
            continue
        if lap["pit_in"].iloc[0] or lap["pit_out"].iloc[0]:
            continue
        m = lap_metrics(lap.reset_index(drop=True), start_m, end_m)
        if m:
            rows.append({"lap": lap_no, "lap_time_s": float(durations[lap_no]), **m})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--races", nargs="+", required=True, help="per-driver race parquet files")
    p.add_argument("--drop-threshold", type=float, default=3.0,
                   help="km/h drop at full throttle counted as a derate event")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    track = load_track(args.track)
    start_m, end_m = longest_flat_out_run(track)
    print(f"Longest flat-out run: {start_m:.0f}-{end_m:.0f} m ({end_m - start_m:.0f} m)\n")

    per_driver = {}
    for path in args.races:
        drv = Path(path).stem.split("_")[-2]
        res = analyse_driver(pd.read_parquet(path), start_m, end_m)
        res.insert(0, "driver", drv)
        per_driver[drv] = res

    allres = pd.concat(per_driver.values(), ignore_index=True)
    allres.to_csv(out / "derate_per_lap.csv", index=False)

    allres["derate"] = allres["flat_out_drop_kmh"] >= args.drop_threshold
    summary = allres.groupby("driver").agg(
        clean_laps=("lap", "count"),
        top_speed_median=("top_speed_kmh", "median"),
        top_speed_spread=("top_speed_kmh", lambda s: s.quantile(0.9) - s.quantile(0.1)),
        derate_laps=("derate", "sum"),
        derate_share=("derate", "mean"),
        mean_drop_when_derated=("flat_out_drop_kmh", lambda s: s[s >= args.drop_threshold].mean()),
    ).round(2)
    summary.to_csv(out / "derate_summary.csv")
    print(summary.to_string())

    corr = allres[["flat_out_drop_kmh", "lap_time_s"]].corr().iloc[0, 1]
    print(f"\nCorrelation(flat-out drop, lap time) across all clean laps: {corr:.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for drv, res in per_driver.items():
        ax[0].plot(res["lap"], res["top_speed_kmh"], ".-", lw=0.8, label=drv)
        ax[1].plot(res["lap"], res["flat_out_drop_kmh"], ".-", lw=0.8, label=drv)
    ax[0].set_ylabel("top speed km/h")
    ax[1].set_ylabel("speed lost at\nfull throttle km/h")
    ax[1].axhline(args.drop_threshold, color="grey", ls="--", lw=0.7)
    ax[1].set_xlabel("lap")
    ax[0].legend(ncol=len(per_driver))
    fig.suptitle("Longest straight: top speed and flat-out speed loss per lap")
    fig.tight_layout()
    fig.savefig(out / "derate_by_lap.png", dpi=130)
    print(f"\nSaved {out}/derate_per_lap.csv, derate_summary.csv, derate_by_lap.png")


if __name__ == "__main__":
    main()
