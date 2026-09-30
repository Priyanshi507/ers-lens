"""Two-panel figure and summary statistics for the flat-out derating result."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from erslens.ingest import load_track
from derate_analysis import STRICT_FROZEN_RUN, longest_flat_out_run

MIN_LAPS = 10


def team_lookup(year: int, event: str) -> dict:
    try:
        import fastf1
        fastf1.Cache.enable_cache("data/cache")
        s = fastf1.get_session(year, event, "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        return s.results.set_index("Abbreviation")["TeamName"].to_dict()
    except Exception:
        return {}


def detrended_corr(d: pd.DataFrame) -> float:
    # Lap times fall as fuel burns off; remove that trend before comparing with speed loss.
    x, y, lap = d["flat_out_drop_kmh"], d["lap_time_s"], d["lap"]
    if x.std() == 0 or len(d) < 10:
        return np.nan
    rx = x - np.polyval(np.polyfit(lap, x, 1), lap)
    ry = y - np.polyval(np.polyfit(lap, y, 1), lap)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--race-glob", required=True, help='e.g. "data/real/2026_china_{drv}_race.parquet"')
    p.add_argument("--per-lap", default="results/derate_per_lap.csv")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--event", default="China")
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--strict", action="store_true",
                   help="drop laps whose speed is frozen for STRICT_FROZEN_RUN+ samples")
    p.add_argument("--out", default="results/abstract_figure.png")
    args = p.parse_args()

    laps = pd.read_csv(args.per_lap)
    if args.strict:
        laps = laps[laps["max_frozen_run"] < STRICT_FROZEN_RUN]
        print(f"Strict mode: laps with speed frozen for >= {STRICT_FROZEN_RUN} samples removed")
    laps["derate"] = laps["flat_out_drop_kmh"] >= args.threshold
    per = laps.groupby("driver").agg(
        n=("lap", "count"),
        share=("derate", "mean"),
        drop=("flat_out_drop_kmh", "median"),
        peak_m=("peak_before_brake_m", "median"),
    )
    per["within_corr"] = laps.groupby("driver").apply(detrended_corr, include_groups=False)
    teams = team_lookup(args.year, args.event)
    per["team"] = [teams.get(d, "") for d in per.index]

    print(f"Drivers: {len(per)}  clean laps: {len(laps)}  "
          f"flat-out drop >= {args.threshold:g} km/h: {laps['derate'].sum()} "
          f"({laps['derate'].mean():.1%})")
    pooled = laps[["flat_out_drop_kmh", "lap_time_s"]].corr().iloc[0, 1]
    wc = per.loc[per["n"] >= MIN_LAPS, "within_corr"].dropna()
    print(f"Pooled corr(drop, lap time): {pooled:.3f}")
    print(f"Within-driver detrended corr: mean {wc.mean():.3f}, median {wc.median():.3f}, "
          f"negative for {(wc < 0).sum()}/{len(wc)} drivers")
    print()
    print(per.sort_values(["team", "share"]).round(2).to_string())

    eligible = per[per["n"] >= MIN_LAPS]
    hi, lo = eligible["drop"].idxmax(), eligible["drop"].idxmin()
    start_m, end_m = longest_flat_out_run(load_track(args.track))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for drv, color in [(hi, "tab:red"), (lo, "tab:blue")]:
        d = laps[laps["driver"] == drv]
        # Among laps closest to the driver's median, prefer the one with the least frozen data.
        d = d[d["lift_before_brake_m"] <= 60]
        d = d.assign(dist_to_median=(d["flat_out_drop_kmh"] - per.loc[drv, "drop"]).abs())
        lap_no = int(d.sort_values(["dist_to_median", "max_frozen_run"]).iloc[0]["lap"])
        race = pd.read_parquet(args.race_glob.format(drv=drv))
        lap = race[(race["lap"] == lap_no) & race["distance_m"].between(start_m, end_m + 100)]
        ax[0].plot(lap["distance_m"] - start_m, lap["speed_kmh"], "-", lw=2, color=color,
                   label=f"{drv} lap {lap_no}")
        flat = lap[(lap["throttle"] >= 98) & ~lap["brake"].astype(bool)]
        ax[0].scatter(flat["distance_m"] - start_m, flat["speed_kmh"], s=8, color="k", zorder=3)
    ax[0].set_xlabel("distance along longest straight (m)")
    ax[0].set_ylabel("speed (km/h)")
    ax[0].set_title("A. Typical lap: black dots = full throttle, no brake")
    ax[0].legend()

    ax[1].scatter(eligible["peak_m"], eligible["share"] * 100, s=30)
    for drv, row in eligible.iterrows():
        ax[1].annotate(f"{drv} ({int(row['n'])})", (row["peak_m"], row["share"] * 100), fontsize=8,
                       xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("median distance from speed peak to braking point (m)")
    ax[1].set_ylabel(f"% of clean laps losing >= {args.threshold:g} km/h at full throttle")
    ax[1].set_title("B. Per-driver clipping (laps analysed in brackets)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    per.to_csv(Path(args.out).with_suffix(".csv"))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
