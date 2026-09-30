"""Measure end-of-straight clipping across 2026 races and compare before/after the Miami rule change.

The FIA cut maximum recharge from 8 MJ to 7 MJ from the Miami GP onward, targeting
about 2-4 s of super-clipping per lap (down from a reported 6-8 s in the first rounds).
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from derate_analysis import analyse_driver, longest_flat_out_run

CLIP_KMH = 3.0
MIN_LAPS = 20  # races with fewer clean laps are reported but excluded from the comparison


def clip_seconds(laps: pd.DataFrame) -> pd.Series:
    """Time spent losing speed at full throttle; zero on laps that did not clip."""
    mean_ms = (laps["top_speed_kmh"] + laps["end_speed_kmh"]) / 2 / 3.6
    secs = laps["peak_before_brake_m"] / mean_ms
    return secs.where(laps["flat_out_drop_kmh"] >= CLIP_KMH, 0.0)


def clip_fraction(laps: pd.DataFrame, straight_m: float) -> pd.Series:
    """Share of the straight spent losing speed at full throttle; comparable across circuits."""
    frac = laps["peak_before_brake_m"] / straight_m
    return frac.where(laps["flat_out_drop_kmh"] >= CLIP_KMH, 0.0).clip(upper=1.0)


def permutation_p(before: list[float], after: list[float]) -> float:
    """Exact one-sided test that races before the change clip more, with races as the unit."""
    values = np.array(before + after)
    observed = np.mean(before) - np.mean(after)
    idx = range(len(values))
    diffs = [values[list(c)].mean() - np.delete(values, list(c)).mean()
             for c in combinations(idx, len(before))]
    return float(np.mean(np.array(diffs) >= observed - 1e-12))


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def race_summary(laps: pd.DataFrame, candidates: int, straight_m: float) -> dict:
    secs = clip_seconds(laps).to_numpy()
    frac = clip_fraction(laps, straight_m).to_numpy()
    lo, hi = bootstrap_ci(secs) if len(secs) > 1 else (np.nan, np.nan)
    return {
        "candidate_laps": candidates,
        "clean_laps": len(laps),
        "drivers": laps["driver"].nunique(),
        "clip_share": float((laps["flat_out_drop_kmh"] >= CLIP_KMH).mean()),
        "clip_s_mean": float(secs.mean()),
        "clip_s_lo": lo,
        "clip_s_hi": hi,
        "clip_frac_mean": float(frac.mean()),
        "drop_kmh_median": float(laps["flat_out_drop_kmh"].median()),
    }


def analyse_event(year: int, event: str, out: Path, session: str = "R") -> tuple[dict, pd.DataFrame]:
    from erslens.ingest import event_info, load_driver_race, race_drivers, reference_track

    name, date = event_info(year, event)
    track = reference_track(year, event)
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
    frames, candidates = [], 0
    pace_ref = "fastest" if session == "Q" else "median"
    for drv in race_drivers(year, event, kind=session):
        race = load_driver_race(year, event, drv, kind=session)
        if race.empty:
            continue
        res = analyse_driver(race, start_m, end_m, pace_ref=pace_ref)
        candidates += len(res) + sum(res.attrs.get("rejected", {}).values())
        if len(res):
            res.insert(0, "driver", drv)
            frames.append(res)
    laps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    laps.insert(0, "event", name)
    laps.to_csv(out / f"{year}_{name.lower().replace(' ', '_')}_{session}_per_lap.csv", index=False)
    straight_m = end_m - start_m
    row = {"event": name, "date": date.date(), "straight_m": round(straight_m),
           **(race_summary(laps, candidates, straight_m) if len(laps)
              else {"candidate_laps": candidates, "clean_laps": 0})}
    return row, laps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--events", nargs="+", required=True, help='e.g. China Japan Miami Spain')
    p.add_argument("--change-event", default="Miami", help="first race under the new rules")
    p.add_argument("--session", choices=["R", "Q"], default="R", help="race or qualifying")
    p.add_argument("--out", default="results/multi")
    args = p.parse_args()

    from erslens.ingest import event_info

    out = Path(args.out) / args.session
    out.mkdir(parents=True, exist_ok=True)
    _, change_date = event_info(args.year, args.change_event)

    rows, all_laps = [], []
    for ev in args.events:
        print(f"\n=== {ev} ===")
        row, laps = analyse_event(args.year, ev, out, args.session)
        row["period"] = "after" if pd.Timestamp(row["date"]) >= change_date else "before"
        rows.append(row)
        if len(laps):
            laps["period"] = row["period"]
            all_laps.append(laps)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})

    summary = pd.DataFrame(rows).sort_values("date")
    summary["used"] = summary["clean_laps"] >= MIN_LAPS
    summary.to_csv(out / "race_summary.csv", index=False)
    print("\n" + summary.round(2).to_string(index=False))

    s = summary[summary["used"]]
    before = s.loc[s["period"] == "before", "clip_frac_mean"].tolist()
    after = s.loc[s["period"] == "after", "clip_frac_mean"].tolist()
    if before and after:
        print(f"\nShare of longest straight spent clipping (race-level means): "
              f"before {np.mean(before):.1%} ({len(before)} events) vs after {np.mean(after):.1%} "
              f"({len(after)} events); exact permutation one-sided p = {permutation_p(before, after):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = s["period"].map({"before": "tab:red", "after": "tab:blue"})
    x = np.arange(len(s))
    ax[0].errorbar(x, s["clip_s_mean"], yerr=[s["clip_s_mean"] - s["clip_s_lo"],
                   s["clip_s_hi"] - s["clip_s_mean"]], fmt="none", ecolor="grey", capsize=3)
    ax[0].scatter(x, s["clip_s_mean"], c=colors, s=40, zorder=3)
    ax[1].scatter(x, s["clip_frac_mean"] * 100, c=colors, s=40)
    for a in ax:
        a.set_xticks(x, [f"{e}\n({n})" for e, n in zip(s["event"].str.replace(" Grand Prix", ""),
                                                        s["clean_laps"])], fontsize=8)
        if (s["period"] == "after").any() and (s["period"] == "before").any():
            a.axvline((s["period"] == "before").sum() - 0.5, color="k", ls="--", lw=0.8)
    ax[0].set_ylabel("seconds per lap slowing at full throttle\n(longest straight, mean, 95% CI)")
    session = "qualifying" if args.session == "Q" else "race"
    ax[0].set_title(f"A. Clipping time, {session}: before (red) vs after (blue)")
    ax[1].set_ylabel("% of the straight spent losing speed\nat full throttle (mean over laps)")
    ax[1].set_title("B. Clipping normalised by straight length")
    fig.tight_layout()
    fig.savefig(out / "multi_race.png", dpi=150)
    print(f"\nSaved {out}/race_summary.csv, multi_race.png, per-lap CSVs")


if __name__ == "__main__":
    main()
