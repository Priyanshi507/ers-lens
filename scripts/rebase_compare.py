"""Verdicts R1-R3 (research log, 2026-11-01): clipping on whole straights against the season test.

    python scripts/rebase_compare.py
"""
import argparse

import pandas as pd
from scipy.stats import spearmanr

NEW = ("Monaco", "Barcelona", "British", "Hungarian", "Dutch", "Italian", "Spanish", "Azerbaijan",
       "Bahrain", "Singapore", "United States")
MIN_LAPS = 20
LONGER_M = 200


def compare(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    m = old.merge(new, on="event", suffixes=("_old", "_new"))
    m["longer_m"] = m.straight_m_new - m.straight_m_old
    enough = (m.clean_laps_old >= MIN_LAPS) & (m.clean_laps_new >= MIN_LAPS)
    longer = m[enough & (m.longer_m >= LONGER_M)]
    r1 = (longer.clip_share_new > longer.clip_share_old)
    fresh = new[new.event.str.contains("|".join(NEW)) & (new.clean_laps >= MIN_LAPS)]
    r2 = fresh.clip_share >= 0.80
    used = new[new.clean_laps >= MIN_LAPS]
    rho = float(spearmanr(used.straight_m, used.clip_share).statistic) if len(used) > 2 else float("nan")
    return {"table": m, "R1": (int(r1.sum()), len(longer), bool(len(longer)) and bool(r1.all())),
            "R2": (int(r2.sum()), len(fresh), len(fresh) > 0 and r2.sum() * 2 >= len(fresh)),
            "R3": (rho, len(used), rho >= 0.7)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="results/multi_season/R/race_summary.csv")
    ap.add_argument("--new", default="results/multi_rebased/R/race_summary.csv")
    args = ap.parse_args()
    v = compare(pd.read_csv(args.old), pd.read_csv(args.new))
    cols = ["event", "straight_m_old", "straight_m_new", "clean_laps_old", "clean_laps_new",
            "clip_share_old", "clip_share_new", "clip_frac_mean_old", "clip_frac_mean_new"]
    print(v["table"][cols].round(3).to_string(index=False))
    n, total, held = v["R1"]
    print(f"R1: higher clip share on every straight at least {LONGER_M} m longer: {n} of {total} -> {'held' if held else 'NOT held'}")
    n, total, held = v["R2"]
    print(f"R2: at least 80% of clean laps clip at half or more of the new races: {n} of {total} -> {'held' if held else 'NOT held'}")
    rho, total, held = v["R3"]
    print(f"R3: Spearman(straight length, clip share) >= 0.7 over {total} races: {rho:.2f} -> {'held' if held else 'NOT held'}")


if __name__ == "__main__":
    main()
