"""Lift-and-coast in real race laps against the model, predictions RL1-RL3 (research log, 2026-11-14).

    python scripts/lift_real.py [--per-lap results/multi_rebased_mid/R] [--runs results/fia_fix_calendar_lc]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hindcast_2026 import FIRST_350, NAMES

LIFT_M = 60.0          # below this the gap can be telemetry sampling alone (~22 m per sample at 300 km/h)
MIN_LAPS = 10


def lifts(laps: pd.DataFrame) -> pd.Series:
    """Throttle closed (< 20%, no brake) for at least LIFT_M before the first brake sample."""
    return laps["coast_before_brake_m"] >= LIFT_M


def per_race(per_lap: Path, runs: Path) -> pd.DataFrame:
    summary = pd.read_csv(per_lap / "race_summary.csv")
    summary = summary[summary.used.astype(bool)]
    model = pd.read_csv(runs / "runs.csv")
    rows = []
    for r in summary.itertuples():
        laps = pd.read_csv(per_lap / f"2026_{r.event.lower().replace(' ', '_')}_R_per_lap.csv")
        if "coast_before_brake_m" not in laps:
            raise SystemExit(f"{r.event}: per-lap file has no coast_before_brake_m; re-run multi_race.py --rebase")
        lift = lifts(laps)
        cap = 350 if pd.Timestamp(r.date) >= FIRST_350 else 250
        name = NAMES.get(r.event, r.event)
        m = model[(model.event == name) & (model.car == "original") & (model.cap_kw == cap)]
        a, b = laps.loc[lift, "flat_out_drop_kmh"], laps.loc[~lift, "flat_out_drop_kmh"]
        rows.append({"event": r.event, "cap_kw": cap, "laps": len(laps), "lift_share": float(lift.mean()),
                     "lap_coast_m": float(laps["lap_coast_m"].median()) if "lap_coast_m" in laps else np.nan,
                     "unmeasured": int(laps.coast_before_brake_m.isna().sum()),
                     "coast_m_median": float(laps.coast_before_brake_m.median()),
                     "model_coast_mj": float(m.coast_mj.iloc[0]),
                     "drop_lift_kmh": a.median() if len(a) >= MIN_LAPS else np.nan,
                     "drop_no_lift_kmh": b.median() if len(b) >= MIN_LAPS else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lap", default="results/multi_rebased_mid/R")
    ap.add_argument("--runs", default="results/fia_fix_calendar_lc")
    args = ap.parse_args()
    t = per_race(Path(args.per_lap), Path(args.runs))
    print(t.round(2).to_string())
    rho = spearmanr(t.model_coast_mj, t.lift_share)[0]
    both = t.dropna(subset=["drop_lift_kmh", "drop_no_lift_kmh"])
    less = int((both.drop_lift_kmh < both.drop_no_lift_kmh).sum())
    print(f"\nRL1 Spearman(model coasting, real lift share) = {rho:.2f}: {'held' if rho >= 0.5 else 'not held'}")
    print(f"RL2 lifting laps lose less at full throttle in {less} of {len(both)} races: "
          f"{'held' if len(both) and less >= 0.7 * len(both) else 'not held'}")
    if t.lap_coast_m.notna().all():
        rho4 = spearmanr(t.model_coast_mj, t.lap_coast_m)[0]
        print(f"RL4 Spearman(model coasting, real whole-lap coast) = {rho4:.2f}: {'held' if rho4 >= 0.5 else 'not held'}")
    pre = t[t.cap_kw == 250]
    print(f"RL3 (reported only) lift share before Miami {pre.lift_share.round(2).tolist()} "
          f"against {t[t.cap_kw == 350].lift_share.median():.2f} after (median)")


if __name__ == "__main__":
    main()
