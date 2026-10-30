"""Do FastF1 and OpenF1 give the same findings 3 and 6 on the seven races both sources hold?

Compares the committed FastF1 results (results/multi/R, results/laptime) with the same scripts run
on OpenF1 data (ERSLENS_SOURCE=openf1, into results/source_check). The tolerances were recorded in
the research log on 2026-10-30 before any OpenF1 run. OpenF1 data is used for the season test only
if every check passes.

    python scripts/compare_sources.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# (column, tolerance, kind): "rel" = relative difference, "abs" = absolute, "same" = must be equal.
MULTI_CHECKS = (("clean_laps", 0.10, "rel"), ("clip_share", 0.03, "abs"),
                ("clip_frac_mean", 0.02, "abs"), ("drop_kmh_median", 1.0, "abs"))
LAPTIME_CHECKS = (("measurable_share", 0.05, "abs"), ("valid", None, "same"), ("t_loss_median_s", 0.02, "abs"))


def compare(ref: pd.DataFrame, new: pd.DataFrame, checks, valid_only: set | None = None) -> pd.DataFrame:
    rows = []
    both = ref.merge(new, on="event", how="outer", suffixes=("_fastf1", "_openf1"), indicator=True)
    for r in both.to_dict("records"):
        if r["_merge"] != "both":
            rows.append({"event": r["event"], "check": "present in both", "fastf1": r["_merge"] != "right_only",
                         "openf1": r["_merge"] != "left_only", "passed": False})
            continue
        for col, tol, kind in checks:
            a, b = r[f"{col}_fastf1"], r[f"{col}_openf1"]
            if valid_only is not None and col == "t_loss_median_s" and r["event"] not in valid_only:
                continue
            if kind == "same":
                ok = bool(a) == bool(b)
            elif pd.isna(a) or pd.isna(b):
                ok = pd.isna(a) and pd.isna(b)
            else:
                diff = abs(a - b) / abs(a) if kind == "rel" and a else abs(a - b)
                ok = diff <= tol
            rows.append({"event": r["event"], "check": col, "fastf1": a, "openf1": b, "passed": ok})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastf1-multi", default="results/multi/R/race_summary.csv")
    ap.add_argument("--openf1-multi", default="results/source_check/multi/R/race_summary.csv")
    ap.add_argument("--fastf1-laptime", default="results/laptime/laptime_per_race.csv")
    ap.add_argument("--openf1-laptime", default="results/source_check/laptime/laptime_per_race.csv")
    ap.add_argument("--out", default="results/source_check/comparison.md")
    args = ap.parse_args()

    multi = compare(pd.read_csv(args.fastf1_multi), pd.read_csv(args.openf1_multi), MULTI_CHECKS)
    lt_ref = pd.read_csv(args.fastf1_laptime)
    laptime = compare(lt_ref, pd.read_csv(args.openf1_laptime), LAPTIME_CHECKS,
                      valid_only=set(lt_ref.loc[lt_ref["valid"], "event"]))
    table = pd.concat([multi.assign(finding="3"), laptime.assign(finding="6")], ignore_index=True)
    passed = bool(table["passed"].all())
    lines = ["# FastF1 against OpenF1 on the seven original races", "",
             f"**{'All checks passed' if passed else 'FAILED'}**: {int(table['passed'].sum())} of {len(table)}.", "",
             "| Finding | Race | Check | FastF1 | OpenF1 | Passed |", "|---|---|---|---|---|---|"]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines += [f"| {r.finding} | {r.event} | {r.check} | {fmt(r.fastf1)} | {fmt(r.openf1)} | {'yes' if r.passed else 'NO'} |"
              for r in table.itertuples()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
