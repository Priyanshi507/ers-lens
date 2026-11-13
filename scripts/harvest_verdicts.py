"""H1-H3 for the harvest-limit correction (research log, 2026-11-04).

    python scripts/harvest_verdicts.py
Compares each *_h run with the same run made with the tapered harvest (*_fixed).
"""
import argparse
from pathlib import Path

import pandas as pd

TOL_S = 0.02
KEYS = {"fia_fix_calendar": ["event", "car", "cap_kw"], "fia_fix": ["event", "car", "cap_kw"],
        "sectors": ["event", "variant"]}


def slower(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    p = old.merge(new, on=keys, suffixes=("_old", "_new"))
    p = p[p.converged_old.astype(bool) & p.converged_new.astype(bool)].copy()
    p["delta_s"] = p.lap_s_new - p.lap_s_old
    return p


def straight_energy(results: Path, name: str) -> pd.Series:
    s = pd.read_csv(results / name / "location_split.csv")
    s = s[(s.car == "original") & (s.cap_kw == 350.0)]
    return s.set_index("event").longest_straight_mj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    r = Path(args.results)

    worst = []
    for name, keys in KEYS.items():
        p = slower(pd.read_csv(r / f"{name}_fixed" / "runs.csv"), pd.read_csv(r / f"{name}_h" / "runs.csv"), keys)
        bad = p[p.delta_s > TOL_S]
        print(f"{name}: {len(p)} converged pairs, {len(bad)} slower by more than {TOL_S} s, "
              f"largest change {p.delta_s.max():+.3f} s, mean {p.delta_s.mean():+.3f} s")
        if len(bad):
            print(bad[keys + ["lap_s_old", "lap_s_new", "delta_s"]].round(3).to_string(index=False))
        worst.append(len(bad))
    print(f"H1: {'held' if sum(worst) == 0 else 'NOT held'}")

    old, new = straight_energy(r, "fia_fix_calendar_fixed"), straight_energy(r, "fia_fix_calendar_h")
    t = pd.DataFrame({"old_mj": old, "new_mj": new})
    t = t[(t.old_mj > 0) | (t.new_mj > 0)]
    up = int((t.new_mj > t.old_mj).sum())
    print(t.round(3).to_string())
    print(f"H2: higher at {up} of {len(t)} circuits that super-clip on the longest straight -> "
          f"{'held' if up >= 0.8 * len(t) else 'NOT held'}")


if __name__ == "__main__":
    main()
