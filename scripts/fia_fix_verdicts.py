"""Verdicts for the FIA-fix experiment (research log, 2026-11-01).

    python scripts/fia_fix_verdicts.py                                   # F1-F4, results/fia_fix
    python scripts/fia_fix_verdicts.py --calendar --runs results/fia_fix_calendar/runs.csv   # C1-C3
"""
import argparse

import pandas as pd

OLD, NEW = 250.0, 350.0


def pairs(runs: pd.DataFrame) -> pd.DataFrame:
    a = runs[runs.cap_kw == OLD].set_index(["event", "car"])
    b = runs[runs.cap_kw == NEW].set_index(["event", "car"])
    p = a.join(b, lsuffix="_250", rsuffix="_350")
    p["both_converged"] = p.converged_250.astype(bool) & p.converged_350.astype(bool)
    p["gain_s"] = p.lap_s_250 - p.lap_s_350
    return p


def verdicts(runs: pd.DataFrame) -> dict:
    p = pairs(runs)
    ok = p.both_converged
    f1 = ok & (p.gain_s >= -0.02) & (p.gain_s < 0.20)
    f2 = ok & (p.longest_superclip_s_350 < p.longest_superclip_s_250)
    f3 = ok & (p.superclip_mj_350 > p.superclip_mj_250)
    sweep = runs[runs.car == "original"].sort_values("cap_kw")
    f4 = {e: bool(g.converged.astype(bool).all() and (g.superclip_mj.diff().dropna() >= -0.05).all())
          for e, g in sweep.groupby("event")}
    return {"pairs": p, "F1": (int(f1.sum()), len(p), bool(f1.all())),
            "F2": (int(f2.sum()), len(p), int(f2.sum()) >= 10), "F3": (int(f3.sum()), len(p), int(f3.sum()) >= 10),
            "F4": (sum(f4.values()), len(f4), all(f4.values()))}


def calendar_verdicts(runs: pd.DataFrame, split: pd.DataFrame) -> dict:
    """C1-C3, original car, every circuit: a pair counts only if both solves converged."""
    p = pairs(runs[runs.car == "original"])
    s = split[split.car == "original"]
    low = s.pivot_table(index="event", columns="cap_kw", values="below_200_kmh_mj")
    straight = s.pivot_table(index="event", columns="cap_kw", values="longest_straight_mj")
    p = p.reset_index().set_index("event")
    p["low_250"], p["low_350"] = low[OLD], low[NEW]
    p["straight_250"], p["straight_350"] = straight[OLD], straight[NEW]
    ok = p.both_converged
    n = len(p)
    c1 = ok & (p.superclip_mj_350 > p.superclip_mj_250)
    c2 = ok & (p.low_350 > p.low_250)
    c3 = ok & (p.gain_s >= -0.02)
    return {"pairs": p, "C1": (int(c1.sum()), n, c1.sum() >= 0.8 * n), "C2": (int(c2.sum()), n, c2.sum() >= 0.8 * n),
            "C3": (int(c3.sum()), n, bool(c3.all())),
            "straight_down": (int((ok & (p.straight_350 < p.straight_250)).sum()), n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/fia_fix/runs.csv")
    ap.add_argument("--calendar", action="store_true", help="C1-C3 over every circuit (original car)")
    args = ap.parse_args()
    runs = pd.read_csv(args.runs)
    if args.calendar:
        from pathlib import Path

        from fia_fix_figures import location_split

        d = Path(args.runs).parent
        split = location_split(d / "profiles", runs)
        split.to_csv(d / "location_split.csv", index=False)
        v = calendar_verdicts(runs, split)
        cols = ["both_converged", "gain_s", "superclip_mj_250", "superclip_mj_350", "low_250", "low_350",
                "straight_250", "straight_350"]
        print(v["pairs"][cols].round(3).to_string())
        for k, text in (("C1", "more super-clipping energy under 350 kW"), ("C2", "more of it below 200 km/h"),
                        ("C3", "no circuit slower by more than 0.02 s")):
            n, total, held = v[k]
            print(f"{k} ({text}): {n} of {total} -> {'held' if held else 'NOT held'}")
        print(f"Exploratory: less super-clipping energy on the longest straight at {v['straight_down'][0]} of "
              f"{v['straight_down'][1]}")
        return
    v = verdicts(runs)
    cols = ["both_converged", "gain_s", "longest_superclip_s_250", "longest_superclip_s_350",
            "superclip_mj_250", "superclip_mj_350", "superclip_s_250", "superclip_s_350"]
    print(v["pairs"][cols].round(3).to_string())
    for k in ("F1", "F2", "F3", "F4"):
        n, total, held = v[k]
        print(f"{k}: {n} of {total} -> {'held' if held else 'NOT held'}")


if __name__ == "__main__":
    main()
