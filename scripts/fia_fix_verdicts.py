"""Verdicts F1-F4 for the FIA-fix experiment (research log, 2026-11-01), from results/fia_fix/runs.csv.

    python scripts/fia_fix_verdicts.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/fia_fix/runs.csv")
    v = verdicts(pd.read_csv(ap.parse_args().runs))
    cols = ["both_converged", "gain_s", "longest_superclip_s_250", "longest_superclip_s_350",
            "superclip_mj_250", "superclip_mj_350", "superclip_s_250", "superclip_s_350"]
    print(v["pairs"][cols].round(3).to_string())
    for k in ("F1", "F2", "F3", "F4"):
        n, total, held = v[k]
        print(f"{k}: {n} of {total} -> {'held' if held else 'NOT held'}")


if __name__ == "__main__":
    main()
