"""Limit advisor: the qualifying recharge limit that keeps super-clipping within the FIA's target,
circuit by circuit, against the limits the FIA chose. Predictions LA1-LA3 (research log, 2026-11-21).

    python scripts/limit_advisor.py --root results/advisor
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import spearmanr

LIMITS = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
TARGET_S = 4.0
MONO_TOL_S = 0.2
BEFORE_MIAMI = {"Australia", "China", "Japan"}


def load(root: Path) -> pd.DataFrame:
    frames = []
    for mj in LIMITS:
        path = root / f"q_{int(mj * 10)}" / "runs.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(limit_mj=mj))
    r = pd.concat(frames)
    r["ok"] = r.converged.astype(str).eq("True")
    return r


def fia_limits(rules: Path) -> dict:
    return {e: float(v) / 1e6 for e, v in yaml.safe_load(open(rules))["qualifying"].items()}


def advise(r: pd.DataFrame) -> pd.Series:
    ok = r[r.ok]
    out = {}
    for e, g in ok.groupby("event"):
        within = g[g.superclip_s <= TARGET_S].limit_mj
        out[e] = within.max() if len(within) else min(LIMITS)
    return pd.Series(out, name="advised_mj")


def verdicts(r: pd.DataFrame, fia: dict) -> dict:
    events = sorted(r.event.unique())
    mono = 0
    for e in events:
        g = r[(r.event == e)].sort_values("limit_mj")
        mono += int(g.ok.all() and (g.superclip_s.diff().dropna() >= -MONO_TOL_S).all())
    advised = advise(r)
    scored = [e for e in events if e in fia and e not in BEFORE_MIAMI and e in advised.index]
    rho = spearmanr([advised[e] for e in scored], [fia[e] for e in scored]).statistic
    at_fia = r[r.ok & r.apply(lambda x: x.event in scored and abs(x.limit_mj - fia[x.event]) < 1e-9, axis=1)]
    meets = int((at_fia.superclip_s <= TARGET_S).sum())
    return {"LA1": (mono, len(events), mono >= len(events) - 2),
            "LA2": (round(float(rho), 2), len(scored), bool(rho >= 0.5)),
            "LA3": (meets, len(scored), meets >= 12),
            "advised": advised, "scored": scored}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/advisor")
    ap.add_argument("--rules", default="configs/rules_2026_races.yaml")
    args = ap.parse_args()
    r = load(Path(args.root))
    fia = fia_limits(Path(args.rules))
    v = verdicts(r, fia)
    table = r.pivot_table(index="event", columns="limit_mj", values="superclip_s").round(1)
    table["advised"] = v["advised"]
    table["fia"] = pd.Series(fia)
    print(table.to_string())
    n, t, held = v["LA1"]
    print(f"LA1: super-clipping does not fall as the limit rises at {n} of {t} -> {'held' if held else 'NOT held'}")
    rho, t, held = v["LA2"]
    print(f"LA2: Spearman advised vs FIA qualifying limit {rho} over {t} circuits -> {'held' if held else 'NOT held'}")
    n, t, held = v["LA3"]
    print(f"LA3: at the FIA's limit super-clipping is {TARGET_S:.0f} s or less at {n} of {t} -> {'held' if held else 'NOT held'}")


if __name__ == "__main__":
    main()
