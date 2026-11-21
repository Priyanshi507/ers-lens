"""What a megajoule of recharge is worth, circuit by circuit: predictions EP1-EP3 (research log, 2026-11-21).

    python scripts/energy_price.py --hi results/energy_price_85 --lo results/energy_price_80
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import spearmanr

TOL_S = 0.005


def load(folder: Path) -> pd.DataFrame:
    r = pd.read_csv(folder / "runs.csv")
    return r[r.cap_kw == 350.0].set_index("event")


def lower_limit_events(rules: Path, below_j: float = 8.5e6) -> set:
    events = yaml.safe_load(open(rules))["events"]
    return {e for e, v in events.items() if float(v["harvest_per_lap_j"]) < below_j}


def verdicts(hi: pd.DataFrame, lo: pd.DataFrame, lowered: set, step_mj: float = 0.5) -> dict:
    j = hi.join(lo, lsuffix="_hi", rsuffix="_lo", how="inner")
    ok = j.converged_hi.astype(str).eq("True") & j.converged_lo.astype(str).eq("True")
    cost = j.lap_s_lo - j.lap_s_hi
    inside = ok & (cost >= step_mj * j.price_s_per_mj_hi - TOL_S) & (cost <= step_mj * j.price_s_per_mj_lo + TOL_S)
    ep1 = (int(inside.sum()), len(j), int(inside.sum()) >= len(j) - 2)
    low = j[j.index.isin(lowered)].price_s_per_mj_hi
    rest = j[~j.index.isin(lowered)].price_s_per_mj_hi
    above = int((low > rest.median()).sum())
    ep2 = (round(low.median(), 3), round(rest.median(), 3), above, len(low),
           bool(low.median() > rest.median() and above >= len(low) - 1))
    rho = spearmanr(j.price_s_per_mj_hi, j.superclip_mj_hi).statistic
    ep3 = (round(float(rho), 2), bool(rho >= 0.6))
    return {"EP1": ep1, "EP2": ep2, "EP3": ep3, "table": j.assign(cost_s=cost)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", default="results/energy_price_85")
    ap.add_argument("--lo", default="results/energy_price_80")
    ap.add_argument("--rules", default="configs/rules_2026_races.yaml")
    args = ap.parse_args()
    v = verdicts(load(Path(args.hi)), load(Path(args.lo)), lower_limit_events(Path(args.rules)))
    t = v["table"].sort_values("price_s_per_mj_hi", ascending=False)
    print(t[["price_s_per_mj_hi", "price_s_per_mj_lo", "cost_s", "superclip_mj_hi"]].round(3).to_string())
    n, total, held = v["EP1"]
    print(f"EP1: half a MJ less costs between the two prices at {n} of {total} -> {'held' if held else 'NOT held'}")
    a, b, k, m, held = v["EP2"]
    print(f"EP2: median price {a} s/MJ where the FIA lowered the limit vs {b} elsewhere; {k} of {m} above -> "
          f"{'held' if held else 'NOT held'}")
    rho, held = v["EP3"]
    print(f"EP3: Spearman price vs super-clip energy {rho} -> {'held' if held else 'NOT held'}")


if __name__ == "__main__":
    main()
