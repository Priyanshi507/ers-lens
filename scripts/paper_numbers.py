"""Every headline number of the paper from committed results. No GPU, no download; about ten seconds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

import hindcast_2026
import quali_test
from fia_fix_verdicts import pairs

R = Path(__file__).parents[1] / "results"
BOOT = 10000


def sign_test(k: int, n: int) -> dict:
    b = binomtest(k, n, 0.5)
    ci = b.proportion_ci(0.95, method="exact")
    return {"k": k, "n": n, "p": float(b.pvalue), "low": float(ci.low), "high": float(ci.high)}


def spearman_interval(x, y, seed: int = 0) -> tuple[float, float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(BOOT):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            out.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(spearmanr(x, y).statistic), float(lo), float(hi)


def numbers(results: Path = R) -> dict:
    p = pairs(pd.read_csv(results / "fia_fix_calendar_c/runs.csv").query("car == 'original'"))
    ok = p.both_converged
    lc = pairs(pd.read_csv(results / "fia_fix_calendar_lc/runs.csv").query("car == 'original'"))
    coasts = lc[lc.both_converged & (lc.coast_mj_250 > 0.05)]
    h = hindcast_2026.table(results / "fia_fix_calendar_c", results / "multi_rebased_mid/R/race_summary.csv")
    clip_match = int(((h.model_drop_kmh >= 3) == (h.clip_share >= 0.5)).sum())
    q = quali_test.table(results / "quali_push", results / "quali_neutral", results / "fia_fix_calendar_c",
                         results / "quali_real/Q/race_summary.csv", results / "multi_rebased_mid/R/race_summary.csv")
    c = q[(q.real_quali - q.real_race).abs() >= quali_test.CONTRAST_KMH]
    same = int((np.sign(c.model_push - c.model_race) == np.sign(c.real_quali - c.real_race)).sum())
    f = pd.read_csv(results / "forecast_2027.csv")
    return {"C1": sign_test(int((ok & (p.superclip_mj_350 > p.superclip_mj_250)).sum()), len(p)),
            "LC2": sign_test(int((coasts.coast_mj_350 < coasts.coast_mj_250).sum()), len(coasts)),
            "coast_mj": (float(lc.coast_mj_250.sum()), float(lc.coast_mj_350.sum())),
            "superclip_mj": (float(lc.superclip_mj_250.sum()), float(lc.superclip_mj_350.sum())),
            "Y1": spearman_interval(h.model_drop_kmh, h.observed_drop_kmh),
            "Y2": sign_test(clip_match, len(h)),
            "QL1": spearman_interval(q.model_push, q.real_quali),
            "QL4": sign_test(same, len(c)),
            "forecast": f.direction.value_counts().to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(R))
    n = numbers(Path(ap.parse_args().results))
    for key in ("C1", "LC2", "Y2", "QL4"):
        s = n[key]
        print(f"{key}: {s['k']} of {s['n']}, sign test p = {s['p']:.3g}, 95% interval {s['low']:.2f}-{s['high']:.2f}")
    for key in ("Y1", "QL1"):
        rho, lo, hi = n[key]
        print(f"{key}: Spearman {rho:.2f}, bootstrap 95% interval {lo:.2f} to {hi:.2f}")
    print("lift-and-coast energy, 250 -> 350 kW: %.1f -> %.1f MJ" % n["coast_mj"])
    print("super-clipping energy with lift-and-coast, 250 -> 350 kW: %.1f -> %.1f MJ" % n["superclip_mj"])
    print("2027 forecast:", n["forecast"])


if __name__ == "__main__":
    main()
