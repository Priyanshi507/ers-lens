"""Instant limit advisor: a network trained on the certified qualifying-lap solves predicts super-clipping
time from a circuit's shape and the recharge limit. Predictions ML1-ML4 (research log, 2026-11-21).

    python scripts/surrogate.py                       # leave-one-circuit-out on results/advisor
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from limit_advisor import BEFORE_MIAMI, LIMITS, TARGET_S, fia_limits, load  # noqa: E402

from erslens import gaussnet  # noqa: E402

SLOW_MS = 150 / 3.6
SEEDS = (0, 1, 2, 3, 4)
NET = {"hidden": (32, 32), "epochs": 300, "batch": 16, "lr": 3e-3}


def runs_of(mask: np.ndarray) -> list:
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def track_features(path: Path) -> dict:
    z = np.load(path)
    d, v = z["distance_m"], z["v_limit_ms"]
    step = np.diff(d, append=d[-1] + (d[-1] - d[-2]))
    free = np.isinf(v)
    straights = [step[i:j].sum() for i, j in runs_of(free)]
    corners = [v[i:j].min() for i, j in runs_of(~free)]
    return {"lap_km": d[-1] / 1000, "free_share": float(step[free].sum() / step.sum()),
            "longest_km": max(straights) / 1000, "long_straights": sum(s >= 300 for s in straights),
            "slow_corners": sum(c < SLOW_MS for c in corners)}


FEATURES = ["lap_km", "free_share", "longest_km", "long_straights", "slow_corners", "limit_mj"]


def dataset(r: pd.DataFrame, tracks: Path) -> pd.DataFrame:
    f = pd.DataFrame({e: track_features(tracks / f"{e}.npz") for e in r.event.unique()}).T
    return r[r.ok].join(f, on="event")[["event", "superclip_s"] + FEATURES].reset_index(drop=True)


def ensemble(train: pd.DataFrame, test: pd.DataFrame):
    mus, sds = [], []
    for s in SEEDS:
        fit = gaussnet.fit(train[FEATURES].to_numpy(float), train.superclip_s.to_numpy(float), seed=s, **NET)
        mu, sd = gaussnet.predict(fit, test[FEATURES].to_numpy(float))
        mus.append(mu)
        sds.append(sd)
    mus, sds = np.array(mus), np.array(sds)
    mean = mus.mean(0)
    return mean, np.sqrt((sds ** 2 + mus ** 2).mean(0) - mean ** 2)


def linear(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x, xt = train[FEATURES].to_numpy(float), test[FEATURES].to_numpy(float)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    a = np.c_[np.ones(len(x)), (x - mu) / sd]
    coef = np.linalg.solve(a.T @ a + 1e-3 * np.eye(a.shape[1]), a.T @ train.superclip_s.to_numpy(float))
    return np.c_[np.ones(len(xt)), (xt - mu) / sd] @ coef


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    med = train.groupby("limit_mj").superclip_s.median()
    return test.limit_mj.map(med).to_numpy(float)


def loco(data: pd.DataFrame, models=("baseline", "linear", "network")) -> pd.DataFrame:
    parts = []
    for e in sorted(data.event.unique()):
        train, test = data[data.event != e], data[data.event == e].copy()
        if "baseline" in models:
            test["baseline"] = baseline(train, test)
        if "linear" in models:
            test["linear"] = linear(train, test)
        if "network" in models:
            test["network"], test["network_sd"] = ensemble(train, test)
        parts.append(test)
    return pd.concat(parts)


def advised(pred: pd.DataFrame, column: str) -> pd.Series:
    out = {}
    for e, g in pred.groupby("event"):
        within = g[g[column] <= TARGET_S].limit_mj
        out[e] = within.max() if len(within) else min(LIMITS)
    return pd.Series(out)


def verdicts(pred: pd.DataFrame, fia: dict) -> dict:
    rmse = {m: float(np.sqrt(np.mean((pred[m] - pred.superclip_s) ** 2))) for m in ("baseline", "linear", "network")}
    z = norm.ppf(0.95)
    cover = float(np.mean(np.abs(pred.superclip_s - pred.network) <= z * pred.network_sd))
    a = advised(pred, "network")
    scored = [e for e in a.index if e in fia and e not in BEFORE_MIAMI]
    rho = float(spearmanr([a[e] for e in scored], [fia[e] for e in scored]).statistic)
    return {"rmse": rmse,
            "ML1": rmse["network"] <= 0.7 * rmse["baseline"],
            "ML2": (round(cover, 3), 0.80 <= cover <= 0.97),
            "ML3": (round(rho, 2), len(scored), rho >= 0.5),
            "ML4": rmse["network"] < rmse["linear"],
            "advised": a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/advisor")
    ap.add_argument("--tracks", default="results/tracks")
    ap.add_argument("--rules", default="configs/rules_2026_races.yaml")
    ap.add_argument("--out", default="results/surrogate")
    args = ap.parse_args()
    data = dataset(load(Path(args.root)), Path(args.tracks))
    pred = loco(data)
    v = verdicts(pred, fia_limits(Path(args.rules)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "loco_predictions.csv", index=False)
    full = data.drop_duplicates("event")
    t0 = time.perf_counter()
    ensemble(data, full)
    print(f"seconds to train five networks on all 20 circuits and advise: {time.perf_counter() - t0:.1f}")
    print("leave-one-circuit-out RMSE (s):", {k: round(x, 2) for k, x in v["rmse"].items()})
    print(f"ML1: network RMSE at most 0.7 x baseline -> {'held' if v['ML1'] else 'NOT held'}")
    print(f"ML2: 90% interval coverage {v['ML2'][0]} (bar 0.80-0.97) -> {'held' if v['ML2'][1] else 'NOT held'}")
    print(f"ML3: Spearman network-advised vs FIA {v['ML3'][0]} over {v['ML3'][1]} -> {'held' if v['ML3'][2] else 'NOT held'}")
    print(f"ML4: network RMSE below linear -> {'held' if v['ML4'] else 'NOT held'}")


if __name__ == "__main__":
    main()
