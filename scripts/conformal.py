"""Honest error bars for the instant advisor: nested leave-one-circuit-out conformal calibration.
Predictions CP1-CP2 (research log, 2026-11-21).

For each held-out circuit, the interval width is calibrated only on the other 19 circuits, by an inner
leave-one-circuit-out of the same network ensemble, so the held-out circuit never informs its own interval.

    python scripts/conformal.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from limit_advisor import load  # noqa: E402
from surrogate import dataset, ensemble  # noqa: E402

LEVEL = 0.90


def scale(scores: np.ndarray, level: float = LEVEL) -> float:
    """Split-conformal quantile with the finite-sample correction."""
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(scores)[k - 1])


def inner_scores(train: pd.DataFrame, fit=ensemble) -> np.ndarray:
    out = []
    for e in sorted(train.event.unique()):
        tr, te = train[train.event != e], train[train.event == e]
        mean, sd = fit(tr, te)
        out.append(np.abs(te.superclip_s.to_numpy(float) - mean) / sd)
    return np.concatenate(out)


def nested(data: pd.DataFrame, fit=ensemble) -> pd.DataFrame:
    parts = []
    for e in sorted(data.event.unique()):
        train, test = data[data.event != e], data[data.event == e].copy()
        q = scale(inner_scores(train, fit))
        mean, sd = fit(train, test)
        test["mean"], test["half_width"], test["q"] = mean, q * sd, q
        parts.append(test)
    return pd.concat(parts)


def verdicts(pred: pd.DataFrame) -> dict:
    inside = (pred.superclip_s - pred["mean"]).abs() <= pred.half_width
    cover = float(inside.mean())
    half = float(pred.half_width.median())
    return {"CP1": (round(cover, 3), 0.80 <= cover <= 0.97), "CP2": (round(half, 2), half <= 5.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/advisor")
    ap.add_argument("--tracks", default="results/tracks")
    ap.add_argument("--out", default="results/surrogate")
    args = ap.parse_args()
    pred = nested(dataset(load(Path(args.root)), Path(args.tracks)))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pred.to_csv(Path(args.out) / "conformal_predictions.csv", index=False)
    v = verdicts(pred)
    print(f"CP1: 90% conformal intervals cover {v['CP1'][0]} of held-out rows (bar 0.80-0.97) -> "
          f"{'held' if v['CP1'][1] else 'NOT held'}")
    print(f"CP2: median half-width {v['CP2'][0]} s (bar 5.0) -> {'held' if v['CP2'][1] else 'NOT held'}")


if __name__ == "__main__":
    main()
