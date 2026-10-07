"""Check the pre-registered predictions S1-S7 against results/strategy/runs.csv. Fast.

Thresholds are copied from the 2026-10-05 research log entry and must not be edited after
results exist. Writes results/strategy/verdicts.md.

    python scripts/strategy_evaluate.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

CLIP_MAX = 0.5          # S1-S3: power at the speed peak at most 50% of its pre-peak mean
SWEEP_SHARE = 0.8       # S2
NO_TAPER_CIRCUITS = 3   # S3: at least 3 of 4
DEPLOY_GAP_KMH = 20.0   # S4
SIMPLE_GAP_S = 0.3      # S5
AGREE_S, AGREE_CORR = 0.02, 0.9  # S6
IQR_CIRCUITS = 3        # S7
CONVERGED_S = 0.05      # DP prediction vs replay, reported for every fine run


def observed_peaks(event: str, year: int) -> np.ndarray:
    from erslens.ingest import event_info
    from export_site import race_id

    name, _ = event_info(year, event)
    data = json.loads(Path(f"site/data/races/{race_id(name)}.json").read_text())
    return np.array([lap["trace"]["d"][lap["trace"]["peak"]] - lap["trace"]["d"][lap["trace"]["flat"][0]]
                     for lap in data["laps"] if lap["clipping"] and "trace" in lap], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/strategy/runs.csv")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    runs = pd.read_csv(args.runs)
    events = list(dict.fromkeys(runs.loc[runs.variant == "nominal", "event"]))
    get = lambda variant: runs[runs.variant == variant].set_index("event")
    nominal, lines, verdicts = get("nominal"), [], {}

    def report(key, held, detail):
        verdicts[key] = held
        lines.append(f"| {key} | {'held' if held else 'NOT held'} | {detail} |")

    conv = (nominal.dp_pred_s - nominal.lap_s).abs()
    lines += ["| Check | Result | Detail |", "|---|---|---|"]
    report("DP converged", bool((conv <= CONVERGED_S).all()),
           ", ".join(f"{e} {conv[e]:.3f} s" for e in events))

    report("S1", bool((nominal.clip_ratio <= CLIP_MAX).all()),
           ", ".join(f"{e} {nominal.clip_ratio[e]:.2f}" for e in events))

    sweep = runs[runs.variant == "sweep"]
    share = sweep.groupby("event").clip_ratio.apply(lambda c: float((c <= CLIP_MAX).mean()))
    n = sweep.groupby("event").size()
    report("S2", bool(len(share) == len(events) and (share >= SWEEP_SHARE).all()),
           ", ".join(f"{e} {share.get(e, np.nan):.0%} of {n.get(e, 0)}" for e in events))

    nt = get("no_taper")
    clipping = int((nt.clip_ratio <= CLIP_MAX).sum())
    report("S3", clipping >= min(NO_TAPER_CIRCUITS, len(events)),
           f"{clipping} of {len(nt)} circuits clip without the taper: "
           + ", ".join(f"{e} {nt.clip_ratio[e]:.2f}" for e in nt.index))

    gap = nominal.full_throttle_mean_kmh - nominal.deploy_weighted_kmh
    report("S4", bool((gap >= DEPLOY_GAP_KMH).all()),
           ", ".join(f"{e} {gap[e]:.0f} km/h lower" for e in events))

    simple = runs[runs.variant.str.startswith("simple_")].groupby("event").lap_s.min()
    s5 = simple - nominal.lap_s
    report("S5", bool(((s5 >= 0) & (s5 < SIMPLE_GAP_S)).all()),
           ", ".join(f"{e} optimum faster by {s5[e]:.3f} s" for e in events))

    m2 = get("m2")
    d6 = (m2.lap_s - nominal.lap_s)
    report("S6", bool(((d6.abs() <= AGREE_S) & (m2.corr_with_nominal >= AGREE_CORR)).all()),
           ", ".join(f"{e} M2-M1 {d6[e]:+.3f} s, r={m2.corr_with_nominal[e]:.2f}" for e in m2.index))

    if "demo" not in events:
        inside = []
        for e in events:
            q25, q75 = np.percentile(observed_peaks(e, args.year), [25, 75])
            opt = nominal.peak_from_start_m[e]
            inside.append(q25 <= opt <= q75)
            lines.append(f"|  | S7 {e} | optimum {opt:.0f} m, observed IQR {q25:.0f}-{q75:.0f} m |")
        report("S7 (exploratory)", sum(inside) >= IQR_CIRCUITS, f"{sum(inside)} of {len(events)} inside")

    soc = runs[runs.variant.str.startswith("soc0_")]
    if len(soc):
        spread = soc.groupby("event").lap_s.agg(lambda s: s.max() - s.min())
        lines.append("| Starting charge | sensitivity | " + ", ".join(
            f"{e} {spread[e]:.3f} s between 30% and 70%" for e in spread.index) + " |")

    text = "\n".join(lines)
    print(text)
    Path(args.runs).with_name("verdicts.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
