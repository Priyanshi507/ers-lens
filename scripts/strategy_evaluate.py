"""Check the pre-registered predictions S1-S7 against results/strategy/runs.csv. Fast.

Thresholds are copied from the 2026-10-05 research log entry and must not be edited after
results exist. Only results passing the convergence rule (erslens.strategy.CONVERGED_S) are
used; a case with no converged result counts against the prediction it belongs to, never for
it. Writes results/strategy/verdicts.md.

    python scripts/strategy_evaluate.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from erslens.strategy import CONVERGED_S, REFINED_TAG, resolve_runs

sys.path.insert(0, "scripts")

CLIP_MAX = 0.5          # S1-S3: power at the speed peak at most 50% of its pre-peak mean
SWEEP_SHARE = 0.8       # S2
NO_TAPER_CIRCUITS = 3   # S3: at least 3 of 4
DEPLOY_GAP_KMH = 20.0   # S4
SIMPLE_GAP_S = 0.3      # S5
AGREE_S, AGREE_CORR = 0.02, 0.9  # S6
IQR_CIRCUITS = 3        # S7


def observed_peaks(event: str, year: int) -> np.ndarray:
    from erslens.ingest import event_info
    from export_site import race_id

    name, _ = event_info(year, event)
    data = json.loads(Path(f"site/data/races/{race_id(name)}.json").read_text())
    return np.array([lap["trace"]["d"][lap["trace"]["peak"]] - lap["trace"]["d"][lap["trace"]["flat"][0]]
                     for lap in data["laps"] if lap["clipping"] and "trace" in lap], dtype=float)


def clips(series: pd.Series) -> pd.Series:
    """Missing (excluded) results are False, so they can only count against a prediction."""
    return series.le(CLIP_MAX).fillna(False)


def num(x, fmt=".2f", unit=""):
    return "excluded" if pd.isna(x) else f"{x:{fmt}}{unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/strategy/runs.csv")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    runs = pd.read_csv(args.runs)
    used, excluded = resolve_runs(runs)
    events = list(dict.fromkeys(runs.loc[runs.variant == "nominal", "event"]))
    get = lambda variant: used[used.variant == variant].set_index("event").reindex(events)
    nominal, lines, extra = get("nominal"), [], []

    def report(key, held, detail):
        lines.append(f"| {key} | {'held' if held else 'NOT held'} | {detail} |")

    m1 = used[~used.variant.str.startswith(("m2", "simple_"))]
    n_cases = len(m1) + len(excluded)
    lines += ["| Check | Result | Detail |", "|---|---|---|"]
    report(f"Convergence rule (<= {CONVERGED_S} s)", excluded.empty,
           f"{len(m1)} of {n_cases} M1 results converged, {int(m1.refined.sum())} after grid refinement; "
           f"{len(excluded)} excluded" + (f" ({', '.join(excluded.event + ' ' + excluded.variant + ' ' + excluded.car_id)})"
                                          if len(excluded) else ""))

    report("S1", bool(clips(nominal.clip_ratio).all()),
           ", ".join(f"{e} {num(nominal.clip_ratio[e])}" for e in events))

    sweep_runs = runs[runs.variant == "sweep"]
    sweep_all = sweep_runs.car_id.str.split(REFINED_TAG, regex=False).str[0].groupby(sweep_runs.event).nunique()
    sweep = used[used.variant == "sweep"]
    n_clip = sweep.groupby("event").clip_ratio.apply(lambda c: int(clips(c).sum())).reindex(events).fillna(0)
    share = n_clip / sweep_all.reindex(events)
    report("S2", bool((share >= SWEEP_SHARE).all()),
           ", ".join(f"{e} {int(n_clip[e])} of {int(sweep_all.get(e, 0))} ({share[e]:.0%})" for e in events)
           + "; excluded cars counted as not clipping")

    nt = get("no_taper")
    need = min(NO_TAPER_CIRCUITS, len(events))
    report("S3", int(clips(nt.clip_ratio).sum()) >= need,
           f"{int(clips(nt.clip_ratio).sum())} of {len(events)} circuits clip without the taper: "
           + ", ".join(f"{e} {num(nt.clip_ratio[e])}" for e in events))

    gap = nominal.full_throttle_mean_kmh - nominal.deploy_weighted_kmh
    report("S4", bool(gap.ge(DEPLOY_GAP_KMH).all()),
           ", ".join(f"{e} {num(gap[e], ".0f", " km/h")} lower" for e in events))

    simple = runs[runs.variant.str.startswith("simple_")].groupby("event").lap_s.min().reindex(events)
    s5 = simple - nominal.lap_s
    report("S5", bool(((s5 >= 0) & (s5 < SIMPLE_GAP_S)).all()),
           ", ".join(f"{e} optimum faster by {num(s5[e], ".3f", " s")}" for e in events))

    m2 = get("m2")
    d6 = m2.lap_s - nominal.lap_s
    report("S6", bool(((d6.abs() <= AGREE_S) & (m2.corr_with_nominal >= AGREE_CORR)).all()),
           ", ".join(f"{e} M2-M1 {d6[e]:+.3f} s, r={m2.corr_with_nominal[e]:.2f}" for e in events))

    if "demo" not in events:
        inside = []
        for e in events:
            q25, q75 = np.percentile(observed_peaks(e, args.year), [25, 75])
            opt = nominal.peak_from_start_m[e]
            inside.append(q25 <= opt <= q75)
            lines.append(f"|  | S7 {e} | optimum {opt:.0f} m, observed IQR {q25:.0f}-{q75:.0f} m |")
        report("S7 (exploratory)", sum(inside) >= IQR_CIRCUITS, f"{sum(inside)} of {len(events)} inside")

    if (runs.variant == "rules").any():
        rules = get("rules")
        extra += ["", "**Corrected 2026 race rules** (configs/rules_2026_races.yaml; logged 2026-10-07, "
                  "before these runs). Same thresholds as above.", "",
                  "| Variant | Circuit | Clip ratio | Clips? | Deploy speed gap | Lap vs original | Convergence |",
                  "|---|---|---|---|---|---|---|"]
        for variant in ("rules", "rules_no_taper", "rules_h8", "rules_deploy250"):
            v = get(variant)
            for e in events:
                r = v.loc[e]
                if pd.isna(r.lap_s):
                    if (runs.variant.eq(variant) & runs.event.eq(e)).any():
                        extra.append(f"| {variant} | {e} | excluded | - | - | - | not converged |")
                    continue
                note = " (refined grid)" if r.refined else ""
                extra.append(f"| {variant} | {e} | {r.clip_ratio:.2f} | {'yes' if r.clip_ratio <= CLIP_MAX else 'no'} | "
                             f"{r.full_throttle_mean_kmh - r.deploy_weighted_kmh:.0f} km/h | "
                             f"{r.lap_s - nominal.lap_s[e]:+.3f} s | {abs(r.dp_pred_s - r.lap_s):.3f} s{note} |")
        nt_r = get("rules_no_taper")
        report("S1 (corrected rules)", bool(clips(rules.clip_ratio).all()),
               ", ".join(f"{e} {num(rules.clip_ratio[e])}" for e in events))
        report("S3 (corrected rules)", int(clips(nt_r.clip_ratio).sum()) >= need,
               f"{int(clips(nt_r.clip_ratio).sum())} of {len(events)} clip without the taper: "
               + ", ".join(f"{e} {num(nt_r.clip_ratio[e])}" for e in events))
        gap_r = rules.full_throttle_mean_kmh - rules.deploy_weighted_kmh
        report("S4 (corrected rules)", bool(gap_r.ge(DEPLOY_GAP_KMH).all()),
               ", ".join(f"{e} {num(gap_r[e], ".0f", " km/h")}" for e in events))

    soc = used[used.variant.str.startswith("soc0_")]
    if len(soc):
        spread = soc.groupby("event").lap_s.agg(lambda s: s.max() - s.min())
        lines.append("| Starting charge | sensitivity | " + ", ".join(
            f"{e} {spread[e]:.3f} s between 30% and 70%" for e in spread.index) + " |")

    text = "\n".join(lines + extra)
    print(text)
    Path(args.runs).with_name("verdicts.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
