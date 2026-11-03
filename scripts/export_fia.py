"""Export the FIA-fix experiment to site/data/fia.json for the site's FIA-fix page.

Reads only committed results: results/fia_fix_calendar_fixed (18 circuits, original car, 250 and 350 kW,
recharge limits from the FIA's per-race documents)
and results/fia_fix_fixed (four circuits, three cars, corrected solver) for the F1-F4 verdicts. Every energy on the page
is recomputed from the saved profiles and checked against runs.csv, so the page cannot drift from
the research log.

    python scripts/export_fia.py
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fia_fix_figures import location_split, longest_free_run  # noqa: E402
from fia_fix_verdicts import calendar_verdicts, verdicts  # noqa: E402

SCHEMA_VERSION = 1
OLD, NEW = 250, 350
PREDICTIONS = (
    ("C1", "Super-clipping harvests more energy under 350 kW, at 80% or more of 18 circuits"),
    ("C2", "More of that energy is taken below 200 km/h, at 80% or more of them"),
    ("C3", "No circuit's optimal lap is slower by more than 0.02 s"),
    ("F1", "Four circuits, three cars: the lap is never slower and gains less than 0.20 s"),
    ("F2", "Four circuits, three cars: the longest super-clipping stretch is shorter, at 10 of 12 or more"),
    ("F3", "Four circuits, three cars: super-clipping harvests more energy, at 10 of 12 or more"),
    ("F4", "Cap swept 150-450 kW: super-clipping energy never steps down"),
)


def label(event: str) -> str:
    return event.replace(" Grand Prix", "")


def circuit(d: Path, runs: pd.DataFrame, split: pd.DataFrame, event: str, assumed: dict) -> dict:
    rows = runs[(runs.event == event) & (runs.car == "original")].set_index("cap_kw")
    profiles = {cap: np.load(d / "profiles" / f"{event}_original_{cap}.npz") for cap in (OLD, NEW)}
    for cap, z in profiles.items():
        mj = float((z["superclip"] * z["dt"]).sum() / 1e6)
        if abs(mj - rows.loc[float(cap), "superclip_mj"]) > 1e-9:
            raise ValueError(f"{event} {cap} kW: profile gives {mj} MJ, runs.csv {rows.loc[float(cap), 'superclip_mj']}")
    z = profiles[OLD]
    a, b = longest_free_run(z["v_limit"])
    s = split[(split.event == event) & (split.car == "original")].set_index("cap_kw")
    below = lambda cap: float(s.loc[float(cap), "below_200_kmh_mj"] + s.loc[float(cap), "200_250_kmh_mj"])
    per = lambda cap: {
        "lap_s": round(float(rows.loc[float(cap), "lap_s"]), 3),
        "converged": bool(rows.loc[float(cap), "converged"]),
        "superclip_mj": round(float(rows.loc[float(cap), "superclip_mj"]), 3),
        "superclip_s": round(float(rows.loc[float(cap), "superclip_s"]), 2),
        "longest_superclip_s": round(float(rows.loc[float(cap), "longest_superclip_s"]), 2),
        "below_250_mj": round(below(cap), 3),
        "above_250_mj": round(float(s.loc[float(cap), "above_250_kmh_mj"]), 3),
        "straight_mj": round(float(s.loc[float(cap), "longest_straight_mj"]), 3),
        "v_kmh": np.round(profiles[cap]["v"] * 3.6, 1).tolist(),
        "superclip_kw": np.round(profiles[cap]["superclip"] / 1e3).astype(int).tolist(),
    }
    return {"event": event, "label": label(event), "assumed_recharge_mj": assumed.get(event),
            "ds_m": float(z["distance_m"][1] - z["distance_m"][0]), "straight": [int(a), int(b)],
            "caps": {str(OLD): per(OLD), str(NEW): per(NEW)}}


def build(calendar: Path, four: Path) -> dict:
    runs = pd.read_csv(calendar / "runs.csv")
    split = location_split(calendar / "profiles", runs)
    assumed = dict(pd.read_csv(calendar / "assumptions.csv")[["event", "recharge_mj"]].values) \
        if (calendar / "assumptions.csv").exists() else {}
    c, f = calendar_verdicts(runs, split), verdicts(pd.read_csv(four / "runs.csv"))
    results = {**{k: c[k] for k in ("C1", "C2", "C3")}, **{k: f[k] for k in ("F1", "F2", "F3", "F4")}}
    events = list(dict.fromkeys(runs.event))
    return {
        "schema": SCHEMA_VERSION,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "predictions": [{"id": k, "text": t, "count": int(results[k][0]), "total": int(results[k][1]),
                         "held": bool(results[k][2])} for k, t in PREDICTIONS],
        "straight_down": {"count": c["straight_down"][0], "total": c["straight_down"][1]},
        "circuits": [circuit(calendar, runs, split, e, assumed) for e in events],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calendar", default="results/fia_fix_calendar_fixed")
    ap.add_argument("--four", default="results/fia_fix_fixed")
    ap.add_argument("--out", default="site/data/fia.json")
    args = ap.parse_args()
    data = build(Path(args.calendar), Path(args.four))
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {args.out}: {len(data['circuits'])} circuits, {Path(args.out).stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
