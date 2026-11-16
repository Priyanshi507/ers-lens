"""Export every solved rule to site/data/rules.json for the site's rule-checker page.

One row per circuit and rule, from committed results only: the race lap under 250 and 350 kW with
and without lift-and-coast, the qualifying push lap under both caps, and the 2027 rules under four
speed curves and two caps; beside them the same quantity measured on real race and qualifying laps.
The speed loss is recomputed from each saved profile by hindcast_2026.drop_from_profile.

    python scripts/export_rules.py
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hindcast_2026 import FIRST_350, NAMES, drop_from_profile  # noqa: E402

SCHEMA_VERSION = 1
MIN_QUALI_LAPS = 10
CURVES = {"t280": "taper from 280 km/h", "t300": "taper from 300 km/h", "t320": "taper from 320 km/h",
          "none": "no speed taper"}
FIA = (("race", "results/fia_fix_calendar_c", "2026 race lap"),
       ("race_lc", "results/fia_fix_calendar_lc", "2026 race lap, may lift and coast"),
       ("quali", "results/quali_push", "2026 qualifying push lap"))


def fia_rules(root: Path):
    for key, folder, text in FIA:
        runs = pd.read_csv(root / folder / "runs.csv")
        for cap in (250, 350):
            rows = {}
            for r in runs[(runs.car == "original") & (runs.cap_kw == cap)].itertuples():
                rows[r.event] = row(r, root / folder / "profiles" / f"{r.event}_original_{cap}.npz")
            yield {"id": f"{key}_{cap}", "group": text, "label": f"{text}, {cap} kW super-clip limit",
                   "real": "quali" if key == "quali" else "race", "rows": rows}


def rules_2027(root: Path):
    runs = pd.concat([pd.read_csv(root / f"results/forecast_2027_{s}/runs.csv").assign(folder=f"results/forecast_2027_{s}")
                      for s in "ab"], ignore_index=True)
    for curve, text in CURVES.items():
        for cap in (350, 375):
            rows = {}
            for r in runs[(runs.curve == curve) & (runs.superclip_kw == cap)].itertuples():
                rows[r.event] = row(r, root / r.folder / "profiles" / f"{r.event}_{curve}_{cap}.npz")
            yield {"id": f"y2027_{curve}_{cap}", "group": "2027 race lap",
                   "label": f"2027 race lap, {text}, {cap} kW super-clip limit", "real": None, "rows": rows}


def row(r, profile: Path) -> dict:
    return {"lap_s": round(float(r.lap_s), 3), "superclip_mj": round(float(r.superclip_mj), 2),
            "coast_mj": round(float(getattr(r, "coast_mj", 0.0)), 2),
            "drop_kmh": round(drop_from_profile(profile), 1), "proven": bool(r.converged) and float(r.optimality_gap_s) <= 0.01}


def real_laps(root: Path) -> dict:
    race = pd.read_csv(root / "results/multi_rebased_mid/R/race_summary.csv")
    quali = pd.read_csv(root / "results/quali_real/Q/race_summary.csv")
    out = {}
    for kind, t in (("race", race[race.used.astype(bool)]), ("quali", quali[quali.clean_laps >= MIN_QUALI_LAPS])):
        for r in t.itertuples():
            out.setdefault(NAMES.get(r.event, r.event), {})[kind] = {
                "drop_kmh": float(r.drop_kmh_median), "clip_share": round(float(r.clip_share), 2), "laps": int(r.clean_laps),
                "cap_kw": 350 if pd.Timestamp(r.date) >= FIRST_350 else 250}
    return out


def build(root: Path = Path(".")) -> dict:
    rules = [*fia_rules(root), *rules_2027(root)]
    events = list(rules[0]["rows"])
    if any(set(r["rows"]) != set(events) for r in rules):
        raise ValueError("a rule is missing a circuit")
    return {"schema": SCHEMA_VERSION, "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "circuits": [{"event": e, "label": e.replace(" Grand Prix", "")} for e in events],
            "rules": rules, "real": real_laps(root)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data/rules.json")
    args = ap.parse_args()
    data = build()
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {args.out}: {len(data['rules'])} rules, {len(data['circuits'])} circuits, {Path(args.out).stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
