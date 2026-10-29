"""Export the Strategy Lab results to site/data/strategy.json for the Explorer's Strategy Lab page.

Reads only committed results: the optimal profiles and runs.csv of the three cars (original car
and both ends of the calibration ridge), their verdicts.md, and the real clipping laps already
published in site/data/races. Every number on the page is recomputed from the saved profiles and
checked against runs.csv, so the page cannot drift from the research log.

    python scripts/export_strategy.py
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from erslens.params import load_car_params
from erslens.physics import mguk_cap_kmh
from erslens.strategy import CONVERGED_S, race_rules, resolve_runs

SCHEMA_VERSION = 1
EVENTS = ("Australia", "China", "Miami", "Canada")
SITE_ID = {"Australia": "australian", "China": "chinese", "Miami": "miami", "Canada": "canadian"}
CARS = (
    {"id": "original", "label": "Original car", "detail": "CdA 0.95, 400 kW (estimates)",
     "results": "results/strategy_rules_q", "config": "configs/car_2026.yaml"},
    {"id": "low", "label": "Low-drag end", "detail": "CdA 1.00, 310 kW (fitted)",
     "results": "results/strategy_q_low", "config": "configs/car_2026_q_low.yaml"},
    {"id": "high", "label": "High-drag end", "detail": "CdA 1.35, 400 kW (fitted)",
     "results": "results/strategy_q_high", "config": "configs/car_2026_q_high.yaml"},
)
LIMITS = {"race": "profile_rules_", "car": "profile_"}
VARIANT = {"race": "rules", "car": "nominal"}
CHECKS = ("Convergence rule (<= 0.05 s)", "S1", "S2", "S3", "S4", "S5", "S6", "S7 (exploratory)",
          "S1 (corrected rules)", "S3 (corrected rules)", "S4 (corrected rules)")
CLIP_MAX = 0.5
AT_CAP, AT_ZERO = 0.98, 0.02
BRAKE_BIN_M = 10


def longest_run(free: np.ndarray) -> tuple[int, int]:
    best, a = (0, 0), None
    for i, f in enumerate(np.append(free, False)):
        if f and a is None:
            a = i
        elif not f and a is not None:
            if i - a > best[1] - best[0]:
                best = (a, i)
            a = None
    return best


def lap_profile(p, car, ds: float) -> dict:
    """Arrays for the lap chart, plus the numbers the page quotes, all from one saved optimum."""
    v, deploy, harvest = p["v"], p["deploy"], p["harvest"]
    dt = 2.0 * ds / (v + np.roll(v, -1))
    soc = car.es_capacity_j * 0.5 + np.cumsum(car.eta_harvest * harvest * dt - deploy / car.eta_deploy * dt)
    soc = np.concatenate([[car.es_capacity_j * 0.5], soc[:-1]])
    free = np.isinf(p["v_limit"])
    a, b = longest_run(free)
    net = (deploy - harvest)[a:b]
    peak = int(np.argmax(v[a:b]))
    pre = net[:peak].mean() if peak > 0 else np.nan
    cap = mguk_cap_kmh(v * 3.6, car)
    on_cap, at_zero = deploy[free] >= AT_CAP * cap[free], deploy[free] <= AT_ZERO * cap[free]
    return {
        "v": [round(float(x), 1) for x in v * 3.6],
        "deploy": [int(x) for x in np.round(deploy / 1e3)],
        "harvest": [int(x) for x in np.round(harvest / 1e3)],
        "cap": [int(x) for x in np.round(cap / 1e3)],
        "soc": [round(float(x), 3) for x in soc / 1e6],
        "straight": [a, b], "peak": a + peak,
        "lap_s": round(float(dt.sum()), 3),
        "clip_ratio": round(float(net[peak] / pre), 2) if pre > 0 else None,
        "deployed_mj": round(float((deploy / car.eta_deploy * dt).sum() / 1e6), 2),
        "harvested_mj": round(float((car.eta_harvest * harvest * dt).sum() / 1e6), 2),
        "soc_end_mj": round(float(soc[-1] + car.eta_harvest * harvest[-1] * dt[-1]
                                  - deploy[-1] / car.eta_deploy * dt[-1]) / 1e6, 3),
        "bang_bang": {"cap": round(float(on_cap.mean()), 3), "zero": round(float(at_zero.mean()), 3),
                      "between": round(float((~on_cap & ~at_zero).mean()), 3)},
        "peak_to_brake_m": int(round((b - (a + peak)) * ds)),
    }


def parse_verdicts(text: str) -> dict:
    """Result and detail of each check in a verdicts.md written by strategy_evaluate.py."""
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in CHECKS:
            out[cells[0]] = {"held": cells[1] == "held", "detail": cells[2]}
    missing = [c for c in CHECKS if c not in out]
    if missing:
        raise ValueError(f"verdicts.md has no row for {missing}")
    return out


def parse_s7(text: str) -> dict:
    out = {}
    for m in re.finditer(r"\| S7 (\w+) \| optimum (\d+) m, observed IQR (\d+)-(\d+) m", text):
        out[m.group(1)] = {"optimum_m": int(m.group(2)), "iqr_m": [int(m.group(3)), int(m.group(4))]}
    return out


def sweep(used: pd.DataFrame, excluded: pd.DataFrame, event: str) -> list[dict]:
    rows = [{"car": r.car_id, "clip_ratio": round(float(r.clip_ratio), 2), "converged": True}
            for r in used[(used.variant == "sweep") & (used.event == event)].itertuples()]
    rows += [{"car": r.car_id, "clip_ratio": None, "converged": False}
             for r in excluded[(excluded.variant == "sweep") & (excluded.event == event)].itertuples()]
    return sorted(rows, key=lambda r: r["car"])


def real_laps(event: str, races_dir: Path) -> dict:
    """Real clipping laps on the same straight, aligned at the braking point (end of full throttle).

    The corner fixes where full throttle ends for the reference lap and race laps alike, so this
    alignment needs no assumption about where each lap's run began (research log, 2026-10-10).
    """
    data = json.loads((races_dir / f"{SITE_ID[event]}.json").read_text())
    traces = [lap["trace"] for lap in data["laps"] if lap["clipping"] and "trace" in lap]
    grid = np.arange(0, 1500 + BRAKE_BIN_M, BRAKE_BIN_M)
    speeds, peaks = [], []
    for t in traces:
        d, v = np.asarray(t["d"], float), np.asarray(t["v"], float)
        a, b = t["flat"]
        to_brake = d[b] - d[a:b + 1]
        order = np.argsort(to_brake)
        row = np.interp(grid, to_brake[order], v[a:b + 1][order], left=np.nan, right=np.nan)
        speeds.append(row)
        peaks.append(d[b] - d[t["peak"]])
    speeds = np.asarray(speeds)
    keep = np.flatnonzero(np.sum(~np.isnan(speeds), axis=0) >= max(20, len(traces) // 4))
    q25, q50, q75 = (np.nanpercentile(speeds[:, keep], q, axis=0) for q in (25, 50, 75))
    return {
        "laps": len(traces),
        "to_brake_m": [int(x) for x in grid[keep]],
        "v_median": [round(float(x), 1) for x in q50],
        "v_q25": [round(float(x), 1) for x in q25],
        "v_q75": [round(float(x), 1) for x in q75],
        "peak_to_brake_m": sorted(int(round(x)) for x in peaks),
    }


def build(root: Path = Path(".")) -> dict:
    cars, circuits = [], {e: {"cars": {}} for e in EVENTS}
    for spec in CARS:
        folder = root / spec["results"]
        car = load_car_params(root / spec["config"])
        used, excluded = resolve_runs(pd.read_csv(folder / "runs.csv"))
        text = (folder / "verdicts.md").read_text()
        verdicts, s7 = parse_verdicts(text), parse_s7(text)
        cars.append({k: spec[k] for k in ("id", "label", "detail")} | {
            "cda": car.cda_straight, "ice_kw": round(car.ice_power_w / 1e3), "verdicts": verdicts,
            "converged": verdicts.get("Convergence rule (<= 0.05 s)", {}).get("detail")})
        for e in EVENTS:
            entry = {"profiles": {}, "sweep": sweep(used, excluded, e), "s7": s7.get(e)}
            for limits, prefix in LIMITS.items():
                c = race_rules(car, e) if limits == "race" else car
                p = np.load(folder / f"{prefix}{e}.npz")
                ds = float(p["distance_m"][1] - p["distance_m"][0])
                prof = lap_profile(p, c, ds)
                row = used[(used.event == e) & (used.variant == VARIANT[limits]) & (used.car_id == "nominal")]
                if len(row) != 1:
                    raise ValueError(f"{spec['id']} {e} {limits}: expected one converged result")
                r = row.iloc[0]
                if abs(prof["lap_s"] - r.lap_s) > 1e-3 or abs(prof["soc_end_mj"] - r.soc_end_mj) > 1e-3:
                    raise ValueError(f"{spec['id']} {e} {limits}: profile does not reproduce runs.csv")
                prof["converged_gap_s"] = round(abs(float(r.dp_pred_s) - float(r.lap_s)), 3)
                prof["harvest_limit_mj"] = round(c.harvest_per_lap_j / 1e6, 2)
                prof["ds_m"] = ds
                entry["profiles"][limits] = prof
            circuits[e]["cars"][spec["id"]] = entry
    for e in EVENTS:
        circuits[e]["real"] = real_laps(e, root / "site/data/races")
    return {"schema": SCHEMA_VERSION, "generated": datetime.now(timezone.utc).date().isoformat(),
            "thresholds": {"clip_max": CLIP_MAX, "at_cap": AT_CAP, "at_zero": AT_ZERO,
                           "converged_s": CONVERGED_S},
            "events": list(EVENTS), "cars": cars, "circuits": circuits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site/data/strategy.json")
    args = ap.parse_args()
    data = build()
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
