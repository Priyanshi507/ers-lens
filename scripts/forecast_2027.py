"""The 2026 model car under the ratified 2027 rules, every circuit, a family of unknown curves.

FIA-F1-DOC-111, which holds the 2027 speed-dependent deployment curves, is not public. So every
circuit is solved for each member of a family of curves (deployment held at 300 kW up to a taper
start speed, then the 2026 slope of 5 kW per km/h; or no taper) and for both plausible super-clip
limits (350 kW as in 2026, 375 kW the new absolute ERS-K power). Only what holds across the whole
family is used for a forecast. Rules and predictions: research log, 2026-11-13.

Each result is appended to <out>/runs.csv as soon as it is finished; finished ones are skipped.

    python scripts/forecast_2027.py --out results/forecast_2027
    python scripts/forecast_2027.py --forecast --out results/forecast_2027_a results/forecast_2027_b \
        --baseline results/fia_fix_calendar_c        # writes results/forecast_2027.csv
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fia_fix import superclip_metrics
from hindcast_2026 import drop_from_profile, model_drop_kmh

CAR = "configs/car_2027.yaml"
TAPER_START_KMH = {"t280": 280.0, "t300": 300.0, "t320": 320.0, "none": None}
SUPERCLIP_W = (350e3, 375e3)
EVENTS = ("Australia", "China", "Miami", "Canada", "Japan", "Austria", "Belgium", "Monaco Grand Prix",
          "Barcelona Grand Prix", "British Grand Prix", "Hungarian Grand Prix", "Dutch Grand Prix",
          "Italian Grand Prix", "Spanish Grand Prix", "Azerbaijan Grand Prix", "Bahrain Grand Prix",
          "Singapore Grand Prix", "United States Grand Prix")
FIELDS = ["event", "curve", "superclip_kw", "lap_s", "dp_pred_s", "converged", "optimality_gap_s",
          "harvest_mj", "superclip_mj", "model_drop_kmh", "clip_ratio", "seconds"]


def car_2027(event: str, curve: str, superclip_w: float):
    """2027 car with the event's 2026 race recharge limit (C5.2.10 is unchanged in wording)."""
    from erslens.params import load_car_params
    from erslens.strategy import race_rules

    car = race_rules(load_car_params(CAR), event).with_(superclip_max_w=superclip_w)
    start = TAPER_START_KMH[curve]
    # 300 kW meets a1 - 5v at v = start; with no taper the deployment cap alone applies until the
    # 2026 high-speed tail (340-345 km/h), which C5.2.8 no longer states but top speed never reaches.
    a1 = 1e12 if start is None else (300.0 + 5.0 * start) * 1e3
    return car.with_(taper_a1_w=a1)


def solve_all(out: Path, events, grid: dict | None = None):
    from erslens.ingest import load_track
    from erslens.optimal import solve_refined
    from erslens.strategy import clip_metrics, solution_ok

    (out / "profiles").mkdir(parents=True, exist_ok=True)
    path = out / "runs.csv"
    done = set()
    if path.exists() and path.stat().st_size:
        done = {(r["event"], r["curve"], float(r["superclip_kw"])) for r in csv.DictReader(open(path))}
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for e in events:
            track = load_track(f"results/tracks/{e}.npz")
            for curve in TAPER_START_KMH:
                for sc in SUPERCLIP_W:
                    if (e, curve, sc / 1e3) in done:
                        continue
                    t = time.perf_counter()
                    sol = solve_refined(track, car_2027(e, curve, sc), **(grid or {}))
                    o = {k: np.asarray(v) for k, v in sol.out.items()}
                    name = f"{e}_{curve}_{int(sc / 1e3)}.npz"
                    np.savez_compressed(out / "profiles" / name, distance_m=sol.track.distance_m,
                                        v_limit=sol.track.v_limit_ms, cmd_w=sol.cmd_w,
                                        **{k: o[k] for k in ("v", "deploy", "harvest", "superclip", "dt")})
                    sm = superclip_metrics(o, sol.track.ds, sc)
                    r = {"event": e, "curve": curve, "superclip_kw": sc / 1e3, "lap_s": sol.lap_time_sim_s,
                         "dp_pred_s": sol.lap_time_dp_s, "converged": solution_ok(sol),
                         "optimality_gap_s": sol.gap_or_nan, "harvest_mj": sol.harvested_j / 1e6,
                         "superclip_mj": sm["superclip_mj"],
                         "model_drop_kmh": drop_from_profile(out / "profiles" / name),
                         "clip_ratio": clip_metrics(sol.track, o)["clip_ratio"],
                         "seconds": round(time.perf_counter() - t, 1)}
                    w.writerow(r)
                    f.flush()
                    print(f"{e} {curve} {sc / 1e3:.0f} kW: lap {r['lap_s']:.3f} s, drop {r['model_drop_kmh']:.1f} km/h, "
                          f"gap {r['optimality_gap_s']:.3f} s, converged {r['converged']}", flush=True)
    print(f"All results in {path}")


def baseline_drops(baseline: Path) -> pd.Series:
    """2026 drop under the super-clip cap in force at each circuit's 2026 race."""
    from erslens.params import load_car_params
    from erslens.strategy import race_rules

    car = load_car_params("configs/car_2026.yaml")
    out = {}
    for e in EVENTS:
        cap = int(race_rules(car, e).superclip_max_w / 1e3)
        out[e] = model_drop_kmh(baseline / "profiles", e, cap)
    return pd.Series(out, name="drop_2026_kmh")


def remeasured(out: Path) -> pd.DataFrame:
    """runs.csv with each drop recomputed from its saved profile by the current drop_from_profile
    (the runs' own column predates the braking fix of 2026-11-14)."""
    runs = pd.read_csv(out / "runs.csv")
    paths = [out / "profiles" / f"{r.event}_{r.curve}_{int(r.superclip_kw)}.npz" for r in runs.itertuples()]
    if all(p.exists() for p in paths):
        runs["model_drop_kmh"] = [drop_from_profile(p) for p in paths]
    return runs


def forecast(outs: list[Path], baseline: Path) -> pd.DataFrame:
    """The forecast rule written before the runs (research log, 2026-11-13, rule R)."""
    runs = pd.concat([remeasured(o) for o in outs], ignore_index=True)
    ok = runs.groupby("event").apply(lambda g: bool(g.converged.astype(bool).all()), include_groups=False)
    d = runs.pivot_table(index="event", columns=["curve", "superclip_kw"], values="model_drop_kmh")
    base = baseline_drops(baseline).reindex(d.index)
    change = d.sub(base, axis=0)
    t = pd.DataFrame({"drop_2026_kmh": base, "drop_2027_median_kmh": d.median(axis=1),
                      "change_min_kmh": change.min(axis=1), "change_max_kmh": change.max(axis=1),
                      "all_converged": ok.reindex(d.index)})
    t["direction"] = np.where(~t.all_converged, "no forecast",
                              np.where(t.change_max_kmh < 0, "less",
                                       np.where(t.change_min_kmh > 0, "more", "no forecast")))
    # No ranking: Y1 failed on the corrected measure, and the rule written with it then allows directions only.
    return t.sort_values("change_max_kmh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", nargs="+", default=["results/forecast_2027"],
                    help="one folder to solve into; with --forecast, every folder to read")
    ap.add_argument("--events", nargs="+", default=list(EVENTS))
    ap.add_argument("--forecast", action="store_true", help="apply the forecast rule to finished runs")
    ap.add_argument("--baseline", default="results/fia_fix_calendar_c")
    ap.add_argument("--quick", action="store_true", help="coarse grid, for a smoke test only")
    args = ap.parse_args()
    outs = [Path(o) for o in args.out]
    if args.forecast:
        t = forecast(outs, Path(args.baseline))
        t.to_csv(outs[0].parent / "forecast_2027.csv")
        print(t.round(2).to_string())
        return
    solve_all(outs[0], args.events, {"dv": 1.0, "n_soc": 41} if args.quick else None)


if __name__ == "__main__":
    main()
