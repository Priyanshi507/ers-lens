"""Compute everything the pre-registered predictions S1-S7 need. Slow; safe to interrupt.

Each result is appended to results/strategy/runs.csv as soon as it is finished, and finished
results are skipped on a re-run, so an interrupted run continues where it stopped. Stages run
most-important first: nominal cars, the no-taper counterfactual, the gradient method, the tuned
simple policies, the legal-car sweep, and a starting-charge sensitivity check.

    python scripts/strategy_run.py --events Australia China Miami Canada
    python scripts/strategy_run.py --demo --quick      # smoke test, about a minute
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from erslens.gradopt import solve_gradient
from erslens.optimal import race_car, solve_refined, start_at_slowest_point
from erslens.params import load_car_params
from erslens.simple import tune
from erslens.strategy import clip_metrics, legal_sweep, no_taper, race_rules, rules_sensitivity
from erslens.track import demo_circuit

FIELDS = ["event", "variant", "car_id", "lap_s", "dp_pred_s", "harvest_mj", "soc_start_mj",
          "soc_end_mj", "clip_ratio", "peak_from_start_m", "run_length_m", "deploy_weighted_kmh",
          "full_throttle_mean_kmh", "corr_with_nominal", "seconds"]


def load_track(event: str, year: int):
    if event == "demo":
        return demo_circuit()
    from erslens.ingest import reference_track

    return reference_track(year, event)


def row(event, variant, car_id, sol, seconds, corr=np.nan):
    m = clip_metrics(sol.track, sol.out)
    return {"event": event, "variant": variant, "car_id": car_id, "lap_s": sol.lap_time_sim_s,
            "dp_pred_s": sol.lap_time_dp_s, "harvest_mj": sol.harvested_j / 1e6,
            "soc_start_mj": sol.soc0_j / 1e6, "soc_end_mj": sol.soc_end_j / 1e6, **m,
            "corr_with_nominal": corr, "seconds": seconds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=["Australia", "China", "Miami", "Canada"])
    ap.add_argument("--demo", action="store_true", help="use the synthetic demo circuit")
    ap.add_argument("--quick", action="store_true", help="coarse grids and 3 sweep cars: smoke test only")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--out", default="results/strategy")
    ap.add_argument("--rules", action="store_true",
                    help="only the corrected per-race 2026 rules stage (configs/rules_2026_races.yaml)")
    args = ap.parse_args()

    events = ["demo"] if args.demo else args.events
    fine = dict(dv=1.0, n_soc=81) if args.quick else dict(dv=0.125, n_soc=321)
    medium = dict(dv=1.0, n_soc=81) if args.quick else dict(dv=0.25, n_soc=161)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ("runs_quick.csv" if args.quick else "runs.csv")
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["event"], r["variant"], r["car_id"]) for r in csv.DictReader(f)}
    write_header = not path.exists()
    f = path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()

    car = load_car_params("configs/car_2026.yaml")
    tracks = {e: load_track(e, args.year) for e in events}
    sweep = legal_sweep(car)[:3] if args.quick else legal_sweep(car)
    nominal_cmd = {}

    def record(event, variant, car_id, compute):
        key = (event, variant, car_id)
        if key in done:
            return
        t = time.time()
        result = compute()
        result["seconds"] = round(time.time() - t, 1)
        writer.writerow(result)
        f.flush()
        done.add(key)
        print(f"{event:10s} {variant:14s} {car_id:9s} lap {float(result['lap_s']):.3f} s "
              f"clip {float(result['clip_ratio']):.2f} ({result['seconds']} s)", flush=True)

    def nominal(event):
        sol = solve_refined(tracks[event], car, **fine)
        np.savez_compressed(out_dir / f"profile_{event}.npz", cmd_w=sol.cmd_w, v=sol.out["v"],
                            deploy=sol.out["deploy"], harvest=sol.out["harvest"],
                            distance_m=sol.track.distance_m, v_limit=sol.track.v_limit_ms)
        return row(event, "nominal", "nominal", sol, 0)

    def nominal_profile(event):
        if event not in nominal_cmd:
            nominal_cmd[event] = np.load(out_dir / f"profile_{event}.npz")["cmd_w"]
        return nominal_cmd[event]

    def gradient(event):
        seed = min(tune(tracks[event], car).values(), key=lambda t: t.lap_time_s)
        sol = solve_gradient(tracks[event], car, init_cmd_w=seed.cmd_w)
        return row(event, "m2", "nominal", sol, 0, np.corrcoef(sol.cmd_w, nominal_profile(event))[0, 1])

    def simple(event):
        rows = []
        for family, t in tune(tracks[event], car).items():
            rows.append({"event": event, "variant": f"simple_{family}", "car_id": "nominal",
                         "lap_s": t.lap_time_s, **{k: np.nan for k in FIELDS[4:-1]}})
        return rows

    if args.rules:
        sens = rules_sensitivity()
        for e in events:
            rc = race_rules(car, e)

            def corrected():
                sol = solve_refined(tracks[e], rc, **fine)
                np.savez_compressed(out_dir / f"profile_rules_{e}.npz", cmd_w=sol.cmd_w, v=sol.out["v"],
                                    deploy=sol.out["deploy"], harvest=sol.out["harvest"],
                                    distance_m=sol.track.distance_m, v_limit=sol.track.v_limit_ms)
                return row(e, "rules", "nominal", sol, 0)

            record(e, "rules", "nominal", corrected)
            record(e, "rules_no_taper", "nominal",
                   lambda: row(e, "rules_no_taper", "nominal", solve_refined(tracks[e], no_taper(rc), **fine), 0))
            low = rc.with_(harvest_per_lap_j=float(sens["harvest_low_j"]))
            record(e, "rules_h8", "nominal", lambda: row(e, "rules_h8", "nominal", solve_refined(tracks[e], low, **fine), 0))
            if rc.superclip_max_w >= 350e3:
                d250 = rc.with_(deploy_max_w=float(sens["deploy_outside_zones_w"]))
                record(e, "rules_deploy250", "nominal",
                       lambda: row(e, "rules_deploy250", "nominal", solve_refined(tracks[e], d250, **fine), 0))
        f.close()
        print(f"All results in {path}")
        return

    for e in events:
        record(e, "nominal", "nominal", lambda: nominal(e))
    for e in events:
        record(e, "no_taper", "nominal", lambda: row(e, "no_taper", "nominal",
                                                     solve_refined(tracks[e], no_taper(car), **fine), 0))
    for e in events:
        record(e, "m2", "nominal", lambda: gradient(e))
    for e in events:
        for r in simple(e):
            key = (r["event"], r["variant"], r["car_id"])
            if key not in done:
                r["seconds"] = 0
                writer.writerow(r)
                f.flush()
                done.add(key)
    for e in events:
        for car_id, c in sweep:
            record(e, "sweep", car_id, lambda: row(e, "sweep", car_id, solve_refined(tracks[e], c, **medium), 0))
    for e in events:
        for frac in (0.3, 0.7):
            record(e, f"soc0_{frac}", "nominal",
                   lambda: row(e, f"soc0_{frac}", "nominal", solve_refined(tracks[e], car, soc0_frac=frac, **medium), 0))
    f.close()
    print(f"All results in {path}")


if __name__ == "__main__":
    sys.exit(main())
