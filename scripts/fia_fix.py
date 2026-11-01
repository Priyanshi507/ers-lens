"""Did the FIA's Miami fix change super-clipping? The same circuits solved under both rules.

From the Miami Grand Prix the FIA raised the peak power a car may harvest while accelerating at
full throttle (super-clipping) from 250 to 350 kW. Real races cannot show the effect on their
own: each circuit ran under one rule only. Here the Strategy Lab solves every circuit with each
cap, everything else equal (that event's race recharge limit, same car), so any change is the
rule's. Predictions F1-F4 are in the research log (2026-11-01), written before any run.

Each result is appended to <out>/runs.csv as soon as it is finished; finished ones are skipped,
so an interrupted run continues. Tracks come from results/tracks (scripts/export_tracks.py).

    python scripts/fia_fix.py                    # all cars, events and caps
    python scripts/fia_fix.py --demo --quick     # smoke test on the synthetic circuit
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

EVENTS = ("Australia", "China", "Miami", "Canada")
CARS = {"original": "configs/car_2026.yaml", "low": "configs/car_2026_q_low.yaml",
        "high": "configs/car_2026_q_high.yaml"}
BOTH_RULES_W = (250e3, 350e3)
SWEEP_W = (150e3, 200e3, 250e3, 300e3, 350e3, 450e3)
ACTIVE_W = 1e3
FIELDS = ["event", "car", "cap_kw", "lap_s", "dp_pred_s", "converged", "harvest_mj", "superclip_mj",
          "superclip_s", "superclip_m", "longest_superclip_s", "superclip_at_cap_share",
          "peak_superclip_kw", "clip_ratio", "seconds"]


def superclip_metrics(out: dict, ds: float, cap_w: float) -> dict:
    sc, dt = np.asarray(out["superclip"]), np.asarray(out["dt"])
    on = sc > ACTIVE_W
    longest = run = 0.0
    for active, t in zip(on, dt):
        run = run + t if active else 0.0
        longest = max(longest, run)
    t_on = float(dt[on].sum())
    return {"superclip_mj": float((sc * dt).sum() / 1e6), "superclip_s": t_on,
            "superclip_m": float(on.sum() * ds), "longest_superclip_s": float(longest),
            "superclip_at_cap_share": float(dt[on & (sc >= 0.99 * cap_w)].sum() / t_on) if t_on > 0 else 0.0,
            "peak_superclip_kw": float(sc.max() / 1e3)}


def plan(cars: dict, events, quick: bool):
    for e in events:
        for name in cars:
            for cap in (SWEEP_W if name == "original" and not quick else BOTH_RULES_W):
                yield e, name, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=list(EVENTS))
    ap.add_argument("--cars", nargs="+", default=list(CARS), choices=list(CARS))
    ap.add_argument("--demo", action="store_true", help="synthetic demo circuit with the original car")
    ap.add_argument("--quick", action="store_true", help="coarse grid and the two rules only: smoke test")
    ap.add_argument("--out", default="results/fia_fix")
    args = ap.parse_args()

    from erslens.ingest import load_track
    from erslens.optimal import solve_refined
    from erslens.params import load_car_params
    from erslens.strategy import CONVERGED_S, clip_metrics, race_rules
    from erslens.track import demo_circuit

    events = ["demo"] if args.demo else args.events
    cars = {"original": CARS["original"]} if args.demo else {c: CARS[c] for c in args.cars}
    grid = {"dv": 0.5, "n_soc": 41} if args.quick else {}
    out_dir = Path(args.out)
    (out_dir / "profiles").mkdir(parents=True, exist_ok=True)
    path = out_dir / "runs.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {(r["event"], r["car"], float(r["cap_kw"])) for r in csv.DictReader(f)}
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for e, name, cap in plan(cars, events, args.quick):
            if (e, name, cap / 1e3) in done:
                continue
            base = load_car_params(cars[name])
            track = demo_circuit() if e == "demo" else load_track(f"results/tracks/{e}.npz")
            rules = base if e == "demo" else race_rules(base, e)
            car = rules.with_(superclip_max_w=cap)
            t = time.perf_counter()
            sol = solve_refined(track, car, **grid)
            seconds = time.perf_counter() - t
            o = {k: np.asarray(v) for k, v in sol.out.items()}
            r = {"event": e, "car": name, "cap_kw": cap / 1e3, "lap_s": sol.lap_time_sim_s,
                 "dp_pred_s": sol.lap_time_dp_s,
                 "converged": abs(sol.lap_time_dp_s - sol.lap_time_sim_s) <= CONVERGED_S,
                 "harvest_mj": sol.harvested_j / 1e6, **superclip_metrics(o, sol.track.ds, cap),
                 "clip_ratio": clip_metrics(sol.track, o)["clip_ratio"], "seconds": round(seconds, 1)}
            writer.writerow(r)
            f.flush()
            np.savez_compressed(out_dir / "profiles" / f"{e}_{name}_{int(cap / 1e3)}.npz",
                                distance_m=sol.track.distance_m, v_limit=sol.track.v_limit_ms, cmd_w=sol.cmd_w,
                                **{k: o[k] for k in ("v", "deploy", "harvest", "superclip", "dt")})
            print(f"{e} {name} {cap / 1e3:.0f} kW: lap {r['lap_s']:.3f} s, superclip {r['superclip_s']:.2f} s "
                  f"(longest {r['longest_superclip_s']:.2f} s), {seconds:.0f} s", flush=True)
    print(f"All results in {path}")


if __name__ == "__main__":
    main()
