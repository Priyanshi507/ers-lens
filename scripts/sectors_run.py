"""The FIA's per-race 250 kW sectors against no sector limit and against 250 kW everywhere.

From Miami, in race sectors listed in each event's FIA Power Unit Information document, MGU-K
deployment is limited to 250 kW below 310 km/h (Section C Iss 21, C5.2.8 iii). The Strategy Lab
so far bounded this only by 250 kW on the whole lap. Each circuit with recorded sectors
(configs/deploy_sectors_2026.yaml) is solved three ways with the original car and the event's
race rules: "none", "fia" (the listed sectors) and "everywhere". Predictions P1-P3 are in the
research log (2026-11-03), written before any run.

    python scripts/sectors_run.py --out results/sectors              # solve (GPU)
    python scripts/sectors_run.py --out results/sectors --verdicts   # P1-P3 from runs.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CAR = "configs/car_2026.yaml"
VARIANTS = ("none", "fia", "everywhere")
TOL_S = 0.02
FIELDS = ["event", "variant", "lap_s", "dp_pred_s", "converged", "clip_ratio", "superclip_mj", "deploy_mj",
          "deploy_in_sectors_mj", "deploy_outside_mj", "peak_deploy_in_sectors_kw", "sector_share", "seconds"]


def solve_all(events, out_dir: Path):
    from erslens.ingest import load_track
    from erslens.optimal import race_car, solve_refined, start_at_slowest_point
    from erslens.params import load_car_params
    from erslens.sectors import SECTOR_CAP_W, with_fia_sectors
    from erslens.strategy import clip_metrics, race_rules, solution_ok

    (out_dir / "profiles").mkdir(parents=True, exist_ok=True)
    path = out_dir / "runs.csv"
    done = {(r["event"], r["variant"]) for r in csv.DictReader(open(path))} if path.exists() else set()
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for e in events:
            car = race_rules(load_car_params(CAR), e)
            plain = load_track(f"results/tracks/{e}.npz")
            fia = with_fia_sectors(plain, e)
            mask = np.isfinite(start_at_slowest_point(fia, race_car(car)).deploy_cap_w)
            for variant in VARIANTS:
                if (e, variant) in done:
                    continue
                track, c = {"none": (plain, car), "fia": (fia, car),
                            "everywhere": (plain, car.with_(deploy_max_w=SECTOR_CAP_W))}[variant]
                t = time.perf_counter()
                sol = solve_refined(track, c)
                o = {k: np.asarray(v) for k, v in sol.out.items()}
                d = o["deploy"] * o["dt"] / 1e6
                r = {"event": e, "variant": variant, "lap_s": sol.lap_time_sim_s, "dp_pred_s": sol.lap_time_dp_s,
                     "converged": solution_ok(sol),
                     "clip_ratio": clip_metrics(sol.track, o)["clip_ratio"],
                     "superclip_mj": float((o["superclip"] * o["dt"]).sum() / 1e6), "deploy_mj": float(d.sum()),
                     "deploy_in_sectors_mj": float(d[mask].sum()), "deploy_outside_mj": float(d[~mask].sum()),
                     "peak_deploy_in_sectors_kw": float(o["deploy"][mask].max() / 1e3) if mask.any() else 0.0,
                     "sector_share": float(mask.mean()), "seconds": round(time.perf_counter() - t, 1)}
                w.writerow(r)
                f.flush()
                np.savez_compressed(out_dir / "profiles" / f"{e}_{variant}.npz", sector=mask,
                                    distance_m=sol.track.distance_m, v_limit=sol.track.v_limit_ms, cmd_w=sol.cmd_w,
                                    **{k: o[k] for k in ("v", "deploy", "harvest", "superclip", "dt")})
                print(f"{e} {variant}: lap {r['lap_s']:.3f} s (DP {r['dp_pred_s']:.3f}), deploy in sectors "
                      f"{r['deploy_in_sectors_mj']:.2f} MJ, peak there {r['peak_deploy_in_sectors_kw']:.0f} kW, "
                      f"{r['seconds']:.0f} s", flush=True)
    print(f"All results in {path}")


def verdicts(runs: pd.DataFrame) -> dict:
    p = runs.pivot_table(index="event", columns="variant",
                         values=["lap_s", "converged", "deploy_in_sectors_mj", "deploy_outside_mj"], aggfunc="first")
    ok = p["converged"][list(VARIANTS)].astype(bool).all(axis=1)
    t = p["lap_s"]
    n = len(p)
    cost_fia, cost_all = t["fia"] - t["none"], t["everywhere"] - t["none"]
    p1 = ok & (t["none"] <= t["fia"] + TOL_S) & (t["fia"] <= t["everywhere"] + TOL_S)
    p2 = ok & (cost_fia <= 0.5 * cost_all)
    inside, outside = p["deploy_in_sectors_mj"], p["deploy_outside_mj"]
    p3 = ok & (inside["fia"] < inside["none"]) & (outside["fia"] > outside["none"])
    table = pd.DataFrame({"converged": ok, "cost_fia_s": cost_fia, "cost_everywhere_s": cost_all,
                          "in_none_mj": inside["none"], "in_fia_mj": inside["fia"],
                          "out_none_mj": outside["none"], "out_fia_mj": outside["fia"]})
    return {"table": table,
            "P1": (int(p1.sum()), n, bool(p1.all())),
            "P2": (int(p2.sum()), n, bool(p2.sum() >= 0.8 * n)),
            "P3": (int(p3.sum()), n, bool(p3.sum() >= 0.8 * n))}


def main():
    from erslens.sectors import sector_events

    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=[n for n, e in sector_events().items() if e["sectors"]])
    ap.add_argument("--out", default="results/sectors")
    ap.add_argument("--verdicts", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.verdicts:
        solve_all(args.events, out)
        return
    v = verdicts(pd.read_csv(out / "runs.csv"))
    print(v["table"].round(3).to_string())
    for k, text in (("P1", "none <= fia <= everywhere (0.02 s)"), ("P2", "FIA sectors cost at most half of everywhere"),
                    ("P3", "energy moves out of the sectors")):
        a, b, held = v[k]
        print(f"{k} ({text}): {a} of {b} -> {'held' if held else 'NOT held'}")
    (out / "verdicts.json").write_text(json.dumps({k: v[k] for k in ("P1", "P2", "P3")}, indent=1))


if __name__ == "__main__":
    main()
