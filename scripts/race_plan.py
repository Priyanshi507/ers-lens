"""Whole-race energy plans (research log, 2026-11-02), in two steps.

    python scripts/race_plan.py tables --out results/race      # GPU: lap tables (ERSLENS_DP_BACKEND=cuda)
    python scripts/race_plan.py plans --out results/race       # anywhere: race plans and W1-W3

Lap tables use the low car and the event's 2026 race rules; the race distance comes from Jolpica's
results (laps completed by the winner), cached like every other Jolpica request.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

EVENTS = {"Australia": "2026-03-08", "China": "2026-03-15", "Miami": "2026-05-03", "Canada": "2026-05-24"}
CAR = "configs/car_2026_q_low.yaml"
LEVELS_J = np.linspace(0.0, 4e6, 9)
START = 4          # 2 MJ, half the battery window
CHECK_S = 0.05


def race_laps(year: int, date: str) -> int:
    from erslens.openf1 import jolpica

    races = jolpica(f"{year}.json")["MRData"]["RaceTable"]["Races"]
    rnd = next(r["round"] for r in races if r["date"] == date)
    results = jolpica(f"{year}/{rnd}/results.json?limit=100")["MRData"]["RaceTable"]["Races"][0]["Results"]
    return max(int(r["laps"]) for r in results)


def tables(events, out: Path):
    from erslens.ingest import load_track
    from erslens.optimal import solve_refined
    from erslens.params import load_car_params
    from erslens.race import lap_table
    from erslens.strategy import race_rules, solution_ok

    (out / "tables").mkdir(parents=True, exist_ok=True)
    for e in events:
        path = out / "tables" / f"{e}.npz"
        if path.exists():
            print(f"{e}: kept {path}")
            continue
        car = race_rules(load_car_params(CAR), e)
        track = load_track(f"results/tracks/{e}.npz")
        # One single-lap solve per level: the check covers every level (research log, 2026-11-04).
        singles = [solve_refined(track, car, soc0_frac=lv / car.es_capacity_j) for lv in LEVELS_J]
        t = lap_table(track, car, LEVELS_J)
        np.savez(path, levels_j=t.levels_j, lap_s=t.lap_s, harvest_j=t.harvest_j, end_j=t.end_j, price=t.price,
                 budget_j=t.budget_j, unconverged=t.unconverged,
                 single_lap_s=np.array([s.lap_time_sim_s for s in singles]),
                 single_ok=np.array([solution_ok(s) for s in singles]))
        gap = np.abs(np.diag(t.lap_s) - [s.lap_time_sim_s for s in singles])
        print(f"{e}: same-charge laps against single-lap solves, largest gap {gap.max():.3f} s (level "
              f"{t.levels_j[int(np.argmax(gap))] / 1e6:g} MJ); single solves converged {sum(map(solution_ok, singles))} "
              f"of {len(singles)}; {int(np.isfinite(t.lap_s).sum())} of {t.lap_s.size} pairs found, "
              f"{int(t.unconverged.sum())} candidates dropped", flush=True)


def plans(events, out: Path, year: int = 2026):
    from erslens.race import LapTable, race_plan

    rows = []
    for e, date in events.items():
        z = np.load(out / "tables" / f"{e}.npz")
        t = LapTable(z["levels_j"], z["lap_s"], z["harvest_j"], z["end_j"], z["price"], float(z["budget_j"]), z["unconverged"])
        n = race_laps(year, date)
        p = race_plan(t, n, START)
        simple = (n - 1) * t.lap_s[START, START] + np.min(t.lap_s[START])
        single = np.atleast_1d(z["single_lap_s"])
        if single.size == 1:   # tables of 2026-11-02: one single-lap solve, at the start level
            gap, ok = abs(t.lap_s[START, START] - single[0]), True
        else:
            gap, ok = float(np.max(np.abs(np.diag(t.lap_s) - single))), bool(np.all(z["single_ok"]))
        rows.append({"event": e, "laps": n, "neutral_lap_s": float(t.lap_s[START, START]),
                     "single_lap_s": float(single[START] if single.size > 1 else single[0]),
                     "largest_level_gap_s": float(gap), "check_ok": bool(ok and gap <= CHECK_S),
                     "race_s": float(p.race_s), "neutral_race_s": float(p.neutral_s), "gain_s": float(p.gain_s),
                     "simple_s": float(simple), "best_minus_simple_s": float(p.race_s - simple),
                     "finish_level_mj": float(t.levels_j[p.levels[-1]] / 1e6),
                     "plan_mj": " ".join(f"{t.levels_j[k] / 1e6:g}" for k in p.levels)})
    with open(out / "plans.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    v = verdicts(rows)
    (out / "verdicts.json").write_text(json.dumps(v, indent=1))
    for r in rows:
        print(f"{r['event']}: {r['laps']} laps, gain {r['gain_s']:.3f} s, best minus simple {r['best_minus_simple_s']:+.3f} s, "
              f"finishes at {r['finish_level_mj']:g} MJ, check {'ok' if r['check_ok'] else 'FAILED'}")
    for k, (n, total, held) in v.items():
        print(f"{k}: {n} of {total} -> {'held' if held else 'NOT held'}")


def verdicts(rows) -> dict:
    if not all(r["check_ok"] for r in rows):
        return {"check": [sum(r["check_ok"] for r in rows), len(rows), False]}
    w1 = [abs(r["best_minus_simple_s"]) <= 0.1 for r in rows]
    w2 = [0.1 <= r["gain_s"] <= 1.0 for r in rows]
    w3 = [r["finish_level_mj"] == 0.0 for r in rows]
    return {k: [sum(v), len(v), all(v)] for k, v in (("W1", w1), ("W2", w2), ("W3", w3))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tables", "plans"])
    ap.add_argument("--events", nargs="+", default=list(EVENTS))
    ap.add_argument("--out", default="results/race")
    args = ap.parse_args()
    out = Path(args.out)
    if args.step == "tables":
        tables(args.events, out)
    else:
        plans({e: EVENTS[e] for e in args.events}, out)


if __name__ == "__main__":
    main()
