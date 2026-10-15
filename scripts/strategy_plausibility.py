"""Compare the optimum and the best simple rule with the real lap the track was built from.

The reference track is the pole lap. The model runs it at qualifying fuel but keeps the race
rule that the battery ends the lap no lower than it started, while a real qualifying lap can
spend stored charge. So the optimum should be no faster than real pole; if it is clearly
faster, the car model is too generous and every gap it reports (S5 included) needs that caveat.
Writes results/strategy/plausibility.md.

    python scripts/strategy_plausibility.py --events Australia China Miami Canada
"""
import argparse
from pathlib import Path

from erslens.optimal import solve_refined
from erslens.params import load_car_params
from erslens.simple import tune
from erslens.strategy import race_rules

QUALI_FUEL_KG = 2.0  # race_car halves this, so the lap runs at 1 kg


def pole_lap_s(year: int, event: str) -> float:
    from erslens.ingest import _session

    lap = _session(year, event, "Q", "data/cache").laps.pick_quicklaps().pick_fastest()
    return lap["LapTime"].total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=["Australia", "China", "Miami", "Canada"])
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    from erslens.ingest import reference_track

    base = load_car_params("configs/car_2026.yaml")
    lines = ["| Circuit | Real pole lap | Optimum | Best simple rule | Optimum minus real |",
             "|---|---|---|---|---|"]
    for event in args.events:
        track = reference_track(args.year, event)
        car = race_rules(base, event).with_(fuel_kg=QUALI_FUEL_KG)
        real = pole_lap_s(args.year, event)
        opt = solve_refined(track, car).lap_time_sim_s
        simple = min(t.lap_time_s for t in tune(track, car).values())
        lines.append(f"| {event} | {real:.3f} s | {opt:.3f} s | {simple:.3f} s | {opt - real:+.3f} s |")
        print(lines[-1], flush=True)

    text = "\n".join(["Qualifying fuel (1 kg), corrected 2026 race rules, battery charge-neutral over the lap.", ""] + lines)
    Path("results/strategy/plausibility.md").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
