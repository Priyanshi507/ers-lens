"""Does a finer harvest-price search change a solve? (research log, 2026-11-13)

    python scripts/price_resolution.py --event "Bahrain Grand Prix" --caps 250 350 --bisect 4 10
Original car, per-race rules, corrected (untapered) harvest; prints one row per (cap, bisection steps).
"""
import argparse
import time

from erslens.ingest import load_track
from erslens.optimal import solve_refined
from erslens.params import load_car_params
from erslens.strategy import race_rules, solution_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="Bahrain Grand Prix")
    ap.add_argument("--caps", nargs="+", type=float, default=[250.0, 350.0])
    ap.add_argument("--bisect", nargs="+", type=int, default=[4, 10])
    args = ap.parse_args()
    track = load_track(f"results/tracks/{args.event}.npz")
    base = race_rules(load_car_params("configs/car_2026.yaml"), args.event)
    print("event,cap_kw,fine_bisect,lap_s,dp_pred_s,converged,harvest_mj,budget_mj,price,optimality_gap_s,seconds")
    for cap in args.caps:
        car = base.with_(superclip_max_w=cap * 1e3)
        for n in args.bisect:
            t = time.perf_counter()
            s = solve_refined(track, car, fine_bisect=n)
            print(f"{args.event},{cap},{n},{s.lap_time_sim_s:.4f},{s.lap_time_dp_s:.4f},{solution_ok(s)},"
                  f"{s.harvested_j / 1e6:.4f},{car.harvest_per_lap_j / 1e6:.2f},{s.harvest_price_s_per_j:.4e},"
                  f"{s.optimality_gap_s:.4f},"
                  f"{time.perf_counter() - t:.0f}", flush=True)


if __name__ == "__main__":
    main()
