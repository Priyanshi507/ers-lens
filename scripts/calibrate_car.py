"""Fit straight-mode drag and ICE wheel power to real pole laps, checked on a held-out circuit.

The track caps speed at the real trace wherever the driver is off full throttle, so the fit
target is the time on each full-throttle run of the pole lap, solved under the event's
qualifying recharge limit. The car's energy use on that lap is not observable; the fit assumes
the pole lap runs close to the energy-optimal strategy, which is stated as a limitation
wherever the calibrated car is used.

Each grid point solves every circuit on a coarse DP grid; results are appended to
results/calibration/grid.csv, so the script can be stopped and resumed. Then each circuit in
turn is held out: the parameters with the lowest error on the other circuits are applied to it.

    python scripts/calibrate_car.py --time-one        # how long one solve takes
    python scripts/calibrate_car.py                   # full grid, then results/calibration/calibration.md
    python scripts/calibrate_car.py --cda 0.95 1.00 1.05 --ice 280 290 300   # extra grid points (kW)
    python scripts/calibrate_car.py --report          # rewrite calibration.md from grid.csv only
    python scripts/calibrate_car.py --out results/calibration_q   # separate folder for a new run
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from erslens.ingest import _car_data, _session
from erslens.optimal import grid_with, race_car, solve
from erslens.params import load_car_params
from erslens.physics import braking_envelope
from erslens.strategy import quali_rules
from erslens.track import track_from_reference

EVENTS = ["Australia", "China", "Miami", "Canada"]
CDA = [0.95, 1.05, 1.15, 1.25, 1.35, 1.45]
ICE_W = [320e3, 340e3, 360e3, 380e3, 400e3]
QUALI_FUEL_KG = 2.0
FULL_THROTTLE = 98.0
MIN_RUN_M = 100.0
DS = 5.0
OUT = Path("results/calibration")


def flat_runs(mask, min_len):
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_len:
                yield i, j
            i = j
        else:
            i += 1


def load(year):
    data = {}
    for event in EVENTS:
        lap = _session(year, event, "Q", "data/cache").laps.pick_quicklaps().pick_fastest()
        df = _car_data(lap)
        track = track_from_reference(event, df["distance_m"].to_numpy(), df["speed_kmh"].to_numpy(),
                                     df["throttle"].to_numpy(), ds=DS)
        v_real = np.interp(track.distance_m, df["distance_m"], df["speed_kmh"]) / 3.6
        flat = np.interp(track.distance_m, df["distance_m"], df["throttle"]) >= FULL_THROTTLE
        dt_real = DS / np.maximum(v_real, 1.0)
        data[event] = (track, list(flat_runs(flat, int(MIN_RUN_M / DS))), dt_real)
    return data


def run_errors(event, track, spans, dt_real, base, cda, ice_w):
    """Model minus real time on each full-throttle run, and on the whole lap."""
    car = quali_rules(base, event).with_(fuel_kg=QUALI_FUEL_KG, cda_straight=cda, ice_power_w=ice_w)
    rc = race_car(car)
    sol = solve(track, car, grid=grid_with(rc, 0.5, 81), soc0_fracs=(0.5,))
    k = int(np.argmin(braking_envelope(track, rc)))
    dt_sim = np.roll(np.asarray(sol.out["dt"]), k)
    per_run = [float(dt_sim[i:j].sum() - dt_real[i:j].sum()) for i, j in spans]
    return per_run, float(dt_sim.sum() - dt_real.sum())


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--time-one", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="folder for grid.csv and calibration.md")
    ap.add_argument("--cda", nargs="+", type=float, default=CDA)
    ap.add_argument("--ice", nargs="+", type=float, default=[w / 1e3 for w in ICE_W], help="kW")
    args = ap.parse_args()
    OUT = Path(args.out)
    if args.report:
        report(OUT / "grid.csv")
        return
    base = load_car_params("configs/car_2026.yaml")
    data = load(args.year)

    if args.time_one:
        track, spans, dt_real = data["Canada"]
        t0 = time.time()
        _, lap_err = run_errors("Canada", track, spans, dt_real, base, base.cda_straight, base.ice_power_w)
        print(f"one coarse solve: {time.time() - t0:.0f} s; Canada, current car: model minus real {lap_err:+.3f} s")
        print(f"full grid: {len(CDA) * len(ICE_W) * len(EVENTS)} solves")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "grid.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {(r["event"], round(float(r["cda"]), 4), round(float(r["ice_w"]))) for r in csv.DictReader(f)}
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["event", "cda", "ice_w", "run_sq_err", "n_runs", "lap_err_s"])
        for cda in args.cda:
            for ice_w in (kw * 1e3 for kw in args.ice):
                for event in EVENTS:
                    if (event, round(cda, 4), round(ice_w)) in done:
                        continue
                    track, spans, dt_real = data[event]
                    per_run, lap_err = run_errors(event, track, spans, dt_real, base, cda, ice_w)
                    w.writerow([event, cda, ice_w, sum(e * e for e in per_run), len(per_run), lap_err])
                    f.flush()
                    print(f"{event:10s} CdA {cda:.2f} ICE {ice_w / 1e3:.0f} kW: model minus real {lap_err:+.3f} s",
                          flush=True)
    report(path)


def report(path):
    import pandas as pd

    g = pd.read_csv(path)
    g["cda"], g["ice_w"] = g.cda.round(4), g.ice_w.round()
    # Only grid points solved at every circuit can be compared.
    complete = g.groupby(["cda", "ice_w"]).event.nunique() == len(EVENTS)
    g = g.set_index(["cda", "ice_w"]).loc[complete[complete].index].reset_index()
    cdas, ices = sorted(g.cda.unique()), sorted(g.ice_w.unique())
    lines = ["# Car calibration against 2026 pole laps", "",
             "Error = model minus real time, qualifying fuel, coarse DP grid (0.5 m/s, 81 charge levels). "
             "Fit target: sum of squared errors over full-throttle runs of at least 100 m.", "",
             "## Leave one circuit out", "",
             "| Held-out circuit | CdA fitted on the other three | ICE wheel power | Held-out lap error | "
             "Same circuit, current car (CdA 0.95, 400 kW) |", "|---|---|---|---|---|"]
    for held in EVENTS:
        fit = g[g.event != held].groupby(["cda", "ice_w"]).run_sq_err.sum().idxmin()
        row = g[(g.event == held) & (g.cda == fit[0]) & (g.ice_w == fit[1])].iloc[0]
        cur = g[(g.event == held) & (g.cda == 0.95) & (g.ice_w == 400e3)].iloc[0]
        lines.append(f"| {held} | {fit[0]:.2f} | {fit[1] / 1e3:.0f} kW | {row.lap_err_s:+.3f} s | "
                     f"{cur.lap_err_s:+.3f} s |")
    best = g.groupby(["cda", "ice_w"]).run_sq_err.sum()
    cda, ice_w = best.idxmin()
    edge = cda in (cdas[0], cdas[-1]) or ice_w in (ices[0], ices[-1])
    lines += ["", f"All four circuits: CdA {cda:.2f}, ICE wheel power {ice_w / 1e3:.0f} kW."
              + (" This is on the edge of the grid, so the grid must be widened before it is used." if edge else ""), "",
              "## Total squared run error over all circuits (s^2)", "",
              "| CdA | " + " | ".join(f"{w / 1e3:.0f} kW" for w in ices) + " |",
              "|---|" + "---|" * len(ices)]
    for c in cdas:
        lines.append(f"| {c:.2f} | " + " | ".join(
            f"{best[(c, w)]:.2f}" if (c, w) in best.index else "" for w in ices) + " |")
    lines += ["", "## Best ICE wheel power for each CdA", "", "| CdA | ICE wheel power | Squared run error |",
              "|---|---|---|"]
    for c in cdas:
        b = best.loc[c]
        lines.append(f"| {c:.2f} | {b.idxmin() / 1e3:.0f} kW | {b.min():.2f} |")
    text = "\n".join(lines)
    (OUT / "calibration.md").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
