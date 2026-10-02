"""Fit the 3-parameter energy strategy to every clean race lap on each circuit's longest straight.

Both clipping models are fitted to every lap: a step (power switches at the clip point)
and a ramp (power falls no faster than the reported 50 kW/s limit). Rule stated before
running: the ramp model is preferred if its median fit error is lower on at least 5 of
the 7 original circuits.

Validation: the fitted swing (D x available power + H) is compared with the independent
matched-speed estimator from Step 1. Prediction stated before running: from Miami the
FIA raised peak super-clipping power from 250 to 350 kW, so if that applies in races the
fitted harvest power H should rise after Miami.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from derate_analysis import lap_quality, longest_flat_out_run
from erslens.energy import electric_power_swing
from erslens.params import load_car_params
from erslens.straightfit import DS, fit_straight
from multi_race import permutation_p

SENSITIVITY_LAPS = 40


def lap_on_grid(lap: pd.DataFrame, grid: np.ndarray):
    d = lap["distance_m"].to_numpy()
    v = np.interp(grid, d, lap["speed_kmh"].to_numpy()) / 3.6
    nearest = np.clip(np.searchsorted(d, grid), 0, len(d) - 1)
    flat = (lap["throttle"].to_numpy()[nearest] >= 98) & ~lap["brake"].astype(bool).to_numpy()[nearest]
    return v, flat


def analyse_event(year: int, event: str, car) -> pd.DataFrame:
    from erslens.ingest import driver_teams, event_info, load_driver_race, race_drivers, reference_track

    name, date = event_info(year, event)
    track = reference_track(year, event)
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
    grid = np.arange(start_m, end_m, DS)
    elev = (np.interp(grid, track.distance_m, track.elevation_m)
            if track.elevation_m is not None else np.zeros_like(grid))
    grade = np.gradient(elev, DS)
    elevation = None if track.elevation_m is None else (track.distance_m, track.elevation_m)
    teams = driver_teams(year, event)
    rows = []
    for drv in race_drivers(year, event):
        race = load_driver_race(year, event, drv)
        if race.empty:
            continue
        durations = race.groupby("lap")["time_s"].max()
        for lap_no, lap in race.groupby("lap"):
            if lap_no == 1 or durations[lap_no] > 1.07 * durations.median():
                continue
            if lap["pit_in"].iloc[0] or lap["pit_out"].iloc[0]:
                continue
            lap = lap.reset_index(drop=True)
            if lap_quality(lap, start_m, end_m)[1] != "ok":
                continue
            v, mask = lap_on_grid(lap, grid)
            mass = car.mass_kg + max(0.0, car.fuel_kg - car.fuel_per_lap_kg * lap_no)
            fs = fit_straight(v, mask, grade, mass, car, ramp=False)
            fr = fit_straight(v, mask, grade, mass, car, ramp=True)
            if fs is None or fr is None:
                continue
            ind = electric_power_swing(lap, start_m, end_m, mass, elevation)
            row = {"event": name, "date": date.date(), "driver": drv, "team": teams.get(drv, ""),
                   "lap": lap_no, "run_start_m": fs.run_start_m,
                   "swing_independent_kw": ind["swing_w"] / 1e3 if ind else np.nan}
            for tag, f in (("step", fs), ("ramp", fr)):
                row.update({f"deploy_frac_{tag}": f.deploy_frac,
                            f"clip_start_frac_{tag}": f.clip_start_m / (grid[-1] - grid[0]),
                            f"harvest_kw_{tag}": f.harvest_w / 1e3, f"swing_fit_kw_{tag}": f.swing_w / 1e3,
                            f"energy_used_kj_{tag}": f.energy_used_kj, f"rmse_kmh_{tag}": f.rmse_kmh,
                            f"clip_range_m_{tag}": f.clip_range_m})
            rows.append({**row, "_v": v, "_mask": mask, "_grade": grade, "_mass": mass})
    print(f"{name}: {len(rows)} laps fitted")
    return pd.DataFrame(rows)


def sensitivity(laps: pd.DataFrame, car, ramp: bool) -> pd.DataFrame:
    tag = "ramp" if ramp else "step"
    sample = laps.sample(min(SENSITIVITY_LAPS, len(laps)), random_state=0)
    out = []
    for label, ice_mult, cda_mult in [("ICE -5%", 0.95, 1), ("ICE +5%", 1.05, 1),
                                      ("drag -10%", 1, 0.9), ("drag +10%", 1, 1.1)]:
        d = []
        for _, r in sample.iterrows():
            f = fit_straight(r["_v"], r["_mask"], r["_grade"], r["_mass"], car,
                             car.ice_power_w * ice_mult, car.cda_straight * cda_mult, ramp=ramp)
            if f is None:
                continue
            d.append((f.deploy_frac - r[f"deploy_frac_{tag}"], f.harvest_w / 1e3 - r[f"harvest_kw_{tag}"],
                      f.swing_w / 1e3 - r[f"swing_fit_kw_{tag}"],
                      f.energy_used_kj - r[f"energy_used_kj_{tag}"]))
        d = np.array(d)
        out.append({"assumption": label, "shift_deploy_frac": d[:, 0].mean(),
                    "shift_harvest_kw": d[:, 1].mean(), "shift_swing_kw": d[:, 2].mean(),
                    "shift_energy_kj": d[:, 3].mean()})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--change-event", default="Miami")
    p.add_argument("--out", default="results/straightfit")
    args = p.parse_args()

    from erslens.ingest import event_info

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    car = load_car_params("configs/car_2026.yaml")
    _, change_date = event_info(args.year, args.change_event)
    laps = pd.concat([analyse_event(args.year, ev, car) for ev in args.events], ignore_index=True)
    laps["period"] = np.where(pd.to_datetime(laps["date"]) >= change_date, "after", "before")
    laps.drop(columns=[c for c in laps.columns if c.startswith("_")]).to_csv(out / "fits_per_lap.csv", index=False)

    agg = {"laps": ("lap", "count"), "swing_independent_kw": ("swing_independent_kw", "median")}
    for tag in ("step", "ramp"):
        for col in ("rmse_kmh", "deploy_frac", "harvest_kw", "swing_fit_kw", "clip_range_m"):
            agg[f"{col}_{tag}"] = (f"{col}_{tag}", "median")
    races = laps.groupby(["event", "date", "period"]).agg(**agg).reset_index().sort_values("date")
    races.to_csv(out / "fits_per_race.csv", index=False)
    with pd.option_context("display.width", 250):
        print("\n" + races.round(2).to_string(index=False))

    ramp_wins = int((races["rmse_kmh_ramp"] < races["rmse_kmh_step"]).sum())
    preferred = "ramp" if ramp_wins >= 5 else "step"
    print(f"\nModel comparison: ramp has lower median fit error on {ramp_wins} of {len(races)} circuits "
          f"-> preferred model: {preferred.upper()} (rule: ramp if >= 5 of 7)")

    both = laps.dropna(subset=["swing_independent_kw"])
    for tag in ("step", "ramp"):
        rho, pval = spearmanr(both[f"swing_fit_kw_{tag}"], both["swing_independent_kw"])
        diff = np.median(both[f"swing_fit_kw_{tag}"] - both["swing_independent_kw"])
        print(f"Validation ({tag}): fitted vs independent swing on {len(both)} laps: "
              f"Spearman rho = {rho:.2f} (p = {pval:.1g}); median difference {diff:.0f} kW")

    sens = sensitivity(laps, car, ramp=preferred == "ramp")
    sens.to_csv(out / "sensitivity.csv", index=False)
    print(f"\nSensitivity of the {preferred} model to pinned assumptions (mean shift per lap):\n"
          + sens.round(2).to_string(index=False))
    print(f"\nSaved {out}/fits_per_lap.csv, fits_per_race.csv, sensitivity.csv")


if __name__ == "__main__":
    main()
