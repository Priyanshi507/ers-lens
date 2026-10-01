"""Fit the 3-parameter energy strategy to every clean race lap on each circuit's longest straight.

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

SENSITIVITY_LAPS = 80


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
            f = fit_straight(v, mask, grade, mass, car)
            if f is None:
                continue
            ind = electric_power_swing(lap, start_m, end_m, mass, elevation)
            rows.append({"event": name, "date": date.date(), "driver": drv, "team": teams.get(drv, ""),
                         "lap": lap_no, "deploy_frac": f.deploy_frac,
                         "clip_start_frac": f.clip_start_m / (grid[-1] - grid[0]),
                         "clip_range_m": f.clip_range_m, "run_start_m": f.run_start_m,
                         "harvest_kw": f.harvest_w / 1e3, "swing_fit_kw": f.swing_w / 1e3,
                         "energy_used_kj": f.energy_used_kj, "rmse_kmh": f.rmse_kmh,
                         "swing_independent_kw": ind["swing_w"] / 1e3 if ind else np.nan,
                         "_v": v, "_mask": mask, "_grade": grade, "_mass": mass})
    print(f"{name}: {len(rows)} laps fitted")
    return pd.DataFrame(rows)


def sensitivity(laps: pd.DataFrame, car) -> pd.DataFrame:
    sample = laps.sample(min(SENSITIVITY_LAPS, len(laps)), random_state=0)
    out = []
    for label, ice_mult, cda_mult in [("ICE -5%", 0.95, 1), ("ICE +5%", 1.05, 1),
                                      ("drag -10%", 1, 0.9), ("drag +10%", 1, 1.1)]:
        d = []
        for _, r in sample.iterrows():
            f = fit_straight(r["_v"], r["_mask"], r["_grade"], r["_mass"], car,
                             car.ice_power_w * ice_mult, car.cda_straight * cda_mult)
            if f is None:
                continue
            d.append((f.deploy_frac - r["deploy_frac"], f.harvest_w / 1e3 - r["harvest_kw"],
                      f.swing_w / 1e3 - r["swing_fit_kw"], f.energy_used_kj - r["energy_used_kj"]))
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

    races = laps.groupby(["event", "date", "period"]).agg(
        laps=("lap", "count"), deploy_frac=("deploy_frac", "median"),
        clip_start_frac=("clip_start_frac", "median"), harvest_kw=("harvest_kw", "median"),
        swing_fit_kw=("swing_fit_kw", "median"), swing_independent_kw=("swing_independent_kw", "median"),
        energy_used_kj=("energy_used_kj", "median"), rmse_kmh=("rmse_kmh", "median"),
        clip_range_m=("clip_range_m", "median"),
    ).reset_index().sort_values("date")
    races.to_csv(out / "fits_per_race.csv", index=False)
    print("\n" + races.round(2).to_string(index=False))

    both = laps.dropna(subset=["swing_independent_kw"])
    rho, pval = spearmanr(both["swing_fit_kw"], both["swing_independent_kw"])
    print(f"\nValidation, fitted vs independent swing on {len(both)} laps: Spearman rho = {rho:.2f} "
          f"(p = {pval:.1g}); median difference {np.median(both.swing_fit_kw - both.swing_independent_kw):.0f} kW")

    before = races.loc[races["period"] == "before", "harvest_kw"].tolist()
    after = races.loc[races["period"] == "after", "harvest_kw"].tolist()
    if before and after:
        print(f"Harvest power H (race medians): before {np.mean(before):.0f} kW vs after {np.mean(after):.0f} kW; "
              f"permutation p (after > before) = {permutation_p(after, before):.3f}")

    sens = sensitivity(laps, car)
    sens.to_csv(out / "sensitivity.csv", index=False)
    print("\nSensitivity to pinned assumptions (mean shift per lap):\n" + sens.round(2).to_string(index=False))
    print(f"\nSaved {out}/fits_per_lap.csv, fits_per_race.csv, sensitivity.csv")


if __name__ == "__main__":
    main()
