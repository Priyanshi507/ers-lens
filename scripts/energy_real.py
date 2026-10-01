"""Electric power swing during clipping on real 2026 races, per lap, team and race.

Prediction stated before running: from Miami the FIA raised peak super-clipping power
from 250 kW to 350 kW, so if that applies in races the swing should rise after Miami.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from derate_analysis import lap_quality, longest_flat_out_run
from erslens.energy import electric_power_swing
from erslens.params import load_car_params
from multi_race import permutation_p

MIN_LAPS = 20


def analyse_event(year: int, event: str, car) -> pd.DataFrame:
    from erslens.ingest import driver_teams, event_info, load_driver_race, race_drivers, reference_track

    name, date = event_info(year, event)
    track = reference_track(year, event)
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
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
            mass = car.mass_kg + max(0.0, car.fuel_kg - car.fuel_per_lap_kg * lap_no)
            r = electric_power_swing(lap, start_m, end_m, mass)
            rows.append({"event": name, "date": date.date(), "driver": drv,
                         "team": teams.get(drv, ""), "lap": lap_no,
                         "swing_kw": r["swing_w"] / 1e3 if r else np.nan,
                         "band_kmh": f"{r['band_lo_kmh']:.0f}-{r['band_hi_kmh']:.0f}" if r else ""})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--change-event", default="Miami")
    p.add_argument("--out", default="results/energy")
    args = p.parse_args()

    from erslens.ingest import event_info

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    car = load_car_params("configs/car_2026.yaml")
    _, change_date = event_info(args.year, args.change_event)
    laps = pd.concat([analyse_event(args.year, ev, car) for ev in args.events], ignore_index=True)
    laps["period"] = np.where(pd.to_datetime(laps["date"]) >= change_date, "after", "before")
    laps.to_csv(out / "swing_per_lap.csv", index=False)

    m = laps.dropna(subset=["swing_kw"])
    races = laps.groupby(["event", "date", "period"]).agg(
        clean_laps=("lap", "count"), measured=("swing_kw", "count"),
        swing_median_kw=("swing_kw", "median"),
        swing_q25=("swing_kw", lambda s: s.quantile(0.25)),
        swing_q75=("swing_kw", lambda s: s.quantile(0.75))).reset_index().sort_values("date")
    races["used"] = races["measured"] >= MIN_LAPS
    races.to_csv(out / "swing_per_race.csv", index=False)
    print(races.round(1).to_string(index=False))

    teams = m.groupby(["team", "period"])["swing_kw"].median().unstack().round(0)
    teams.to_csv(out / "swing_per_team.csv")
    print("\nMedian swing (kW) by team:\n" + teams.to_string())

    u = races[races["used"]]
    before = u.loc[u["period"] == "before", "swing_median_kw"].tolist()
    after = u.loc[u["period"] == "after", "swing_median_kw"].tolist()
    if before and after:
        p_up = permutation_p(after, before)
        print(f"\nRace-level median swing: before {np.mean(before):.0f} kW ({len(before)} races) "
              f"vs after {np.mean(after):.0f} kW ({len(after)} races); "
              f"one-sided permutation p (after > before) = {p_up:.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(u))
    colors = u["period"].map({"before": "tab:red", "after": "tab:blue"})
    ax.errorbar(x, u["swing_median_kw"], yerr=[u["swing_median_kw"] - u["swing_q25"],
                u["swing_q75"] - u["swing_median_kw"]], fmt="none", ecolor="grey", capsize=3)
    ax.scatter(x, u["swing_median_kw"], c=colors, s=45, zorder=3)
    ax.set_xticks(x, [f"{e.replace(' Grand Prix', '')}\n({n})" for e, n in zip(u["event"], u["measured"])],
                  fontsize=8)
    ax.set_ylabel("electric power swing during clipping (kW)\nmedian and interquartile range")
    ax.set_title("Electric power swing: before (red) vs after (blue) the Miami rule change")
    fig.tight_layout()
    fig.savefig(out / "swing_per_race.png", dpi=150)
    print(f"\nSaved {out}/swing_per_lap.csv, swing_per_race.csv, swing_per_team.csv, swing_per_race.png")


if __name__ == "__main__":
    main()
