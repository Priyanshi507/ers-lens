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
from erslens.physics import mguk_power_cap
from multi_race import permutation_p

MIN_LAPS = 20


def analyse_event(year: int, event: str, car) -> pd.DataFrame:
    from erslens.ingest import driver_teams, event_info, load_driver_race, race_drivers, reference_track

    name, date = event_info(year, event)
    track = reference_track(year, event)
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
    elevation = None
    if track.elevation_m is not None:
        elevation = (track.distance_m, track.elevation_m)
        seg = track.elevation_m[(track.distance_m >= start_m) & (track.distance_m <= end_m)]
        print(f"{name}: lap elevation range {np.ptp(track.elevation_m):.0f} m, "
              f"longest straight rises {seg[-1] - seg[0]:+.1f} m over {end_m - start_m:.0f} m")
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
            r = electric_power_swing(lap, start_m, end_m, mass, elevation)
            row = {"event": name, "date": date.date(), "driver": drv,
                   "team": teams.get(drv, ""), "lap": lap_no, "swing_kw": np.nan,
                   "swing_level_kw": np.nan, "cap_kw": np.nan, "frac": np.nan}
            if r:
                mid = (r["band_lo_kmh"] + r["band_hi_kmh"]) / 2
                cap = mguk_power_cap(mid / 3.6, car)
                row.update(swing_kw=r["swing_w"] / 1e3, swing_level_kw=r["swing_level_w"] / 1e3,
                           band_mid_kmh=mid, cap_kw=cap / 1e3,
                           frac=r["swing_w"] / cap if cap > 0 else np.nan)
            rows.append(row)
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

    m = laps.dropna(subset=["frac"])
    q = lambda p: (lambda x: x.quantile(p))
    races = laps.groupby(["event", "date", "period"]).agg(
        clean_laps=("lap", "count"), measured=("frac", "count"),
        band_mid_kmh=("band_mid_kmh", "median"),
        swing_level_kw=("swing_level_kw", "median"), swing_kw=("swing_kw", "median"),
        frac_median=("frac", "median"), frac_q25=("frac", q(0.25)), frac_q75=("frac", q(0.75)),
    ).reset_index().sort_values("date")
    races["used"] = races["measured"] >= MIN_LAPS
    races.to_csv(out / "swing_per_race.csv", index=False)
    print("\n" + races.round(2).to_string(index=False))

    # Team effect within each race cancels circuit differences (gradient, speed band, layout).
    m = m.assign(rel=m["frac"] - m.groupby("event")["frac"].transform("median"))
    teams = m.groupby(["team", "period"])["rel"].median().unstack().round(3)
    teams["laps"] = m.groupby("team")["rel"].count()
    teams.to_csv(out / "team_effect_within_race.csv")
    print("\nTeam fraction withdrawn relative to the race median (+ = clips harder):\n"
          + teams.to_string())

    u = races[races["used"]]
    before = u.loc[u["period"] == "before", "frac_median"].tolist()
    after = u.loc[u["period"] == "after", "frac_median"].tolist()
    if before and after:
        print(f"\nFraction of available electric power withdrawn (race medians): "
              f"before {np.mean(before):.2f} ({len(before)} races) vs after {np.mean(after):.2f} "
              f"({len(after)} races); permutation p (after > before) = {permutation_p(after, before):.3f}, "
              f"(after < before) = {permutation_p(before, after):.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(u))
    colors = u["period"].map({"before": "tab:red", "after": "tab:blue"})
    ax.errorbar(x, u["frac_median"], yerr=[u["frac_median"] - u["frac_q25"],
                u["frac_q75"] - u["frac_median"]], fmt="none", ecolor="grey", capsize=3)
    ax.scatter(x, u["frac_median"], c=colors, s=45, zorder=3)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, [f"{e.replace(' Grand Prix', '')}\n({n})" for e, n in zip(u["event"], u["measured"])],
                  fontsize=8)
    ax.set_ylabel("fraction of available MGU-K power withdrawn\n(elevation-corrected, median and IQR)")
    ax.set_title("Clipping intensity: before (red) vs after (blue) the Miami rule change")
    fig.tight_layout()
    fig.savefig(out / "swing_per_race.png", dpi=150)
    print(f"\nSaved {out}/swing_per_lap.csv, swing_per_race.csv, team_effect_within_race.csv, swing_per_race.png")


if __name__ == "__main__":
    main()
