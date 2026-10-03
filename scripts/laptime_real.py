"""Time lost to clipping on each circuit's longest straight, per lap, race and team.

Checks predictions L1-L3 recorded in docs/research_log.md before running.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from derate_analysis import lap_quality, longest_flat_out_run
from erslens.energy import electric_power_swing
from erslens.laptime import clip_time_loss
from erslens.params import load_car_params

MIN_LAPS = 20


def analyse_event(year: int, event: str, car) -> pd.DataFrame:
    from erslens.ingest import driver_teams, event_info, load_driver_race, race_drivers, reference_track

    name, date = event_info(year, event)
    track = reference_track(year, event)
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
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
            mass = car.mass_kg + max(0.0, car.fuel_kg - car.fuel_per_lap_kg * lap_no)
            r = clip_time_loss(lap, start_m, end_m, mass, car, elevation)
            if r is None:
                continue
            sw = electric_power_swing(lap, start_m, end_m, mass, elevation)
            rows.append({"event": name, "date": date.date(), "driver": drv, "team": teams.get(drv, ""),
                         "lap": lap_no, "clipping": r["clipping"], "t_loss_s": r["t_loss_s"],
                         "t_low_s": r["t_low_s"], "t_high_s": r["t_high_s"],
                         "swing_kw": sw["swing_w"] / 1e3 if sw else np.nan})
    print(f"{name}: {len(rows)} laps analysed")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--out", default="results/laptime")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    car = load_car_params("configs/car_2026.yaml")
    laps = pd.concat([analyse_event(args.year, ev, car) for ev in args.events], ignore_index=True)
    laps.to_csv(out / "laptime_per_lap.csv", index=False)

    q = lambda p: (lambda x: x.quantile(p))  # noqa: E731
    races = laps.groupby(["event", "date"]).agg(
        laps=("lap", "count"), clipping_share=("clipping", "mean"),
        t_loss_median_s=("t_loss_s", "median"), t_loss_q25_s=("t_loss_s", q(0.25)),
        t_loss_q75_s=("t_loss_s", q(0.75))).reset_index().sort_values("date")
    races["used"] = races["laps"] >= MIN_LAPS
    races.to_csv(out / "laptime_per_race.csv", index=False)
    print("\n" + races.round(3).to_string(index=False))

    clip = laps[laps["clipping"]]
    clip = clip.assign(rel=clip["t_loss_s"] - clip.groupby("event")["t_loss_s"].transform("median"))
    teams = clip.groupby("team").agg(laps=("rel", "count"), vs_race_median_s=("rel", "median"))
    teams = teams.sort_values("vs_race_median_s")
    teams.to_csv(out / "laptime_team_vs_race.csv")
    print("\nTime lost to clipping relative to the race median (s per lap; + = loses more):\n"
          + teams.round(3).to_string())

    used = races[races["used"]]
    l1 = used["t_loss_median_s"].between(0.05, 0.50).all()
    both = clip.dropna(subset=["swing_kw"])
    rho = spearmanr(both["t_loss_s"], both["swing_kw"])[0] if len(both) > 10 else np.nan
    big = clip[clip["t_loss_s"] > 0.05]
    width = ((big["t_high_s"] - big["t_low_s"]) / big["t_loss_s"]).median()
    print("\nPredictions recorded before running (docs/research_log.md):")
    for label, ok, detail in [
        ("L1  every circuit's median time loss in 0.05-0.50 s", l1,
         ", ".join(f"{e.replace(' Grand Prix', '')} {m:.3f}" for e, m in zip(used["event"], used["t_loss_median_s"]))),
        ("L2  time loss vs matched-speed power drop, Spearman >= 0.4", rho >= 0.4, f"rho = {rho:.2f} on {len(both)} laps"),
        ("L3  median parameter-ambiguity range < 10% of estimate", width < 0.10, f"median width {width:.1%}"),
    ]:
        print(f"  {'HELD    ' if ok else 'NOT HELD'}  {label}: {detail}")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(used))
    ax.errorbar(x, used["t_loss_median_s"], yerr=[used["t_loss_median_s"] - used["t_loss_q25_s"],
                used["t_loss_q75_s"] - used["t_loss_median_s"]], fmt="o", capsize=4, color="tab:red")
    ax.set_xticks(x, [e.replace(" Grand Prix", "") for e in used["event"]], fontsize=9)
    ax.set_ylabel("time lost to clipping per lap (s)\nmedian and interquartile range")
    ax.set_title("Lap time lost to clipping on each circuit's longest straight")
    fig.tight_layout()
    fig.savefig(out / "laptime_per_race.png", dpi=150)
    print(f"\nSaved {out}/laptime_per_lap.csv, laptime_per_race.csv, laptime_team_vs_race.csv, laptime_per_race.png")


if __name__ == "__main__":
    main()
