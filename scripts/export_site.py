"""Export analysis results to the JSON the ERS-Lens Explorer site reads.

Reads results/laptime/laptime_per_lap.csv and laptime_per_race.csv (from laptime_real.py)
and, with FastF1 data, each lap's speed trace on the analysed straight. Writes
site/data/index.json and one site/data/races/<id>.json per race.
"""
import argparse
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from erslens.laptime import SIM_ERROR_90_S
from laptime_real import MAX_FAILURE_SHARE, MIN_LAPS, MIN_MEASURABLE_SHARE

# Bump when the JSON layout changes, so the site can refuse data it does not understand.
SCHEMA_VERSION = 4
MIAMI_CHANGES = pd.Timestamp("2026-05-01")
# A team median from a handful of laps looks as solid on a bar chart as one from forty.
MIN_TEAM_LAPS = 10
# Worst median per-lap error across the out-of-model stress scenarios (laptime_stress.py).
TYPICAL_LAP_ERROR_S = 0.06


def race_id(event_name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", event_name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower().replace("grand prix", "")).strip("-")


def not_reported_reason(row) -> str | None:
    """Plain-language reason a race failed laptime_real's reporting rule; None if it passed."""
    if bool(row["valid"]):
        return None
    if row["laps"] < MIN_LAPS and row["measurable_share"] >= MIN_MEASURABLE_SHARE:
        return f"Only {int(row['laps'])} laps could be measured, too few to report."
    if row["measurable_share"] < MIN_MEASURABLE_SHARE:
        return (f"Only {row['measurable_share']:.0%} of clean laps had enough full-throttle running "
                "before the speed peak to measure.")
    if row["failure_share"] > MAX_FAILURE_SHARE:
        return (f"{row['failure_share']:.0%} of laps gave physically impossible results, "
                "so the method is not valid on this straight.")
    raise ValueError(f"{row['event']} is marked invalid but fails no known rule")


def summarise_races(per_race: pd.DataFrame) -> list[dict]:
    races = []
    for row in per_race.sort_values("date").to_dict("records"):
        row["laps"] = 0 if pd.isna(row.get("laps")) else row["laps"]
        row["failure_share"] = 0.0 if pd.isna(row.get("failure_share")) else row["failure_share"]
        reason = not_reported_reason(row)
        date = pd.Timestamp(row["date"])
        races.append({
            "id": race_id(row["event"]),
            "name": row["event"],
            "date": date.date().isoformat(),
            "period": "before" if date < MIAMI_CHANGES else "after",
            "clean_laps": int(row["clean_laps"]),
            "measured_laps": int(row["laps"]),
            "reported": reason is None,
            "reason": reason,
            "clipping_share": None if reason else round(float(row["clipping_share"]), 3),
            "loss_median_s": None if reason else round(float(row["t_loss_median_s"]), 3),
            "loss_q25_s": None if reason else round(float(row["t_loss_q25_s"]), 3),
            "loss_q75_s": None if reason else round(float(row["t_loss_q75_s"]), 3),
        })
    return races


def usable_laps(laps: pd.DataFrame) -> pd.DataFrame:
    measured = laps[laps["measurable"].astype(bool)]
    return measured[measured["t_loss_s"] >= -SIM_ERROR_90_S]


def team_table(laps: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Teams with enough clipping laps to chart, and the teams left out with their lap counts."""
    clip = usable_laps(laps)
    clip = clip[clip["clipping"].astype(bool)]
    if clip.empty:
        return [], []
    race_median = clip["t_loss_s"].median()
    rows = (clip.groupby("team")["t_loss_s"].agg(["median", "count"]).reset_index()
            .sort_values("median"))
    shown = [{"team": r.team, "laps": int(r["count"]), "loss_median_s": round(float(r["median"]), 3),
              "vs_race_s": round(float(r["median"] - race_median), 3)}
             for _, r in rows.iterrows() if r["count"] >= MIN_TEAM_LAPS]
    left_out = [{"team": r.team, "laps": int(r["count"])}
                for _, r in rows.sort_values("team").iterrows() if r["count"] < MIN_TEAM_LAPS]
    return shown, left_out


def lap_records(laps: pd.DataFrame, traces: dict, max_power_kw: float) -> list[dict]:
    """One record per usable lap.

    The interval is the spread across plausible car setups widened by the simulation's 90th-percentile
    error, matching clip_time_loss's interval_low_s/interval_high_s. A power drop larger than the
    MGU-K can deliver is an estimator failure on that lap, so it is withheld rather than shown.
    """
    records = []
    for r in usable_laps(laps).sort_values(["team", "driver", "lap"]).itertuples(index=False):
        trace = traces.get((r.driver, int(r.lap)))
        drop = None if pd.isna(r.swing_kw) or r.swing_kw > max_power_kw else round(float(r.swing_kw))
        rec = {"driver": r.driver, "team": r.team, "lap": int(r.lap), "clipping": bool(r.clipping),
               "loss_s": round(float(r.t_loss_s), 3),
               "loss_low_s": round(float(r.t_low_s) - SIM_ERROR_90_S, 3),
               "loss_high_s": round(float(r.t_high_s) + SIM_ERROR_90_S, 3),
               "power_drop_kw": drop, "power_drop_implausible": bool(r.swing_kw > max_power_kw)}
        if trace is not None:
            rec["trace"] = trace
        records.append(rec)
    return records


def lap_trace(lap: pd.DataFrame, start_m: float, end_m: float) -> dict | None:
    """Speed along the straight, distance from its start, and the full-throttle run."""
    from erslens.throttle import full_throttle_mask, run_around

    seg = lap[lap["distance_m"].between(start_m, end_m + 100)]
    if len(seg) < 5:
        return None
    flat = full_throttle_mask(seg["throttle"].to_numpy(), seg["brake"].to_numpy())
    v = seg["speed_kmh"].to_numpy()
    if not flat.any():
        return None
    peak = int(np.argmax(np.where(flat, v, -np.inf)))
    a, b = run_around(flat, peak)
    return {"d": [int(round(x)) for x in seg["distance_m"].to_numpy() - start_m],
            "v": [int(round(x)) for x in v], "flat": [a, b], "peak": peak}


def collect_traces(year: int, event: str, laps: pd.DataFrame) -> tuple[dict, dict]:
    from derate_analysis import longest_flat_out_run
    from erslens.ingest import load_driver_race, reference_track

    track = reference_track(year, event)
    start_m, end_m = longest_flat_out_run(track, avoid_boundary=True)
    wanted = set(zip(laps["driver"], laps["lap"].astype(int)))
    traces = {}
    for drv in sorted(laps["driver"].unique()):
        race = load_driver_race(year, event, drv)
        for lap_no, lap in race.groupby("lap"):
            if (drv, int(lap_no)) in wanted:
                t = lap_trace(lap.reset_index(drop=True), start_m, end_m)
                if t is not None:
                    traces[(drv, int(lap_no))] = t
    straight = {"start_m": round(float(start_m)), "end_m": round(float(end_m)),
                "length_m": round(float(end_m - start_m))}
    return traces, straight


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--events", nargs="+", required=True, help="FastF1 event names, e.g. Australia China")
    p.add_argument("--results", default="results/laptime")
    p.add_argument("--out", default="site/data")
    p.add_argument("--no-traces", action="store_true")
    args = p.parse_args()

    per_race = pd.read_csv(Path(args.results) / "laptime_per_race.csv")
    per_lap = pd.read_csv(Path(args.results) / "laptime_per_lap.csv")
    out = Path(args.out)
    (out / "races").mkdir(parents=True, exist_ok=True)

    from erslens.params import load_car_params

    max_power_kw = load_car_params("configs/car_2026.yaml").mguk_power_w / 1e3
    races = summarise_races(per_race)
    by_name = {r["name"]: r for r in races}
    for event in args.events:
        from erslens.ingest import event_info

        name, _ = event_info(args.year, event)
        race = by_name.get(name)
        if race is None:
            continue
        laps = per_lap[per_lap["event"] == name]
        skip = args.no_traces or not race["reported"]
        traces, straight = ({}, None) if skip else collect_traces(args.year, event, usable_laps(laps))
        teams, few = team_table(laps) if race["reported"] else ([], [])
        detail = {"schema": SCHEMA_VERSION, **race, "straight": straight,
                  "teams": teams, "teams_too_few_laps": few, "min_team_laps": MIN_TEAM_LAPS,
                  "laps": lap_records(laps, traces, max_power_kw) if race["reported"] else []}
        (out / "races" / f"{race['id']}.json").write_text(json.dumps(detail, separators=(",", ":")))
        withheld = sum(lap["power_drop_implausible"] for lap in detail["laps"])
        print(f"{name}: {len(detail['laps'])} laps, {len(traces)} traces, "
              f"{withheld} power drops above {max_power_kw:.0f} kW withheld")

    # A car's unknown setup biases all of its laps the same way, so a team median keeps the full
    # per-car error; the band uses the 90th-percentile simulation error, not the per-lap typical one.
    index = {"schema": SCHEMA_VERSION, "season": args.year,
             "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
             "accuracy": {"typical_lap_s": TYPICAL_LAP_ERROR_S, "team_band_s": SIM_ERROR_90_S},
             "races": races}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"Wrote {out}/index.json and {len(races)} race files")


if __name__ == "__main__":
    main()
