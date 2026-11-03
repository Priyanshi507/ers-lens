"""Detect energy derating on the longest straight: speed falling while throttle stays flat out."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from erslens.ingest import load_track
from erslens.throttle import FULL_THROTTLE, NOISE_THROTTLE, full_throttle_mask
from erslens.track import Track

# ~24 m between samples at 340 km/h and 4 Hz; a larger gap means dropped telemetry.
MAX_GAP_M = 60.0
STALE_RUN = 4
STALE_MARGIN_KMH = 15.0
# Sensitivity check: identical speed for this many samples (~130 m) even at the peak.
STRICT_FROZEN_RUN = 6
# Throttle below full for more than this before the brake point counts as a partial lift.
PARTIAL_LIFT_M = 60.0
MIN_BRAKE_DROP_KMH = 100.0
MIN_FLAT_RUN_M = 300.0
# Far longer than a genuine top-speed plateau holds one integer value (~230 m).
FROZEN_ANYWHERE_RUN = 10


def longest_flat_out_run(track: Track, avoid_boundary: bool = False) -> tuple[float, float]:
    """Longest full-throttle stretch of the reference lap.

    avoid_boundary skips stretches touching the start/finish line: lap distance
    resets there, so a straight crossing it is split across two laps.
    """
    flat = ~np.isfinite(track.v_limit_ms)
    best, best_len, i = (0, 0), 0, 0
    while i < track.n:
        if flat[i]:
            j = i
            while j < track.n and flat[j]:
                j += 1
            touches = i == 0 or j == track.n
            if j - i > best_len and not (avoid_boundary and touches):
                best, best_len = (i, j), j - i
            i = j
        else:
            i += 1
    return float(track.distance_m[best[0]]), float(track.distance_m[best[1] - 1])


def boundary_offset(track: Track) -> float:
    """A lap boundary that no straight touches: the middle of the slowest corner. Its first
    point would leave the straight before that corner ending at the boundary."""
    corner = np.isfinite(track.v_limit_ms)
    k = int(np.argmin(np.where(corner, track.v_limit_ms, np.inf)))
    a = b = k
    while corner[(a - 1) % track.n] and (a - 1) % track.n != k:
        a -= 1
    while corner[(b + 1) % track.n] and (b + 1) % track.n != k:
        b += 1
    return float(track.distance_m[((a + b) // 2) % track.n])


def straight_at_boundary_m(track: Track) -> float:
    """Length of a full-throttle stretch touching the lap boundary (0 if none): the check that
    no straight is skipped."""
    flat = ~np.isfinite(track.v_limit_ms)
    run = 0
    if flat[0]:
        run += int(np.argmin(flat)) if not flat.all() else track.n
    if flat[-1]:
        run += int(np.argmin(flat[::-1])) if not flat.all() else 0
    return float(run * track.ds)


def roll_track(track: Track, offset_m: float) -> Track:
    """The reference lap re-started at offset_m, so distance 0 sits at that point."""
    k = int(np.searchsorted(track.distance_m, offset_m))
    roll = lambda a: None if a is None else np.roll(a, -k)
    return Track(track.name, track.distance_m, roll(track.v_limit_ms), roll(track.straight_mode),
                 roll(track.elevation_m), roll(track.deploy_cap_w))


def rebase_laps(race: pd.DataFrame, offset_m: float) -> pd.DataFrame:
    """Laps re-cut at offset_m instead of the start/finish line, so a straight crossing the line
    is measured whole. New lap k runs from offset_m in lap k to offset_m in lap k+1; it is marked
    as a pit lap if either source lap was."""
    laps = dict(tuple(race.groupby("lap")))
    out = []
    for k, cur in laps.items():
        nxt = laps.get(k + 1)
        if nxt is None:
            continue
        a = cur[cur["distance_m"] >= offset_m]
        b = nxt[nxt["distance_m"] < offset_m]
        if a.empty or b.empty:
            continue
        lap_m, lap_s, t0 = cur["distance_m"].max(), cur["time_s"].max(), a["time_s"].iloc[0]
        a = a.assign(distance_m=a["distance_m"] - offset_m, time_s=a["time_s"] - t0)
        b = b.assign(distance_m=b["distance_m"] + lap_m - offset_m, time_s=b["time_s"] + lap_s - t0)
        pit = bool(cur["pit_in"].any() or cur["pit_out"].any() or nxt["pit_in"].any() or nxt["pit_out"].any())
        out.append(pd.concat([a, b]).assign(lap=k, pit_in=pit, pit_out=pit))
    return pd.concat(out, ignore_index=True) if out else race.iloc[0:0]


def lap_quality(lap: pd.DataFrame, start_m: float, end_m: float) -> tuple[dict | None, str]:
    """Return (metrics, "ok") or (None, reason the lap was rejected)."""
    win = lap[(lap["distance_m"] >= start_m) & (lap["distance_m"] <= end_m + 100)]
    thr = win["throttle"].to_numpy()
    no_brake = ~win["brake"].astype(bool).to_numpy()
    flat = full_throttle_mask(thr, ~no_brake)
    if flat.sum() < 5:
        return None, "no_flat_out"
    speed = win["speed_kmh"].to_numpy()
    dist = win["distance_m"].to_numpy()
    peak = int(np.argmax(np.where(flat, speed, -np.inf)))
    # Follow only the unbroken flat-out stretch after the peak; later flat-out
    # samples belong to the exit of the next corner.
    last = peak
    while last + 1 < len(flat) and flat[last + 1]:
        last += 1
    if last + 1 >= len(flat):
        return None, "data_ends_before_braking"
    # Repeated identical speeds while still accelerating hard mean a frozen channel.
    # Near the peak the car barely accelerates, so repeats there are genuine.
    accel = (np.arange(len(speed)) < peak) & flat & (speed < speed[peak] - STALE_MARGIN_KMH)
    run = 1
    for k in range(1, len(speed)):
        run = run + 1 if accel[k] and accel[k - 1] and speed[k] == speed[k - 1] else 1
        if run >= STALE_RUN:
            return None, "stale_speed"
    brake_zone = np.flatnonzero((np.arange(len(speed)) > last)
                                & ((thr < 20) | ~no_brake))
    if len(brake_zone) == 0:
        return None, "data_ends_before_braking"
    brake_idx = int(brake_zone[0])
    # Gaps anywhere up to the braking point can hide a speed loss, not just gaps
    # inside the flat-out stretch.
    if np.diff(dist[: brake_idx + 1]).max(initial=0.0) > MAX_GAP_M:
        return None, "telemetry_gap"
    # Physical plausibility: a genuine lap brakes hard for the corner, has a long
    # flat-out run, and never shows an identical speed repeated along the straight.
    if speed[brake_idx:].min() > speed[peak] - MIN_BRAKE_DROP_KMH:
        return None, "no_braking_decel"
    first = peak
    while first > 0 and flat[first - 1]:
        first -= 1
    if dist[last] - dist[first] < MIN_FLAT_RUN_M:
        return None, "short_flat_run"
    # Checked through the braking zone too: speed cannot stay constant once the
    # throttle is closed, so a repeat there is a frozen channel.
    same = np.concatenate([[False], speed[1:] == speed[:-1]])
    run = 0
    for k in range(len(speed)):
        run = run + 1 if same[k] else 0
        if run + 1 >= FROZEN_ANYWHERE_RUN:
            return None, "frozen_channel"
    frozen, longest = 1, 1
    for k in range(1, last + 1):
        frozen = frozen + 1 if flat[k] and flat[k - 1] and speed[k] == speed[k - 1] else 1
        longest = max(longest, frozen)
    return {
        "max_frozen_run": longest,
        "top_speed_kmh": float(speed[peak]),
        "end_speed_kmh": float(speed[last]),
        "flat_out_drop_kmh": float(speed[peak] - speed[last]),
        "peak_before_brake_m": float(dist[last] - dist[peak]),
        "lift_before_brake_m": float(dist[brake_idx] - dist[last]),
        "lift_min_throttle": float(thr[last + 1:brake_idx].min(initial=100.0)),
        "drop_to_brake_kmh": float(speed[peak] - speed[brake_idx - 1]),
    }, "ok"


def lap_metrics(lap: pd.DataFrame, start_m: float, end_m: float) -> dict | None:
    return lap_quality(lap, start_m, end_m)[0]


def analyse_driver(race: pd.DataFrame, start_m: float, end_m: float,
                   pace_ref: str = "median") -> pd.DataFrame:
    """pace_ref="fastest" suits qualifying, where out- and cool-down laps dominate the median."""
    durations = race.groupby("lap")["time_s"].max()
    ref = durations.min() if pace_ref == "fastest" else durations.median()
    clean = durations[durations <= 1.07 * ref].index
    rows, reasons = [], {}
    for lap_no, lap in race.groupby("lap"):
        if lap_no == 1 or lap_no not in clean:
            continue
        if lap["pit_in"].iloc[0] or lap["pit_out"].iloc[0]:
            continue
        m, reason = lap_quality(lap.reset_index(drop=True), start_m, end_m)
        if m:
            rows.append({"lap": lap_no, "lap_time_s": float(durations[lap_no]), **m})
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    out = pd.DataFrame(rows)
    out.attrs["rejected"] = reasons
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", required=True)
    p.add_argument("--races", nargs="+", required=True, help="per-driver race parquet files")
    p.add_argument("--drop-threshold", type=float, default=3.0,
                   help="km/h drop at full throttle counted as a derate event")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    track = load_track(args.track)
    start_m, end_m = longest_flat_out_run(track)
    print(f"Longest flat-out run: {start_m:.0f}-{end_m:.0f} m ({end_m - start_m:.0f} m)\n")

    per_driver, totals = {}, {}
    for path in args.races:
        drv = Path(path).stem.split("_")[-2]
        res = analyse_driver(pd.read_parquet(path), start_m, end_m)
        for reason, count in res.attrs.get("rejected", {}).items():
            totals[reason] = totals.get(reason, 0) + count
        res.insert(0, "driver", drv)
        per_driver[drv] = res

    allres = pd.concat(per_driver.values(), ignore_index=True)
    n_rej = sum(totals.values())
    print(f"Laps kept: {len(allres)}, rejected for data quality: {n_rej} "
          f"({n_rej / (len(allres) + n_rej):.0%})")
    for reason, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {reason}: {count}")
    print()
    allres.to_csv(out / "derate_per_lap.csv", index=False)

    allres["derate"] = allres["flat_out_drop_kmh"] >= args.drop_threshold
    summary = allres.groupby("driver").agg(
        clean_laps=("lap", "count"),
        top_speed_median=("top_speed_kmh", "median"),
        top_speed_spread=("top_speed_kmh", lambda s: s.quantile(0.9) - s.quantile(0.1)),
        derate_laps=("derate", "sum"),
        derate_share=("derate", "mean"),
        mean_drop_when_derated=("flat_out_drop_kmh", lambda s: s[s >= args.drop_threshold].mean()),
    ).round(2)
    summary.to_csv(out / "derate_summary.csv")
    print(summary.to_string())

    strict = allres[allres["max_frozen_run"] < STRICT_FROZEN_RUN]
    sens = pd.DataFrame({
        "share_normal": allres.groupby("driver")["derate"].mean(),
        "laps_strict": strict.groupby("driver")["lap"].count(),
        "share_strict": strict.groupby("driver")["derate"].mean(),
    }).round(2)
    print(f"\nSensitivity: dropping laps with speed frozen for >= {STRICT_FROZEN_RUN} samples "
          f"keeps {len(strict)}/{len(allres)} laps; "
          f"flat-out drop share {allres['derate'].mean():.1%} -> {strict['derate'].mean():.1%}")
    print(sens.sort_values("share_strict").to_string())
    sens.to_csv(out / "derate_sensitivity.csv")

    allres["partial_lift"] = ((allres["lift_before_brake_m"] > PARTIAL_LIFT_M)
                              & (allres["lift_min_throttle"] < 90))
    allres["any_drop"] = allres["drop_to_brake_kmh"] >= args.drop_threshold
    s2 = allres[allres["max_frozen_run"] < STRICT_FROZEN_RUN]
    styles = pd.DataFrame({
        "laps": s2.groupby("driver")["lap"].count(),
        "clip_at_full_throttle": s2.groupby("driver")["derate"].mean(),
        "partial_lift": s2.groupby("driver")["partial_lift"].mean(),
        "any_speed_loss_before_brake": s2.groupby("driver")["any_drop"].mean(),
    }).round(2)
    print(f"\nEnergy technique by driver (strict laps; partial lift = throttle below full "
          f"below 90% for > {PARTIAL_LIFT_M:g} m before braking):")
    print(styles.sort_values("clip_at_full_throttle").to_string())
    styles.to_csv(out / "derate_styles.csv")

    corr = allres[["flat_out_drop_kmh", "lap_time_s"]].corr().iloc[0, 1]
    print(f"\nCorrelation(flat-out drop, lap time) across all clean laps: {corr:.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for drv, res in per_driver.items():
        ax[0].plot(res["lap"], res["top_speed_kmh"], ".-", lw=0.8, label=drv)
        ax[1].plot(res["lap"], res["flat_out_drop_kmh"], ".-", lw=0.8, label=drv)
    ax[0].set_ylabel("top speed km/h")
    ax[1].set_ylabel("speed lost at\nfull throttle km/h")
    ax[1].axhline(args.drop_threshold, color="grey", ls="--", lw=0.7)
    ax[1].set_xlabel("lap")
    ax[0].legend(ncol=len(per_driver))
    fig.suptitle("Longest straight: top speed and flat-out speed loss per lap")
    fig.tight_layout()
    fig.savefig(out / "derate_by_lap.png", dpi=130)
    print(f"\nSaved {out}/derate_per_lap.csv, derate_summary.csv, derate_by_lap.png")


if __name__ == "__main__":
    main()
