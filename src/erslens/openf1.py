"""OpenF1 (api.openf1.org) as a second source of the same F1 timing feed.

Since about 15 October 2026 the F1 live-timing archive that FastF1 reads returns HTTP 401 without
an F1 TV subscription (FastF1 issue #978). OpenF1 serves its own recording of the feed, free for
2023 onward. This module rebuilds the tables erslens.ingest returns from FastF1, following
FastF1's conventions where they affect results: lap telemetry is padded and interpolated to the
exact lap start and end, distance is integrated from speed the way FastF1's add_distance does, and
position units are the feed's tenths of a metre. scripts/compare_sources.py checks the two sources
against each other on races where both exist, before OpenF1 data is used for anything.

Select it with ERSLENS_SOURCE=openf1. Responses are cached under data/cache/openf1.
"""
import hashlib
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

BASE = "https://api.openf1.org/v1"
CACHE = Path("data/cache/openf1")
# The free tier allows 30 requests a minute.
MIN_INTERVAL_S = 2.1
SESSION_NAME = {"R": "Race", "Q": "Qualifying"}
_last_request = [0.0]


def _url(endpoint: str, filters: list[tuple[str, object]]) -> str:
    """Filters are (key with operator, value) pairs, e.g. ("date>=", t): OpenF1 puts the operator in the key."""
    parts = [f"{k}{'' if k[-1] in '<>=' else '='}{quote(str(v), safe='')}" for k, v in filters]
    return f"{BASE}/{endpoint}" + ("?" + "&".join(parts) if parts else "")


def get(endpoint: str, *filters: tuple[str, object], cache_dir: Path = CACHE) -> list[dict]:
    import requests

    url = _url(endpoint, list(filters))
    path = Path(cache_dir) / f"{endpoint}_{hashlib.sha1(url.encode()).hexdigest()[:16]}.json"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request[0])
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
        res = requests.get(url, timeout=120)
        if res.status_code == 404:
            data = []  # OpenF1 answers 404 when a filter matches nothing
            break
        if res.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (attempt + 1))
            continue
        res.raise_for_status()
        data = res.json()
        break
    else:
        raise RuntimeError(f"OpenF1 kept failing for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True, format="ISO8601")


def match_meeting(meetings: list[dict], event_date: pd.Timestamp) -> dict:
    """The meeting whose last day is closest to the race date (FastF1's EventDate)."""
    best = min((m for m in meetings if not m.get("is_cancelled")),
               key=lambda m: abs((pd.Timestamp(m["date_end"]).tz_convert(None).normalize()
                                  - pd.Timestamp(event_date).normalize()).days))
    gap = abs((pd.Timestamp(best["date_end"]).tz_convert(None).normalize() - pd.Timestamp(event_date).normalize()).days)
    if gap > 2:
        raise LookupError(f"no OpenF1 meeting within 2 days of {event_date.date()}")
    return best


@lru_cache(maxsize=32)
def session_key(year: int, event_date: pd.Timestamp, kind: str) -> int:
    meeting = match_meeting(get("meetings", ("year", year)), event_date)
    sessions = [s for s in get("sessions", ("meeting_key", meeting["meeting_key"]))
                if s["session_name"] == SESSION_NAME[kind]]
    if len(sessions) != 1:
        raise LookupError(f"{meeting['meeting_name']}: {len(sessions)} OpenF1 sessions named {SESSION_NAME[kind]}")
    return int(sessions[0]["session_key"])


def drivers(key: int) -> pd.DataFrame:
    return pd.DataFrame(get("drivers", ("session_key", key))).drop_duplicates("driver_number")


def lap_windows(laps: pd.DataFrame) -> pd.DataFrame:
    """Start and end time of every lap with a known start; the end is the next lap's start if the duration is missing."""
    laps = laps[laps["date_start"].notna()].sort_values("lap_number").copy()
    laps["start"] = _utc(laps["date_start"]).to_numpy()
    nxt = laps["start"].shift(-1)
    dur = pd.to_timedelta(laps["lap_duration"], unit="s")
    laps["end"] = (laps["start"] + dur).where(laps["lap_duration"].notna(), nxt)
    return laps[laps["end"].notna()]


def slice_lap(car: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Samples inside [start, end] plus interpolated samples at both edges, as FastF1's
    Lap.get_car_data(pad=1, interpolate_edges=True); distance integrated as FastF1's add_distance."""
    t = (car["date"] - start).dt.total_seconds().to_numpy()
    span = (end - start).total_seconds()
    inside = np.flatnonzero((t >= 0) & (t <= span))
    if len(inside) < 5:
        return None
    lo, hi = max(inside[0] - 1, 0), min(inside[-1] + 1, len(t) - 1)
    edge_t = np.array([0.0, span])
    cont = {c: np.interp(edge_t, t[lo:hi + 1], car[c].to_numpy(dtype=float)[lo:hi + 1])
            for c in ("speed", "throttle", "rpm")}
    near = [lo if t[inside[0]] > 0 else inside[0], hi if t[inside[-1]] < span else inside[-1]]
    tt = np.concatenate([[0.0], t[inside], [span]])
    pick = lambda c: np.concatenate([[car[c].to_numpy()[near[0]]], car[c].to_numpy()[inside],  # noqa: E731
                                     [car[c].to_numpy()[near[1]]]])
    speed = np.concatenate([[cont["speed"][0]], car["speed"].to_numpy(dtype=float)[inside], [cont["speed"][1]]])
    dt = np.diff(tt, prepend=0.0)
    return pd.DataFrame({
        "time_s": tt,
        "distance_m": np.cumsum(speed / 3.6 * dt),
        "speed_kmh": speed,
        "throttle": np.concatenate([[cont["throttle"][0]], car["throttle"].to_numpy(dtype=float)[inside],
                                    [cont["throttle"][1]]]),
        "brake": pick("brake").astype(float) > 0,
        "gear": pick("n_gear"),
        "rpm": np.concatenate([[cont["rpm"][0]], car["rpm"].to_numpy(dtype=float)[inside], [cont["rpm"][1]]]),
    })


def _car(key: int, number: int, *window) -> pd.DataFrame:
    car = pd.DataFrame(get("car_data", ("session_key", key), ("driver_number", number), *window))
    if car.empty:
        return car
    car["date"] = _utc(car["date"]).to_numpy()
    return car.sort_values("date").reset_index(drop=True)


def driver_race(key: int, acronym: str) -> pd.DataFrame:
    d = drivers(key)
    row = d[d["name_acronym"] == acronym]
    if row.empty:
        return pd.DataFrame()
    number = int(row["driver_number"].iloc[0])
    laps = pd.DataFrame(get("laps", ("session_key", key), ("driver_number", number)))
    car = _car(key, number)
    if laps.empty or car.empty:
        return pd.DataFrame()
    pit_laps = {int(p["lap_number"]) for p in get("pit", ("session_key", key), ("driver_number", number))
                if p.get("lap_number") is not None}
    stints = get("stints", ("session_key", key), ("driver_number", number))
    frames = []
    for lap in lap_windows(laps).itertuples(index=False):
        df = slice_lap(car, lap.start, lap.end)
        if df is None:
            continue
        n = int(lap.lap_number)
        df.insert(0, "lap", n)
        df["compound"] = next((s["compound"] for s in stints
                               if s["lap_start"] is not None and s["lap_start"] <= n <= (s["lap_end"] or n)), None)
        df["pit_in"] = n in pit_laps
        df["pit_out"] = bool(lap.is_pit_out_lap)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


DELETED = re.compile(r"CAR (\d+).*?TIME (\d+):(\d+\.\d+) DELETED")


def deleted_laps(messages: list[dict]) -> set[tuple[int, float]]:
    """(driver number, lap time) of every lap time race control deleted."""
    out = set()
    for m in messages:
        hit = DELETED.search(m.get("message") or "")
        if hit:
            out.add((int(hit.group(1)), round(int(hit.group(2)) * 60 + float(hit.group(3)), 3)))
    return out


def fastest_valid_lap(laps: pd.DataFrame, deleted: set[tuple[int, float]]) -> pd.Series:
    laps = laps[laps["lap_duration"].notna() & ~laps["is_pit_out_lap"].fillna(False).astype(bool)]
    keep = [(int(r.driver_number), round(float(r.lap_duration), 3)) not in deleted for r in laps.itertuples()]
    laps = laps[np.asarray(keep, bool)]
    if laps.empty:
        raise LookupError("no valid timed lap")
    return laps.loc[laps["lap_duration"].idxmin()]


def reference_lap(key: int, pos_unit_m: float) -> tuple[pd.DataFrame, np.ndarray | None]:
    """The fastest valid lap of a session as car data, and its elevation along distance if available."""
    laps = pd.DataFrame(get("laps", ("session_key", key)))
    lap = fastest_valid_lap(laps, deleted_laps(get("race_control", ("session_key", key))))
    start = _utc([lap["date_start"]]).iloc[0]
    end = start + pd.to_timedelta(lap["lap_duration"], unit="s")
    pad = pd.Timedelta(2, unit="s")
    window = (("date>", (start - pad).isoformat()), ("date<", (end + pad).isoformat()))
    number = int(lap["driver_number"])
    df = slice_lap(_car(key, number, *window), start, end)
    if df is None:
        raise LookupError("no car data for the reference lap")
    loc = pd.DataFrame(get("location", ("session_key", key), ("driver_number", number), *window))
    elevation = None
    if not loc.empty and "z" in loc:
        t = (_utc(loc["date"]) - start).dt.total_seconds().to_numpy()
        order = np.argsort(t)
        elevation = np.interp(df["time_s"].to_numpy(), t[order], loc["z"].to_numpy(dtype=float)[order]) * pos_unit_m
    return df, elevation
