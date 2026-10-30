import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .track import Track, track_from_reference


POS_UNIT_M = 0.1


def source() -> str:
    """Timing-data source: "fastf1" (default) or "openf1" (see erslens.openf1)."""
    value = os.environ.get("ERSLENS_SOURCE", "fastf1").lower()
    if value not in ("fastf1", "openf1"):
        raise ValueError(f"ERSLENS_SOURCE must be fastf1 or openf1, not {value!r}")
    return value


def _openf1_key(year: int, event: str | int, kind: str) -> int:
    from . import openf1

    return openf1.session_key(year, event_info(year, event)[1], kind)


@lru_cache(maxsize=4)
def _session(year: int, event: str | int, kind: str, cache_dir: str | Path):
    import fastf1

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    session = fastf1.get_session(year, event, kind)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def _car_data(lap) -> pd.DataFrame:
    tel = lap.get_car_data().add_distance()
    return pd.DataFrame({
        "time_s": tel["Time"].dt.total_seconds().to_numpy(),
        "distance_m": tel["Distance"].to_numpy(),
        "speed_kmh": tel["Speed"].to_numpy(dtype=float),
        "throttle": tel["Throttle"].to_numpy(dtype=float),
        "brake": tel["Brake"].to_numpy(dtype=bool),
        "gear": tel["nGear"].to_numpy(),
        "rpm": tel["RPM"].to_numpy(dtype=float),
    })


def load_driver_race(year: int, event: str | int, driver: str, kind: str = "R",
                     cache_dir: str | Path = "data/cache") -> pd.DataFrame:
    """All of one driver's laps as a single telemetry table, keyed by lap number."""
    if source() == "openf1":
        from . import openf1

        return openf1.driver_race(_openf1_key(year, event, kind), driver)
    session = _session(year, event, kind, cache_dir)
    laps = session.laps.pick_drivers(driver)
    frames = []
    for _, lap in laps.iterlaps():
        try:
            df = _car_data(lap)
        except Exception:
            continue  # FastF1 raises on laps with missing car data (e.g. red flags)
        df.insert(0, "lap", int(lap["LapNumber"]))
        df["compound"] = lap["Compound"]
        df["pit_in"] = not pd.isna(lap["PitInTime"])
        df["pit_out"] = not pd.isna(lap["PitOutTime"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def reference_track(year: int, event: str | int, driver: str | None = None,
                    kind: str = "Q", cache_dir: str | Path = "data/cache",
                    ds: float = 5.0) -> Track:
    """Build a Track from a fast lap (qualifying by default: least traffic and lift-and-coast)."""
    name = f"{year}_{str(event).lower().replace(' ', '_')}"
    if source() == "openf1":
        if driver is not None:
            raise NotImplementedError("a named reference driver is only supported with FastF1")
        from . import openf1

        df, elevation = openf1.reference_lap(_openf1_key(year, event, kind), POS_UNIT_M)
        track = track_from_reference(name, df["distance_m"].to_numpy(), df["speed_kmh"].to_numpy(),
                                     df["throttle"].to_numpy(), ds=ds)
        track.elevation_m = None if elevation is None else np.interp(track.distance_m, df["distance_m"].to_numpy(), elevation)
        return track
    session = _session(year, event, kind, cache_dir)
    laps = session.laps if driver is None else session.laps.pick_drivers(driver)
    lap = laps.pick_quicklaps().pick_fastest()
    df = _car_data(lap)
    track = track_from_reference(name, df["distance_m"].to_numpy(), df["speed_kmh"].to_numpy(),
                                 df["throttle"].to_numpy(), ds=ds)
    try:
        tel = lap.get_telemetry()
        # FastF1 position coordinates are in tenths of a metre.
        track.elevation_m = np.interp(track.distance_m, tel["Distance"].to_numpy(),
                                      tel["Z"].to_numpy(dtype=float) * POS_UNIT_M)
    except Exception:
        track.elevation_m = None
    return track


def save_track(track: Track, path: str | Path) -> None:
    extra = {} if track.elevation_m is None else {"elevation_m": track.elevation_m}
    np.savez(path, name=track.name, distance_m=track.distance_m,
             v_limit_ms=track.v_limit_ms, straight_mode=track.straight_mode, **extra)


def load_track(path: str | Path) -> Track:
    z = np.load(path)
    elev = z["elevation_m"] if "elevation_m" in z.files else None
    return Track(str(z["name"]), z["distance_m"], z["v_limit_ms"], z["straight_mode"], elev)


def race_drivers(year: int, event: str | int, kind: str = "R",
                 cache_dir: str | Path = "data/cache") -> list[str]:
    if source() == "openf1":
        from . import openf1

        return [d for d in openf1.drivers(_openf1_key(year, event, kind))["name_acronym"] if isinstance(d, str)]
    session = _session(year, event, kind, cache_dir)
    return [d for d in session.results["Abbreviation"] if isinstance(d, str)]


def event_info(year: int, event: str | int) -> tuple[str, pd.Timestamp]:
    import fastf1

    ev = fastf1.get_event(year, event)
    return str(ev["EventName"]), pd.Timestamp(ev["EventDate"])


def driver_teams(year: int, event: str | int, kind: str = "R",
                 cache_dir: str | Path = "data/cache") -> dict[str, str]:
    if source() == "openf1":
        from . import openf1

        d = openf1.drivers(_openf1_key(year, event, kind))
        return d.set_index("name_acronym")["team_name"].to_dict()
    session = _session(year, event, kind, cache_dir)
    return session.results.set_index("Abbreviation")["TeamName"].to_dict()
