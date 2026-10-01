from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .track import Track, track_from_reference


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
    session = _session(year, event, kind, cache_dir)
    laps = session.laps if driver is None else session.laps.pick_drivers(driver)
    lap = laps.pick_quicklaps().pick_fastest()
    df = _car_data(lap)
    name = f"{year}_{str(event).lower().replace(' ', '_')}"
    return track_from_reference(name, df["distance_m"].to_numpy(), df["speed_kmh"].to_numpy(),
                                df["throttle"].to_numpy(), ds=ds)


def save_track(track: Track, path: str | Path) -> None:
    np.savez(path, name=track.name, distance_m=track.distance_m,
             v_limit_ms=track.v_limit_ms, straight_mode=track.straight_mode)


def load_track(path: str | Path) -> Track:
    z = np.load(path)
    return Track(str(z["name"]), z["distance_m"], z["v_limit_ms"], z["straight_mode"])


def race_drivers(year: int, event: str | int, kind: str = "R",
                 cache_dir: str | Path = "data/cache") -> list[str]:
    session = _session(year, event, kind, cache_dir)
    return [d for d in session.results["Abbreviation"] if isinstance(d, str)]


def event_info(year: int, event: str | int) -> tuple[str, pd.Timestamp]:
    import fastf1

    ev = fastf1.get_event(year, event)
    return str(ev["EventName"]), pd.Timestamp(ev["EventDate"])


def driver_teams(year: int, event: str | int, kind: str = "R",
                 cache_dir: str | Path = "data/cache") -> dict[str, str]:
    session = _session(year, event, kind, cache_dir)
    return session.results.set_index("Abbreviation")["TeamName"].to_dict()
