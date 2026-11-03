"""The FIA's per-race 250 kW deployment sectors (configs/deploy_sectors_2026.yaml) on a track."""
import numpy as np
import yaml

from .track import Track

SECTOR_CAP_W = 250e3
PATH = "configs/deploy_sectors_2026.yaml"


def sector_events(path: str = PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["events"]


def sector_mask(track: Track, centreline_m: float, sectors) -> np.ndarray:
    """Points inside any window, with FIA lap distance scaled to the track file's length."""
    d = track.distance_m * (centreline_m / (track.n * track.ds))
    inside = np.zeros(track.n, bool)
    for a, b in sectors:
        inside |= (d >= a) & (d < b)
    return inside


def with_fia_sectors(track: Track, event: str, path: str = PATH) -> Track:
    e = sector_events(path)[event]
    cap = np.where(sector_mask(track, e["centreline_m"], e["sectors"]), SECTOR_CAP_W, np.inf)
    return Track(track.name, track.distance_m, track.v_limit_ms, track.straight_mode, track.elevation_m, cap)
