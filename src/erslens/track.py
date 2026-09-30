from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    name: str
    distance_m: np.ndarray
    v_limit_ms: np.ndarray      # inf where the car is power-limited
    straight_mode: np.ndarray   # active-aero low-drag zones

    @property
    def ds(self) -> float:
        return float(self.distance_m[1] - self.distance_m[0])

    @property
    def n(self) -> int:
        return len(self.distance_m)

    @property
    def length_m(self) -> float:
        return self.n * self.ds


def _long_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = np.zeros_like(mask)
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def track_from_reference(
    name: str,
    distance_m: np.ndarray,
    speed_kmh: np.ndarray,
    throttle_pct: np.ndarray,
    ds: float = 5.0,
    full_throttle_pct: float = 98.0,
    min_straight_m: float = 250.0,
) -> Track:
    grid = np.arange(0.0, float(distance_m[-1]), ds)
    v = np.interp(grid, distance_m, speed_kmh) / 3.6
    thr = np.interp(grid, distance_m, throttle_pct)
    flat_out = thr >= full_throttle_pct
    v_limit = np.where(flat_out, np.inf, v)
    straight = _long_runs(flat_out, int(min_straight_m / ds))
    return Track(name, grid, v_limit, straight)


def toy_track(segments: list[tuple[str, float, float | None]], ds: float = 5.0,
              name: str = "toy", min_straight_m: float = 250.0) -> Track:
    """segments: ("straight", length_m, None) or ("corner", length_m, speed_kmh)."""
    limits, straight = [], []
    for kind, length, speed in segments:
        k = int(round(length / ds))
        if kind == "straight":
            limits.append(np.full(k, np.inf))
            straight.append(np.full(k, length >= min_straight_m))
        else:
            limits.append(np.full(k, speed / 3.6))
            straight.append(np.zeros(k, dtype=bool))
    v_limit = np.concatenate(limits)
    return Track(name, np.arange(len(v_limit)) * ds, v_limit, np.concatenate(straight))


def demo_circuit() -> Track:
    return toy_track([
        ("straight", 1100, None), ("corner", 90, 95),
        ("straight", 450, None), ("corner", 150, 180),
        ("straight", 700, None), ("corner", 120, 120),
        ("straight", 300, None), ("corner", 200, 230),
        ("straight", 900, None), ("corner", 100, 80),
        ("straight", 250, None), ("corner", 160, 150),
    ], name="demo")
