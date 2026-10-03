"""One definition of full-throttle running, shared across the project."""
import numpy as np

FULL_THROTTLE = 98.0
NOISE_THROTTLE = 90.0


def full_throttle_mask(throttle: np.ndarray, brake: np.ndarray) -> np.ndarray:
    """Full throttle and no brake, bridging single samples just below full throttle.

    A single sample between NOISE_THROTTLE and FULL_THROTTLE between two full-throttle
    samples is sensor noise, not a lift; left unbridged it would hide a real speed loss.
    """
    no_brake = ~np.asarray(brake, dtype=bool)
    thr = np.asarray(throttle, dtype=float)
    flat = (thr >= FULL_THROTTLE) & no_brake
    blip = np.zeros_like(flat)
    blip[1:-1] = (~flat[1:-1] & (thr[1:-1] >= NOISE_THROTTLE) & no_brake[1:-1]
                  & flat[:-2] & flat[2:])
    return flat | blip


def run_around(flat: np.ndarray, i: int) -> tuple[int, int]:
    """First and last index of the unbroken True run containing index i."""
    a = b = i
    while a > 0 and flat[a - 1]:
        a -= 1
    while b + 1 < len(flat) and flat[b + 1]:
        b += 1
    return a, b
