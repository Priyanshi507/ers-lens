import numpy as np

from .params import CarParams
from .track import Track

G = 9.81


def resistive_force(v: float, mass: float, cda: float, car: CarParams) -> float:
    return 0.5 * car.rho * cda * v * v + car.crr * mass * G


def mguk_cap_kmh(kmh, car: CarParams, xp=np):
    """Available MGU-K power (W) at speed kmh; works on scalars and NumPy or JAX arrays."""
    low = xp.minimum(car.mguk_power_w, car.taper_a1_w - car.taper_b1_w_per_kmh * kmh)
    cap = xp.where(kmh >= car.taper_knee_kmh, car.taper_a2_w - car.taper_b2_w_per_kmh * kmh, low)
    return xp.maximum(cap, 0.0)


def mguk_power_cap(v: float, car: CarParams) -> float:
    return float(mguk_cap_kmh(v * 3.6, car))


def brake_decel(v: float, car: CarParams) -> float:
    # Downforce makes peak braking grow roughly with v^2.
    ratio = min(1.0, (v / car.brake_v_ref_ms) ** 2)
    return G * (car.brake_g_low + (car.brake_g_high - car.brake_g_low) * ratio)


def braking_envelope(track: Track, car: CarParams) -> np.ndarray:
    """Max speed at each point that still lets the car make every corner ahead.

    Swept twice around the loop so corners near the start/finish line
    constrain the end of the lap.
    """
    v = track.v_limit_ms.copy()
    n, ds = track.n, track.ds
    for _ in range(2):
        for k in range(n - 1, -1, -1):
            nxt = v[(k + 1) % n]
            if np.isfinite(nxt):
                v[k] = min(v[k], np.sqrt(nxt * nxt + 2.0 * brake_decel(nxt, car) * ds))
    return v
