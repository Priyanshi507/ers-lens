"""Electric power swing during clipping, measured from public telemetry alone.

On a clipping lap the car passes the same speed twice at full throttle: accelerating
before the speed peak and decelerating after it. At equal speed, drag, rolling
resistance and engine power are identical, so the difference in net power between the
two passes is purely electric: lost MGU-K deployment plus any harvesting. Over a speed
band [lo, hi] the energy balance gives

    swing = dKE * (1 / T_up + 1 / T_down),   dKE = m * (hi^2 - lo^2) / 2

which needs only the car mass, not its drag or engine power. Gravity cancels only if
both passes climb equally, so on hilly straights the elevation change of each pass is
added to the energy balance (track elevation profile from FastF1 position data). Fitting those per car from
speed alone is not identifiable: high drag with full deployment produces the same trace
as low drag with partial deployment (see docs/research_log.md).
"""
import numpy as np
import pandas as pd

from .physics import G

DT = 0.25
FULL_THROTTLE = 98.0
PEAK_MARGIN_KMH = 1.5
MIN_BAND_KMH = 4.0


def resample_uniform(lap: pd.DataFrame, dt: float = DT) -> pd.DataFrame:
    """Uniform time grid; FastF1 car data arrives at irregular ~4 Hz intervals."""
    t = lap["time_s"].to_numpy()
    grid = np.arange(t[0], t[-1], dt)
    nearest = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    return pd.DataFrame({
        "time_s": grid,
        "distance_m": np.interp(grid, t, lap["distance_m"]),
        "v": np.interp(grid, t, lap["speed_kmh"]) / 3.6,
        "throttle": lap["throttle"].to_numpy()[nearest],
        "brake": lap["brake"].astype(bool).to_numpy()[nearest],
    })


def _crossing_time(t: np.ndarray, v: np.ndarray, level: float, rising: bool) -> float:
    hit = v >= level if rising else v <= level
    i = int(np.argmax(hit))
    if not hit[i] or i == 0:
        return np.nan
    return float(np.interp(level, *((v[i - 1:i + 1], t[i - 1:i + 1]) if rising
                                    else (v[i - 1:i + 1][::-1], t[i - 1:i + 1][::-1]))))


def electric_power_swing(lap: pd.DataFrame, start_m: float, end_m: float, mass_kg: float,
                         elevation: tuple[np.ndarray, np.ndarray] | None = None) -> dict | None:
    """Drop in electric power (W) after the speed peak, or None if the lap barely clips.

    elevation is (distance_m, height_m) for the circuit; without it the straight is
    assumed level.
    """
    u = resample_uniform(lap)
    u = u[u["distance_m"].between(start_m, end_m)
          & (u["throttle"] >= FULL_THROTTLE) & ~u["brake"]]
    if len(u) < 10:
        return None
    t, v, d = u["time_s"].to_numpy(), u["v"].to_numpy(), u["distance_m"].to_numpy()
    peak = int(np.argmax(v))
    # Running max/min make each branch monotone so noise cannot create extra crossings.
    tu, vu = t[:peak + 1], np.maximum.accumulate(v[:peak + 1])
    td, vd = t[peak:], np.minimum.accumulate(v[peak:])
    lo = max(vu[0], vd[-1]) + PEAK_MARGIN_KMH / 3.6
    hi = v[peak] - PEAK_MARGIN_KMH / 3.6
    if hi - lo < MIN_BAND_KMH / 3.6:
        return None
    up_lo, up_hi = _crossing_time(tu, vu, lo, True), _crossing_time(tu, vu, hi, True)
    dn_hi, dn_lo = _crossing_time(td, vd, hi, False), _crossing_time(td, vd, lo, False)
    t_up, t_down = up_hi - up_lo, dn_lo - dn_hi
    if not (t_up > 0 and t_down > 0):
        return None
    dke = 0.5 * mass_kg * (hi ** 2 - lo ** 2)
    flat = dke * (1 / t_up + 1 / t_down)
    dz_up = dz_down = 0.0
    if elevation is not None:
        def z(tc):
            return float(np.interp(np.interp(tc, t, d), *elevation))
        dz_up, dz_down = z(up_hi) - z(up_lo), z(dn_lo) - z(dn_hi)
    mg = mass_kg * G
    swing = (dke + mg * dz_up) / t_up + (dke - mg * dz_down) / t_down
    return {"swing_w": swing, "swing_level_w": flat, "dz_up_m": dz_up, "dz_down_m": dz_down,
            "band_lo_kmh": lo * 3.6, "band_hi_kmh": hi * 3.6, "t_up_s": t_up, "t_down_s": t_down}


def true_swing(sim: pd.DataFrame, lap: int, start_m: float, end_m: float,
               band_kmh: tuple[float, float]) -> float:
    """Ground truth from the simulator: mean electric power before minus after the peak, in band."""
    x = sim[(sim["lap"] == lap) & sim["distance_m"].between(start_m, end_m)
            & (sim["phase"] == "accel")]
    v = x["speed_kmh"].to_numpy()
    peak = int(np.argmax(v))
    pe = (x["deploy_w"] - x["harvest_w"]).to_numpy()
    inb = (v >= band_kmh[0]) & (v <= band_kmh[1])
    idx = np.arange(len(v))
    up, down = inb & (idx < peak), inb & (idx > peak)
    return float(pe[up].mean() - pe[down].mean()) if up.any() and down.any() else np.nan
