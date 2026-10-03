"""Time lost to clipping on a straight, with bounds over unknown car parameters.

Before the speed peak the car accelerates at full throttle with its pre-clip power, so the
telemetry pins down its net driving force at the speeds it reached. The counterfactual
(no clipping until the braking point) keeps accelerating above the observed peak, which
requires extrapolating that force. How the force splits between engine power and drag is
not identifiable, so every plausible split is tried and the range of time loss reported.

Pre-clip electric power is modelled as d * cap(v), a fixed fraction of the speed-dependent
regulation limit; d is fitted for each (ICE power, drag area) combination.
"""
import numpy as np
import pandas as pd

from .energy import resample_uniform
from .throttle import full_throttle_mask, run_around
from .params import CarParams
from .physics import G, mguk_cap_kmh

WINDOW = 8            # 2 s windows at 4 Hz for the energy balance
MIN_FIT_KMH = 210.0   # below this the car can be traction-limited
MIN_GAIN_KMH = 3.0    # smaller speed losses are within telemetry noise
FIT_TOLERANCE = 1.25  # admissible fits: rms residual within 25% of the best
STEP_M = 1.0
# Clipping starts before the speed peak, so the fit stops this far below peak speed to
# keep partly-clipped running out of the pre-clip model.
PEAK_GUARD_KMH = 3.0
# 90th-percentile absolute error on simulated laps with 4 Hz rounded, noisy telemetry
# (scripts/laptime_validate.py); added to the parameter range to form each lap's interval.
SIM_ERROR_90_S = 0.09


def _flat_run(lap: pd.DataFrame, start_m: float, end_m: float) -> pd.DataFrame | None:
    """The full-throttle run containing the speed peak, as the Phase 2 detector defines it:
    shared noise bridging, and a window reaching 100 m past the straight to the braking point."""
    raw = lap[lap["distance_m"].between(start_m, end_m + 100)].reset_index(drop=True)
    if len(raw) < 5:
        return None
    flat_raw = full_throttle_mask(raw["throttle"].to_numpy(), raw["brake"].to_numpy())
    if not flat_raw.any():
        return None
    u = resample_uniform(raw)
    nearest = np.clip(np.searchsorted(raw["time_s"].to_numpy(), u["time_s"].to_numpy()), 0, len(raw) - 1)
    flat = flat_raw[nearest]
    if not flat.any():
        return None
    peak = int(np.argmax(np.where(flat, u["v"].to_numpy(), -np.inf)))
    a, b = run_around(flat, peak)
    return u.iloc[a:b + 1].reset_index(drop=True)


def _fit_range(v: np.ndarray, peak: int) -> tuple[int, int]:
    start = int(np.argmax(v * 3.6 >= MIN_FIT_KMH))
    guard = v[peak] - PEAK_GUARD_KMH / 3.6
    stop = start
    while stop + 1 < peak and v[stop + 1] <= guard:
        stop += 1
    return start, stop


def _windows(r: pd.DataFrame, start: int, stop: int, mass: float, car: CarParams, z: np.ndarray):
    v, t = r["v"].to_numpy(), r["time_s"].to_numpy()
    cap = mguk_cap_kmh(v * 3.6, car)
    rows = []
    for a in range(start, stop - WINDOW + 1, WINDOW):
        b = a + WINDOW
        seg = slice(a, b + 1)
        rows.append((
            0.5 * mass * (v[b] ** 2 - v[a] ** 2) + mass * G * (z[b] - z[a])
            + car.crr * mass * G * np.trapezoid(v[seg], t[seg]),
            t[b] - t[a],
            np.trapezoid(0.5 * car.rho * v[seg] ** 3, t[seg]),
            np.trapezoid(cap[seg], t[seg]),
        ))
    return np.array(rows)


def _counterfactual_time(v0, s0, s1, ice, d, cda, mass, car, grade_fn):
    s, v, t = s0, v0, 0.0
    while s < s1:
        p = ice + d * mguk_cap_kmh(v * 3.6, car)
        f = min(p / v, car.traction_max_n)
        a = (f - 0.5 * car.rho * cda * v * v - car.crr * mass * G - mass * G * grade_fn(s)) / mass
        v_next = np.sqrt(max(v * v + 2 * a * STEP_M, 25.0))
        t += 2 * STEP_M / (v + v_next)
        v, s = v_next, s + STEP_M
    return t


def clip_time_loss(lap: pd.DataFrame, start_m: float, end_m: float, mass: float, car: CarParams,
                   elevation: tuple[np.ndarray, np.ndarray] | None = None,
                   ice_range: float = 0.10, cda_range: float = 0.20, mass_range_kg: float = 35.0,
                   n_grid: int = 17) -> dict | None:
    """Seconds lost to clipping before the end of full throttle, with bounds over car parameters.

    The counterfactual starts where the pre-clip fit starts and runs to the end of full
    throttle; before clipping begins it tracks the observed car, so the loss accumulates
    from wherever clipping actually starts, including before the speed peak.
    """
    r = _flat_run(lap, start_m, end_m)
    if r is None or len(r) < 2 * WINDOW + 2:
        return None
    v, d_m, t = r["v"].to_numpy(), r["distance_m"].to_numpy(), r["time_s"].to_numpy()
    peak = int(np.argmax(v))
    if (v[peak] - v[-1]) * 3.6 < MIN_GAIN_KMH:
        return {"t_loss_s": 0.0, "t_low_s": 0.0, "t_high_s": 0.0, "clipping": False}
    start, stop = _fit_range(v, peak)
    if stop - start < 2 * WINDOW:
        return None

    if elevation is None:
        z = np.zeros_like(d_m)
        grade_fn = lambda s: 0.0  # noqa: E731
    else:
        z = np.interp(d_m, *elevation)
        grade_fn = lambda s: (np.interp(s + 1, *elevation) - np.interp(s - 1, *elevation)) / 2  # noqa: E731
    obs_time = t[-1] - t[start]

    fits = []
    for m in mass + mass_range_kg * np.linspace(-1, 1, 3):
        w = _windows(r, start, stop, m, car, z)
        if len(w) < 2:
            return None
        lhs, T, drag_x, cap_x = w.T
        for ice in car.ice_power_w * (1 + ice_range * np.linspace(-1, 1, n_grid)):
            for cda in car.cda_straight * (1 + cda_range * np.linspace(-1, 1, n_grid)):
                # Energy balance per window: lhs = ice*T - cda*drag_x + d*cap_x, solved for d.
                y = lhs - ice * T + cda * drag_x
                d = float(np.dot(cap_x, y) / np.dot(cap_x, cap_x))
                rms = float(np.sqrt(np.mean((y - d * cap_x) ** 2)))
                fits.append((m, ice, cda, d, rms))
    best_rms = min(f[4] for f in fits)
    admissible = [f for f in fits if 0.0 <= f[3] <= 1.0 and f[4] <= FIT_TOLERANCE * best_rms]
    if not admissible:
        return None

    def loss(f):
        m, ice, cda, d, _ = f
        return obs_time - _counterfactual_time(v[start], d_m[start], d_m[-1], ice, d, cda, m, car, grade_fn)

    losses = [loss(f) for f in admissible]
    nominal = min(admissible, key=lambda f: f[4])
    return {"t_loss_s": float(loss(nominal)), "t_low_s": float(min(losses)), "t_high_s": float(max(losses)),
            "interval_low_s": float(min(losses) - SIM_ERROR_90_S),
            "interval_high_s": float(max(losses) + SIM_ERROR_90_S),
            "clipping": True, "n_admissible": len(admissible), "deploy_frac_nominal": nominal[3],
            "peak_kmh": float(v[peak] * 3.6), "span_from_m": float(d_m[start]),
            "span_to_m": float(d_m[-1])}
