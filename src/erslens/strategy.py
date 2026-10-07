"""Measurements on a strategy, using the definitions fixed in the pre-registration (S1-S7)."""
import numpy as np

from .params import CarParams
from .track import Track


def no_taper(car: CarParams) -> CarParams:
    """S3 counterfactual: a flat MGU-K cap at all speeds, everything else unchanged."""
    return car.with_(taper_a1_w=1e12, taper_knee_kmh=1e9)


def legal_sweep(car: CarParams) -> list[tuple[str, CarParams]]:
    """The 27 cars of S2: ICE power +/-10%, straight-line drag area +/-20%, mass +/-35 kg."""
    cars = []
    for i, ice in enumerate((0.9, 1.0, 1.1)):
        for j, drag in enumerate((0.8, 1.0, 1.2)):
            for k, dm in enumerate((-35.0, 0.0, 35.0)):
                cars.append((f"i{i}d{j}m{k}", car.with_(ice_power_w=car.ice_power_w * ice,
                                                        cda_straight=car.cda_straight * drag,
                                                        mass_kg=car.mass_kg + dm)))
    return cars


def longest_full_throttle_run(track: Track) -> tuple[int, int]:
    """Index range [a, b) of the longest run of points with no speed limit."""
    free = np.isinf(track.v_limit_ms)
    best, a = (0, 0), None
    for i, f in enumerate(np.append(free, False)):
        if f and a is None:
            a = i
        elif not f and a is not None:
            if i - a > best[1] - best[0]:
                best = (a, i)
            a = None
    return best


def clip_metrics(track: Track, out: dict) -> dict:
    """Clipping on the longest straight (S1-S3), where energy is spent (S4), peak position (S7)."""
    a, b = longest_full_throttle_run(track)
    v = out["v"][a:b]
    net = (out["deploy"] - out["harvest"])[a:b]
    peak = int(np.argmax(v))
    pre = net[:peak].mean() if peak > 0 else np.nan
    free = np.isinf(track.v_limit_ms)
    energy = out["deploy"] * out["dt"]
    return {
        "clip_ratio": float(net[peak] / pre) if pre > 0 else np.nan,
        "peak_from_start_m": float(peak * track.ds),
        "run_length_m": float((b - a) * track.ds),
        "deploy_weighted_kmh": float((out["v"] * energy).sum() / energy.sum() * 3.6),
        "full_throttle_mean_kmh": float(out["v"][free].mean() * 3.6),
    }
