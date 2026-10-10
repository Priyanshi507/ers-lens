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


def race_rules(car: CarParams, event: str, path: str = "configs/rules_2026_races.yaml") -> CarParams:
    """The car with an event's 2026 race energy rules applied (see the rules file for sources)."""
    import yaml

    with open(path) as f:
        rules = yaml.safe_load(f)["events"]
    if event not in rules:
        raise KeyError(f"no 2026 race rules recorded for {event}; add them with a source")
    return car.with_(**{k: float(v) for k, v in rules[event].items()})


def rules_sensitivity(path: str = "configs/rules_2026_races.yaml") -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)["sensitivity"]


# Pre-registered rule (research log, 2026-10-07): an M1 result whose own DP prediction differs
# from its simulator replay by more than this is not used in any verdict.
CONVERGED_S = 0.05
REFINED_TAG = "|refined"
NOT_M1 = ("m2", "simple_")
# Finer grids tried, in order, for a result that fails the rule; each stays within ~4 GB.
REFINE_LADDER = ((0.125, 321), (0.125, 641), (0.0625, 321))
GRID = {"fine": (0.125, 321), "medium": (0.25, 161)}


def original_grid(variant: str) -> tuple[float, int]:
    return GRID["medium"] if variant == "sweep" or variant.startswith("soc0_") else GRID["fine"]


def variant_setup(variant: str, car_id: str, event: str, car: CarParams) -> tuple[CarParams, float]:
    """The car and starting charge behind a row of runs.csv, so any result can be re-solved."""
    base_id = car_id.split(REFINED_TAG)[0]
    if variant == "sweep":
        return dict(legal_sweep(car))[base_id], 0.5
    if variant.startswith("soc0_"):
        return car, float(variant.removeprefix("soc0_"))
    if variant == "nominal":
        return car, 0.5
    if variant == "no_taper":
        return no_taper(car), 0.5
    if variant.startswith("rules"):
        rc = race_rules(car, event)
        sens = rules_sensitivity()
        return {"rules": rc, "rules_no_taper": no_taper(rc),
                "rules_h8": rc.with_(harvest_per_lap_j=float(sens["harvest_low_j"])),
                "rules_deploy250": rc.with_(deploy_max_w=float(sens["deploy_outside_zones_w"])),
                }[variant], 0.5
    raise ValueError(f"no M1 setup for variant {variant!r}")


def resolve_runs(runs):
    """Rows each verdict may use, and the M1 cases excluded because nothing converged.

    For every (event, variant, car) case, the last converged row is used, so a refined re-solve
    appended later replaces an unconverged original; rows are never edited. M2 and simple-policy
    rows carry no DP prediction and pass through unchanged.
    """
    import pandas as pd

    runs = runs.copy()
    runs["base_id"] = runs.car_id.str.split(REFINED_TAG, regex=False).str[0]
    other = runs.variant.str.startswith(NOT_M1)
    m1 = runs[~other].copy()
    m1["converged"] = (m1.dp_pred_s - m1.lap_s).abs() <= CONVERGED_S
    m1["refined"] = m1.car_id.str.contains(REFINED_TAG, regex=False)
    used, excluded = [], []
    for _, case in m1.groupby(["event", "variant", "base_id"], sort=False):
        ok = case[case.converged]
        (used if len(ok) else excluded).append(ok.tail(1) if len(ok) else case.tail(1))
    cols = list(runs.columns)
    used = pd.concat(used + [runs[other].assign(converged=True, refined=False)], ignore_index=True)
    excluded = pd.concat(excluded, ignore_index=True) if excluded else m1.iloc[0:0]
    for df in (used, excluded):
        df["car_id"] = df["base_id"]
    return used[cols + ["converged", "refined"]].drop(columns="base_id"), excluded
