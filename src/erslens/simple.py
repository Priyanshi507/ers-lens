"""Simple energy rules a strategy engineer could state in one sentence, tuned by exhaustive search.

Every family deploys on full-throttle running and harvests in grip-limited corners; they differ
in where deployment happens. All candidates are simulated in one batched call, and only
energy-neutral candidates (battery ends the lap at least as charged as it started) are eligible.
"""
from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from .jaxsim import params_of, simulate_laps, static_track
from .optimal import race_car, start_at_slowest_point
from .params import CarParams
from .track import Track

NEUTRAL_TOL_J = 1e3
DEPLOY_W = np.arange(150e3, 350e3 + 1, 25e3)
HARVEST_W = np.array([0.0, 75e3, 150e3, 250e3, 350e3])
CLIP_FRAC = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
CLIP_HARVEST_W = np.array([0.0, 150e3, 350e3])


@dataclass
class Tuned:
    family: str
    params: dict
    cmd_w: np.ndarray
    lap_time_s: float


def _profiles(track: Track) -> dict[str, list[tuple[dict, np.ndarray]]]:
    full = np.isinf(track.v_limit_ms)
    straight = track.straight_mode
    progress = track.straight_progress
    out = {"flat_out": [], "straights_only": [], "clip_end": []}
    for p, h in product(DEPLOY_W, HARVEST_W):
        out["flat_out"].append(({"deploy_w": p, "harvest_w": h}, np.where(full, p, -h)))
        out["straights_only"].append(({"deploy_w": p, "harvest_w": h}, np.where(straight, p, -h)))
        for f, c in product(CLIP_FRAC, CLIP_HARVEST_W):
            cmd = np.where(full, p, -h)
            cmd = np.where(straight & (progress >= 1.0 - f), -c, cmd)
            out["clip_end"].append(({"deploy_w": p, "harvest_w": h, "clip_frac": f, "clip_harvest_w": c}, cmd))
    return out


def tune(track: Track, car: CarParams, soc0_frac: float = 0.5) -> dict[str, Tuned]:
    car = race_car(car)
    track = start_at_slowest_point(track, car)
    st = static_track(track, car)
    p, v0 = params_of(car), float(st.env[-1])
    soc0 = soc0_frac * car.es_capacity_j

    def evaluate(cmd):
        out = simulate_laps(p, cmd, st, car, 1, soc0_frac, v0)
        o = {k: a[0] for k, a in out.items()}
        end = jnp.clip(o["soc"][-1] + (car.eta_harvest * o["harvest"][-1]
                                       - o["deploy"][-1] / car.eta_deploy) * o["dt"][-1],
                       0.0, car.es_capacity_j)
        return o["dt"].sum(), end

    batch = jax.jit(jax.vmap(evaluate))
    best = {}
    for family, cands in _profiles(track).items():
        cmds = jnp.asarray(np.stack([c for _, c in cands]))
        times, ends = (np.asarray(a) for a in batch(cmds))
        ok = ends >= soc0 - NEUTRAL_TOL_J
        if not ok.any():
            continue
        i = int(np.argmin(np.where(ok, times, np.inf)))
        best[family] = Tuned(family, cands[i][0], cands[i][1], float(times[i]))
    return best
