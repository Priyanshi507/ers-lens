from pathlib import Path

import numpy as np
import pandas as pd

from .params import CarParams
from .policies import random_policy
from .simulate import simulate
from .track import Track

OBSERVED = ["time_s", "distance_m", "speed_kmh", "throttle", "brake"]
LABELS = ["soc_frac", "deploy_w", "harvest_w"]


def randomize_car(base: CarParams, rng: np.random.Generator) -> CarParams:
    def jitter(x, pct):
        return float(x * rng.uniform(1 - pct, 1 + pct))

    return base.with_(
        cda_straight=jitter(base.cda_straight, 0.10),
        cda_corner=jitter(base.cda_corner, 0.10),
        ice_power_w=jitter(base.ice_power_w, 0.05),
        eta_harvest=float(np.clip(jitter(base.eta_harvest, 0.05), 0.7, 0.97)),
        eta_deploy=float(np.clip(jitter(base.eta_deploy, 0.03), 0.8, 0.98)),
        fuel_kg=jitter(base.fuel_kg, 0.3),
    )


def sensor_model(sim: pd.DataFrame, car: CarParams, rng: np.random.Generator,
                 hz: float = 4.0) -> pd.DataFrame:
    """Resample a perfect simulation to look like the public F1 car-data feed."""
    t_end = sim.attrs.get("end_time_s", sim["time_s"].iloc[-1])
    t = np.arange(0.0, t_end, 1.0 / hz) + rng.uniform(0, 1.0 / hz)
    t = t[t < sim["time_s"].iloc[-1]]
    idx = np.searchsorted(sim["time_s"].to_numpy(), t, side="right") - 1

    def interp(col):
        return np.interp(t, sim["time_s"], sim[col])

    ds = sim["distance_m"].iloc[1] - sim["distance_m"].iloc[0]
    lap_offset = sim["lap"].to_numpy() * (sim["distance_m"].max() + ds)
    out = pd.DataFrame({
        "time_s": t,
        "lap": sim["lap"].to_numpy()[idx],
        "distance_m": np.interp(t, sim["time_s"], sim["distance_m"] + lap_offset),
        "speed_kmh": np.round(interp("speed_kmh") + rng.normal(0, 0.8, len(t))),
        "throttle": np.clip(np.round(interp("throttle") + rng.normal(0, 1.0, len(t))), 0, 100),
        "brake": sim["brake"].to_numpy()[idx],
        "label_soc_frac": interp("soc_j") / car.es_capacity_j,
        "label_deploy_w": sim["deploy_w"].to_numpy()[idx],
        "label_harvest_w": sim["harvest_w"].to_numpy()[idx],
    })
    return out


def generate_dataset(track: Track, base: CarParams, n_episodes: int, out_dir: str | Path,
                     laps_per_episode: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, meta = [], []
    for ep in range(n_episodes):
        car = randomize_car(base, rng)
        policy = random_policy(rng)
        soc0 = float(rng.uniform(0.1, 0.9))
        sim = simulate(track, car, policy, n_laps=laps_per_episode, soc0_frac=soc0)
        obs = sensor_model(sim, car, rng)
        obs.insert(0, "episode", ep)
        frames.append(obs)
        meta.append({"episode": ep, "policy": policy.name, "soc0": soc0,
                     **{k: getattr(car, k) for k in ("cda_straight", "ice_power_w", "eta_harvest")}})
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(out_dir / f"{track.name}_synthetic.parquet", index=False)
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(out_dir / f"{track.name}_episodes.csv", index=False)
    return data
