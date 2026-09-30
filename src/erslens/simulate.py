import numpy as np
import pandas as pd

from .params import CarParams
from .physics import braking_envelope, mguk_power_cap, resistive_force
from .policies import EnergyPolicy, StepContext
from .track import Track

V_MIN = 5.0


def simulate(track: Track, car: CarParams, policy: EnergyPolicy, n_laps: int = 1,
             soc0_frac: float = 0.5, fuel0_kg: float | None = None) -> pd.DataFrame:
    env = braking_envelope(track, car)
    progress = track.straight_progress
    ds, n = track.ds, track.n
    soc = soc0_frac * car.es_capacity_j
    fuel = car.fuel_kg if fuel0_kg is None else fuel0_kg
    v = float(min(env[0], 80.0))
    t = 0.0
    rows = []

    for lap in range(n_laps):
        harvested = 0.0
        mass = car.mass_kg + fuel
        for i in range(n):
            straight = bool(track.straight_mode[i])
            cda = car.cda_straight if straight else car.cda_corner
            v_eff = max(v, V_MIN)
            resist = resistive_force(v_eff, mass, cda, car)
            cap_k = mguk_power_cap(v_eff, car)
            budget = max(0.0, car.harvest_per_lap_j - harvested)
            space = car.es_capacity_j - soc

            ctx = StepContext(v_eff, soc / car.es_capacity_j, budget / car.harvest_per_lap_j,
                              straight, i / n, float(progress[i]))
            cmd = policy.command_w(ctx)
            dt_guess = ds / v_eff

            deploy = min(max(cmd, 0.0), cap_k, soc * car.eta_deploy / dt_guess)
            ice_harvest = 0.0
            if deploy == 0.0 and cmd < 0.0:
                ice_harvest = min(-cmd, cap_k, car.ice_power_w,
                                  budget / (car.eta_harvest * dt_guess),
                                  space / (car.eta_harvest * dt_guess))

            p_wheel = car.ice_power_w + deploy - ice_harvest
            f_drive = min(p_wheel / v_eff, car.traction_max_n)
            a = (f_drive - resist) / mass
            v_acc = np.sqrt(max(v * v + 2.0 * a * ds, V_MIN ** 2))
            target = env[(i + 1) % n]

            brake_harvest = 0.0
            if v_acc <= target:
                v_next, throttle, brake, phase = v_acc, 100.0, False, "accel"
            else:
                v_next = float(target)
                f_needed = mass * (v_next ** 2 - v ** 2) / (2.0 * ds) + resist
                if f_needed >= 0.0:
                    p_need = f_needed * v_eff
                    deploy = 0.0
                    ice_harvest = min(ice_harvest, max(0.0, car.ice_power_w - p_need))
                    throttle = 100.0 * min(1.0, p_need / car.ice_power_w)
                    brake, phase = False, "partial"
                else:
                    p_brake = -f_needed * v_eff
                    deploy = ice_harvest = 0.0
                    brake_harvest = min(p_brake, cap_k,
                                        budget / (car.eta_harvest * dt_guess),
                                        space / (car.eta_harvest * dt_guess))
                    throttle, brake, phase = 0.0, True, "brake"

            dt = 2.0 * ds / (v + v_next) if v + v_next > 0 else dt_guess
            e_in = car.eta_harvest * (ice_harvest + brake_harvest) * dt
            e_out = deploy / car.eta_deploy * dt
            soc = float(np.clip(soc + e_in - e_out, 0.0, car.es_capacity_j))
            harvested += e_in

            rows.append((lap, i, track.distance_m[i], t, v * 3.6, throttle, brake, straight,
                         phase, soc, deploy, ice_harvest + brake_harvest, harvested))
            t += dt
            v = v_next
        fuel = max(0.0, fuel - car.fuel_per_lap_kg)

    df = pd.DataFrame(rows, columns=[
        "lap", "idx", "distance_m", "time_s", "speed_kmh", "throttle", "brake",
        "straight_mode", "phase", "soc_j", "deploy_w", "harvest_w", "harvested_lap_j"])
    df.attrs["end_time_s"] = t
    return df


def lap_times(df: pd.DataFrame) -> pd.Series:
    starts = df.groupby("lap")["time_s"].first()
    ends = starts.shift(-1).fillna(df.attrs.get("end_time_s", df["time_s"].iloc[-1]))
    return ends - starts
