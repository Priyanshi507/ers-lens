from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CarParams:
    mass_kg: float = 768.0
    fuel_kg: float = 70.0
    fuel_per_lap_kg: float = 1.3
    cda_straight: float = 0.95
    cda_corner: float = 1.35
    crr: float = 0.012
    rho: float = 1.20
    ice_power_w: float = 400e3
    mguk_power_w: float = 350e3
    # Reported regulation formula (VERIFY against FIA text): available MGU-K power is
    # a1 - b1*v below the knee speed and a2 - b2*v above it, capped at mguk_power_w.
    taper_a1_w: float = 1.8e6
    taper_b1_w_per_kmh: float = 5e3
    taper_knee_kmh: float = 340.0
    taper_a2_w: float = 6.9e6
    taper_b2_w_per_kmh: float = 2e4
    # Reported FIA limit on how fast deployed power may be reduced (VERIFY).
    mguk_ramp_w_per_s: float = 5e4
    es_capacity_j: float = 4.0e6
    harvest_per_lap_j: float = 8.5e6
    eta_harvest: float = 0.90
    eta_deploy: float = 0.95
    traction_max_n: float = 14000.0
    brake_g_low: float = 1.8
    brake_g_high: float = 5.0
    brake_v_ref_ms: float = 83.0

    def with_(self, **changes) -> "CarParams":
        return replace(self, **changes)


def load_car_params(path: str | Path) -> CarParams:
    with open(path) as f:
        return CarParams(**yaml.safe_load(f)["car"])
