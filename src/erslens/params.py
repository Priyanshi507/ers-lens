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
    mguk_taper_start_kmh: float = 290.0
    mguk_taper_end_kmh: float = 355.0
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
