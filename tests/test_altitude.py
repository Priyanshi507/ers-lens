import numpy as np

from erslens.params import load_car_params
from erslens.strategy import altitude_density_ratio, at_altitude
from erslens.track import demo_circuit

CAR = load_car_params("configs/car_2026.yaml")


def test_density_ratio_matches_the_standard_atmosphere():
    assert altitude_density_ratio(0.0) == 1.0
    assert abs(altitude_density_ratio(2227.0) - 0.803) < 0.001
    assert abs(altitude_density_ratio(747.0) - 0.930) < 0.001


def test_car_is_scaled_only_in_density():
    track = demo_circuit()
    track.elevation_m = np.full(len(track.distance_m), 2000.0)
    high = at_altitude(CAR, track)
    assert high.rho < CAR.rho and high.with_(rho=CAR.rho) == CAR


def test_no_elevation_leaves_the_car_unchanged():
    track = demo_circuit()
    track.elevation_m = None
    assert at_altitude(CAR, track) == CAR
