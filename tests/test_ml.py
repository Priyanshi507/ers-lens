import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("optax")

from erslens.gaussnet import evaluate, fit, predict
from erslens.params import CarParams
from erslens.synthstraight import MAX_ABS_ENERGY_MJ, MIN_SPEED_KMH, generate

CAR = CarParams()


def test_generator_is_deterministic_and_realistic():
    x1, y1 = generate(3, 300, "varied", "flexible", CAR)
    x2, y2 = generate(3, 300, "varied", "flexible", CAR)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert x1.shape == (300, 240)
    assert np.all(np.abs(y1) <= MAX_ABS_ENERGY_MJ)
    assert x1.min() >= MIN_SPEED_KMH
    assert np.all(x1 == np.round(x1))


def test_identical_cars_simple_strategy_energy_has_spread():
    _, y = generate(4, 500, "identical", "simple", CAR)
    assert y.std() > 0.2


def test_gaussian_network_recovers_known_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6000, 8))
    y = x[:, 0] + 0.3 * rng.normal(size=6000)
    f = fit(x[:5000], y[:5000], epochs=15)
    mean, std = predict(f, x[5000:])
    r = evaluate(y[5000:], mean, std)
    assert r["rmse_mj"] < 0.4
    assert 0.8 <= r["coverage_90"] <= 0.97
    assert 0.2 < r["mean_std_mj"] < 0.45


def test_car_ambiguity_is_positive_and_per_lap():
    from erslens.synthstraight import car_ambiguity_mj

    amb = car_ambiguity_mj(5, 300, "flexible", CAR)
    assert amb.shape == (300,)
    assert np.all(amb > 0)
