"""C5.2.8's speed taper limits only power used to propel the car; harvesting is limited by the absolute
MGU-K power, C5.2.7 (research log, 2026-11-04)."""
import jax.numpy as jnp

from erslens.jaxsim import _step, params_of
from erslens.params import CarParams
from erslens.physics import mguk_cap_kmh

CAR = CarParams(superclip_max_w=350e3)


def superclip_at(kmh, car):
    v = kmh / 3.6
    carry = (jnp.asarray(v), jnp.asarray(2e6), jnp.asarray(0.0))
    x = (jnp.asarray(-350e3), jnp.asarray(200.0), jnp.asarray(True))
    _, (_, _, _, harvest, _, superclip) = _step(params_of(car), car, 5.0, car.mass_kg, carry, x)
    return float(superclip)


def test_harvest_is_not_tapered_at_top_speed():
    assert float(mguk_cap_kmh(320.0, CAR)) == 200e3          # deployment would be limited to 200 kW here
    assert superclip_at(320.0, CAR) == 350e3


def test_the_old_tapered_harvest_is_kept_behind_a_flag():
    assert superclip_at(320.0, CAR.with_(harvest_taper=True)) == 200e3


def test_below_the_taper_nothing_changes():
    assert superclip_at(250.0, CAR) == superclip_at(250.0, CAR.with_(harvest_taper=True)) == 350e3
