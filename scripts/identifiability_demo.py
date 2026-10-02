"""Two physically different cars that produce identical speed telemetry.

On a full-throttle straight, m dv/dt = (P_ice + P_e)/v - 0.5 rho CdA v^2 - crr m g.
Replace P_ice -> P_ice + a, CdA -> CdA + delta and P_e(t) -> P_e(t) - a + 0.5 rho delta v(t)^3:
the right-hand side is unchanged, so the speed trace is identical, while the battery
energy used differs by the integral of (-a + 0.5 rho delta v^3) / eta. Electric power is
therefore identifiable from speed only up to a function of speed; differences taken at
matched speeds cancel any such function.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from erslens.energy import electric_power_swing
from erslens.params import load_car_params
from erslens.physics import G, mguk_cap_kmh

DT = 0.01


def drive(ice_w, cda, p_e_fn, car, mass, v0=60.0, t_end=16.0):
    """Integrate a full-throttle straight; p_e_fn(t, v) gives electric power at the wheels."""
    n = int(t_end / DT)
    t, s, v, p = (np.zeros(n) for _ in range(4))
    v[0] = v0
    for k in range(n - 1):
        p[k] = p_e_fn(t[k], v[k])
        a = ((ice_w + p[k]) / v[k] - 0.5 * car.rho * cda * v[k] ** 2 - car.crr * mass * G) / mass
        v[k + 1] = v[k] + a * DT
        s[k + 1] = s[k] + v[k] * DT
        t[k + 1] = t[k] + DT
    p[-1] = p_e_fn(t[-1], v[-1])
    return t, s, v, p


def battery_energy_mj(p, eta_deploy, eta_harvest):
    return float(np.sum(np.where(p > 0, p / eta_deploy, p * eta_harvest)) * DT / 1e6)


def as_telemetry(t, s, v, hz=4.0, round_kmh=True):
    tt = np.arange(0, t[-1], 1 / hz)
    kmh = np.interp(tt, t, v) * 3.6
    return pd.DataFrame({"time_s": tt, "distance_m": np.interp(tt, t, s),
                         "speed_kmh": np.round(kmh) if round_kmh else kmh,
                         "throttle": 100.0, "brake": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ice-change", type=float, default=0.10,
                    help="largest plausible relative change in ICE power")
    ap.add_argument("--max-cda-change", type=float, default=0.20,
                    help="largest plausible relative change in drag area")
    ap.add_argument("--out", default="results/identifiability")
    args = ap.parse_args()

    car = load_car_params("configs/car_2026.yaml")
    mass = car.mass_kg + car.fuel_kg / 2

    def p_a(t, v):  # car A: deploys 60% of the legal cap, then harvests 150 kW after 8 s
        return 0.6 * float(mguk_cap_kmh(v * 3.6, car)) if t < 8.0 else -150e3

    def legal(p, v):
        cap = mguk_cap_kmh(v * 3.6, car)
        return bool(np.all(p <= cap + 1.0) and np.all(p >= -car.mguk_power_w - 1.0))

    ta, sa, va, pa = drive(car.ice_power_w, car.cda_straight, p_a, car, mass)
    ea = battery_energy_mj(pa, car.eta_deploy, car.eta_harvest)
    assert legal(pa, va), "car A must respect the regulations"

    # Search plausible alternative cars whose deployment stays legal at every instant of
    # the identical speed trace: the spread of their battery use bounds what speed reveals.
    best = {}
    for a in np.linspace(-1, 1, 41) * args.max_ice_change * car.ice_power_w:
        for delta in np.linspace(-1, 1, 41) * args.max_cda_change * car.cda_straight:
            pb = pa - a + 0.5 * car.rho * delta * va ** 3
            if not legal(pb, va):
                continue
            eb = battery_energy_mj(pb, car.eta_deploy, car.eta_harvest)
            for key, better in (("max", lambda: eb > best["max"][0]), ("min", lambda: eb < best["min"][0])):
                if key not in best or better():
                    best[key] = (eb, a, delta)
    print(f"Car A: ICE {car.ice_power_w / 1e3:.0f} kW, drag area {car.cda_straight:.2f}, "
          f"battery energy {ea:.2f} MJ")
    print(f"Legal alternatives with identical speed (ICE +/-{args.max_ice_change:.0%}, "
          f"drag +/-{args.max_cda_change:.0%}): battery energy from {best['min'][0]:.2f} "
          f"to {best['max'][0]:.2f} MJ (spread {best['max'][0] - best['min'][0]:.2f} MJ)")

    eb, a, delta = best["max"]

    def p_b(t, v):
        return p_a(t, v) - a + 0.5 * car.rho * delta * v ** 3

    tb, sb, vb, pb = drive(car.ice_power_w + a, car.cda_straight + delta, p_b, car, mass)
    assert legal(pb, vb), "car B must respect the regulations"
    lap_a, lap_b = as_telemetry(ta, sa, va), as_telemetry(tb, sb, vb)
    swing_a = electric_power_swing(lap_a, 0, 1e6, mass)
    swing_b = electric_power_swing(lap_b, 0, 1e6, mass)

    print(f"Car B (max-energy legal alternative): ICE {(car.ice_power_w + a) / 1e3:.0f} kW, "
          f"drag area {car.cda_straight + delta:.2f}")
    print(f"Max speed difference over the straight: {np.abs(va - vb).max():.2e} m/s")
    print(f"Identical 4 Hz telemetry rows: {np.mean(lap_a.speed_kmh.values == lap_b.speed_kmh.values):.0%}")
    print(f"Battery energy used: A {ea:.2f} MJ vs B {eb:.2f} MJ (difference {eb - ea:+.2f} MJ)")
    print(f"Matched-speed swing estimate: A {swing_a['swing_w'] / 1e3:.1f} kW vs B {swing_b['swing_w'] / 1e3:.1f} kW")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(sa, va * 3.6, lw=3, label="Car A", color="tab:red")
    ax[0].plot(sb, vb * 3.6, lw=1.2, ls="--", label="Car B", color="k")
    ax[0].set_xlabel("distance (m)")
    ax[0].set_ylabel("speed (km/h)")
    ax[0].set_title("A. Speed: indistinguishable")
    ax[0].legend()
    ax[1].plot(sa, pa / 1e3, lw=2, label=f"Car A ({ea:.2f} MJ used)", color="tab:red")
    ax[1].plot(sb, pb / 1e3, lw=2, label=f"Car B ({eb:.2f} MJ used)", color="k")
    ax[1].set_xlabel("distance (m)")
    ax[1].set_ylabel("electric power at wheels (kW)")
    cap_line = mguk_cap_kmh(va * 3.6, car) / 1e3
    ax[1].plot(sa, cap_line, lw=1, ls=":", color="grey", label="regulation cap")
    ax[1].set_title("B. Hidden electric power: different, both legal")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "observational_equivalence.png", dpi=150)
    print(f"Saved {out}/observational_equivalence.png")


if __name__ == "__main__":
    main()
