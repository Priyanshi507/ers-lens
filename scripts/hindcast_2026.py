"""Does the lap-time optimum rank circuits as real 2026 cars clipped? Y1-Y2 (research log, 2026-11-13).

    python scripts/hindcast_2026.py [--runs results/fia_fix_calendar_c]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from erslens.params import CarParams, load_car_params
from erslens.physics import braking_envelope
from erslens.track import Track
from fia_fix_figures import longest_free_run

BRAKES = load_car_params(Path(__file__).parents[1] / "configs/car_2026.yaml")  # every car brakes alike

FIRST_350 = pd.Timestamp("2026-05-03")  # Miami, first race with the 350 kW super-clip limit
NAMES = {"Australian Grand Prix": "Australia", "Chinese Grand Prix": "China", "Japanese Grand Prix": "Japan",
         "Miami Grand Prix": "Miami", "Canadian Grand Prix": "Canada", "Austrian Grand Prix": "Austria",
         "Belgian Grand Prix": "Belgium"}


def model_drop_kmh(profiles: Path, event: str, cap: int) -> float:
    return drop_from_profile(profiles / f"{event}_original_{cap}.npz")


def full_throttle(z, car: CarParams) -> np.ndarray:
    """True where the solver's car accelerates below its braking envelope, as jaxsim._step's accel;
    elsewhere it brakes, holds the envelope at part throttle, or lifts and coasts."""
    v_limit = z["v_limit"]
    env = braking_envelope(Track("", z["distance_m"], v_limit, np.zeros(len(v_limit), bool)), car)
    lifting = z["coast"] > 0.0 if "coast" in z.files else np.zeros(len(v_limit), bool)
    return (np.roll(z["v"], -1) < np.roll(env, -1) - 1e-6) & ~lifting


def drop_from_profile(path: Path, car: CarParams | None = None) -> float:
    """Speed lost between the peak of the lap's longest corner-free run and the end of full throttle,
    km/h, as the real laps are measured (drop_kmh_median). Braking is not counted (research log,
    2026-11-14: the first version ran to the end of the corner-free run and so included braking)."""
    z = np.load(path)
    i, j = longest_free_run(z["v_limit"])
    v = z["v"][i:j] * 3.6
    k = int(v.argmax())
    off = np.nonzero(~full_throttle(z, car or BRAKES)[i:j][k:])[0]
    return float(v[k] - v[k + off[0] if len(off) else -1])


def table(runs_dir: Path, observed: Path) -> pd.DataFrame:
    obs = pd.read_csv(observed)
    obs = obs[obs.used.astype(bool)]
    rows = []
    for r in obs.itertuples():
        event = NAMES.get(r.event, r.event)
        cap = 350 if pd.Timestamp(r.date) >= FIRST_350 else 250
        path = runs_dir / "profiles" / f"{event}_original_{cap}.npz"
        if not path.exists():
            continue
        rows.append({"event": event, "cap_kw": cap, "model_drop_kmh": model_drop_kmh(runs_dir / "profiles", event, cap),
                     "observed_drop_kmh": r.drop_kmh_median, "clip_share": r.clip_share})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/fia_fix_calendar_c")
    ap.add_argument("--observed", default="results/multi_rebased_mid/R/race_summary.csv")
    args = ap.parse_args()
    t = table(Path(args.runs), Path(args.observed))
    print(t.round(2).to_string(index=False))
    rho = spearmanr(t.model_drop_kmh, t.observed_drop_kmh).statistic
    agree = ((t.model_drop_kmh >= 3) == (t.clip_share >= 0.5))
    print(f"Y1: Spearman {rho:.2f} over {len(t)} races -> {'held' if rho >= 0.6 else 'NOT held'}")
    print(f"Y2: model clips where real laps clip at {int(agree.sum())} of {len(t)} -> "
          f"{'held' if agree.mean() >= 0.8 else 'NOT held'}")


if __name__ == "__main__":
    main()
