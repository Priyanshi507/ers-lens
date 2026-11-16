"""Each session's lap against its own real laps: predictions QL1-QL4 (research log, 2026-11-16).

    python scripts/quali_test.py [--push results/quali_push] [--neutral results/quali_neutral]
        [--race-model results/fia_fix_calendar_c] [--quali results/quali_real/Q/race_summary.csv]
        [--race results/multi_rebased_mid/R/race_summary.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hindcast_2026 import FIRST_350, NAMES, drop_from_profile

MIN_LAPS = 10
CONTRAST_KMH = 2.0


def table(push: Path, neutral: Path, race_model: Path, quali: Path, race: Path) -> pd.DataFrame:
    q = pd.read_csv(quali)
    q = q[q.clean_laps >= MIN_LAPS]
    r = pd.read_csv(race)
    r = r[r.used.astype(bool)].set_index("event")
    rows = []
    for o in q.itertuples():
        event = NAMES.get(o.event, o.event)
        cap = 350 if pd.Timestamp(o.date) >= FIRST_350 else 250
        name = f"{event}_original_{cap}.npz"
        if not (push / "profiles" / name).exists():
            continue
        rows.append({"event": event, "cap_kw": cap, "quali_laps": int(o.clean_laps),
                     "real_quali": o.drop_kmh_median, "quali_clip_share": o.clip_share,
                     "real_race": r.drop_kmh_median.get(o.event, np.nan),
                     "model_push": drop_from_profile(push / "profiles" / name),
                     "model_neutral": drop_from_profile(neutral / "profiles" / name),
                     "model_race": drop_from_profile(race_model / "profiles" / name)})
    return pd.DataFrame(rows)


def verdicts(t: pd.DataFrame) -> dict:
    rho_push = spearmanr(t.model_push, t.real_quali).statistic
    rho_neutral = spearmanr(t.model_neutral, t.real_quali).statistic
    clip = t[t.quali_clip_share >= 0.5]
    ratio = float((clip.model_push / clip.real_quali).median()) if len(clip) else np.nan
    c = t[(t.real_quali - t.real_race).abs() >= CONTRAST_KMH]
    same = int((np.sign(c.model_push - c.model_race) == np.sign(c.real_quali - c.real_race)).sum())
    return {"QL1": (rho_push >= 0.6, f"Spearman {rho_push:.2f} over {len(t)} races"),
            "QL2": (0.67 <= ratio <= 1.5, f"median ratio {ratio:.2f} over {len(clip)} races"),
            "QL3": (rho_push > rho_neutral, f"push {rho_push:.2f} against control {rho_neutral:.2f}"),
            "QL4": (len(c) > 0 and same >= 0.7 * len(c), f"sign right at {same} of {len(c)} races")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", default="results/quali_push")
    ap.add_argument("--neutral", default="results/quali_neutral")
    ap.add_argument("--race-model", default="results/fia_fix_calendar_c")
    ap.add_argument("--quali", default="results/quali_real/Q/race_summary.csv")
    ap.add_argument("--race", default="results/multi_rebased_mid/R/race_summary.csv")
    a = ap.parse_args()
    t = table(Path(a.push), Path(a.neutral), Path(a.race_model), Path(a.quali), Path(a.race))
    print(t.round(2).to_string(index=False))
    for name, (held, text) in verdicts(t).items():
        print(f"{name}: {text} -> {'held' if held else 'NOT held'}")


if __name__ == "__main__":
    main()
