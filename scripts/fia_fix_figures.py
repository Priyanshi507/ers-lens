"""Figures for the FIA-fix experiment, from results/fia_fix (scripts/fia_fix.py).

    python scripts/fia_fix_figures.py
Writes docs/fia_fix_dose.png, docs/fia_fix_straight.png and results/fia_fix/location_split.csv
(where on the lap super-clipping energy is taken, under each cap).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVENTS = ("Australia", "China", "Miami", "Canada")
# Categorical slots 1-4 of the validated reference palette (light surface); two fall below 3:1
# contrast, so every line is also labelled directly and has its own marker.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dose(runs: pd.DataFrame, out: Path):
    sweep = runs[runs.car == "original"].sort_values("cap_kw")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), facecolor=SURFACE)
    panels = (("superclip_mj", "Energy harvested by super-clipping (MJ per lap)"),
              ("superclip_s", "Time spent super-clipping (s per lap)"))
    for ax, (col, title) in zip(axes, panels):
        style(ax)
        for e, color, marker in zip(EVENTS, SERIES, MARKERS):
            g = sweep[sweep.event == e]
            ax.plot(g.cap_kw, g[col], color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            ends = sweep[sweep.cap_kw == sweep.cap_kw.max()].set_index("event")[col]
            others = ends.drop(e)
            near = others[(others - ends[e]).abs() < 0.06 * ends.max()]
            nudge = 0 if near.empty else (7 if ends[e] > near.iloc[0] else -7)
            ax.annotate(e, (g.cap_kw.iloc[-1], g[col].iloc[-1]), xytext=(6, nudge), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        for cap, label in ((250, "before Miami"), (350, "from Miami")):
            ax.axvline(cap, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.annotate(label, (cap, 0.0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                        textcoords="offset points", va="bottom", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Super-clipping power cap (kW)", fontsize=9, color=MUTED)
        ax.set_xticks([150, 200, 250, 300, 350, 450])
        ax.set_xlim(140, 520)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def longest_free_run(v_limit: np.ndarray) -> tuple[int, int]:
    free = np.isinf(v_limit)
    best, start = (0, 0), None
    for i, f in enumerate(np.append(free, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def straight(profiles: Path, event: str, car: str, out: Path):
    a = np.load(profiles / f"{event}_{car}_250.npz")
    b = np.load(profiles / f"{event}_{car}_350.npz")
    i, j = longest_free_run(a["v_limit"])
    i, j = max(0, i - 20), min(len(a["v_limit"]), j + 10)
    x = a["distance_m"][i:j] - a["distance_m"][i]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.0, 4.6), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [3, 2]})
    for ax in (top, bottom):
        style(ax)
    for z, color, name in ((a, SERIES[0], "250 kW cap (before Miami)"), (b, SERIES[1], "350 kW cap (from Miami)")):
        top.plot(x, z["v"][i:j] * 3.6, color=color, linewidth=2, label=name)
        bottom.plot(x, z["superclip"][i:j] / 1e3, color=color, linewidth=2)
    top.set_title(f"{event}, longest straight, lap-time-optimal strategy: speed (km/h)", fontsize=10,
                  color=INK, loc="left")
    bottom.set_title("Super-clipping power (kW)", fontsize=10, color=INK, loc="left")
    bottom.set_xlabel("Distance along the straight (m)", fontsize=9, color=MUTED)
    bottom.set_ylim(bottom=0)
    top.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def location_split(profiles: Path, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (e, car, cap), _ in runs.groupby(["event", "car", "cap_kw"]):
        z = np.load(profiles / f"{e}_{car}_{int(cap)}.npz")
        energy = z["superclip"] * z["dt"] / 1e6
        kmh = z["v"] * 3.6
        i, j = longest_free_run(z["v_limit"])
        rows.append({"event": e, "car": car, "cap_kw": cap, "superclip_mj": energy.sum(),
                     "longest_straight_mj": energy[i:j].sum(), "below_200_kmh_mj": energy[kmh < 200].sum(),
                     "200_250_kmh_mj": energy[(kmh >= 200) & (kmh < 250)].sum(),
                     "above_250_kmh_mj": energy[kmh >= 250].sum()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/fia_fix_fixed")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--event", default="Canada")
    ap.add_argument("--car", default="original")
    args = ap.parse_args()
    d, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(d / "runs.csv")
    dose(runs, out / "fia_fix_dose.png")
    location_split(d / "profiles", runs).to_csv(d / "location_split.csv", index=False)
    straight(d / "profiles", args.event, args.car, out / "fia_fix_straight.png")
    print(f"Wrote {out / 'fia_fix_dose.png'}, {out / 'fia_fix_straight.png'} and {d / 'location_split.csv'}")


if __name__ == "__main__":
    main()
