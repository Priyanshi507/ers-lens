"""Two figures for the SSAC paper, from committed results only.

    python scripts/paper_figures.py      # docs/paper_trade.png, docs/paper_model_vs_real.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fia_fix_figures import GRID, INK, MUTED, SURFACE  # noqa: E402
from hindcast_2026 import table  # noqa: E402

OLD, NEW, REAL = "#8a94a3", "#2a6fdb", "#c8372d"
PLACE = {"Dutch Grand Prix": "Netherlands", "Italian Grand Prix": "Italy", "British Grand Prix": "Britain",
         "Hungarian Grand Prix": "Hungary", "Spanish Grand Prix": "Spain (Madrid)", "Barcelona Grand Prix": "Barcelona"}
MEXICO_MODEL_KMH = 0.86  # solved on Kaggle at 6da54b6 before the race was opened (research log, 2026-11-15)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def trade(out: Path):
    r = pd.read_csv("results/fia_fix_calendar_lc/runs.csv")
    p = r.pivot_table(index="event", columns="cap_kw", values=["coast_mj", "superclip_mj"])
    p = p[p.coast_mj[250] >= 0.05].sort_values(("coast_mj", 250))
    labels = [PLACE.get(e, e.replace(" Grand Prix", "")) for e in p.index]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), facecolor=SURFACE, sharey=True)
    for ax, col, title in zip(axes, ("coast_mj", "superclip_mj"),
                              ("Lift-and-coast energy per lap (MJ)", "Super-clipping energy per lap (MJ)")):
        style(ax)
        y = range(len(p))
        ax.hlines(y, p[col][250], p[col][350], color=GRID, linewidth=2, zorder=1)
        ax.scatter(p[col][250], y, color=OLD, s=28, zorder=2, label="250 kW limit (before Miami)")
        ax.scatter(p[col][350], y, color=NEW, s=28, zorder=3, label="350 kW limit (from Miami)")
        ax.set_yticks(list(y), labels, fontsize=8, color=INK)
        ax.set_xlabel(title, fontsize=9, color=MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_xlim(left=0)
    axes[0].legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def model_vs_real(out: Path):
    t = table(Path("results/fia_fix_calendar_c"), Path("results/multi_rebased_mid/R/race_summary.csv"))
    mx = pd.read_csv("results/oos_real/R/race_summary.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=SURFACE)
    style(ax)
    hi = 50
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(41, 44.5, "model = real", fontsize=8, color=MUTED, ha="right")
    ax.scatter(t.observed_drop_kmh, t.model_drop_kmh, color=NEW, s=30, zorder=3, label="16 races of 2026")
    ax.scatter([mx.drop_kmh_median], [MEXICO_MODEL_KMH], color=REAL, s=46, marker="D", zorder=4,
               label="Mexico City, predicted before the race was opened")
    for r in t.itertuples():
        if r.event in ("Japan", "Italian Grand Prix", "Canada", "China", "Barcelona Grand Prix", "Austria"):
            ax.annotate(PLACE.get(r.event, r.event), (r.observed_drop_kmh, r.model_drop_kmh),
                        xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(-1, hi)
    ax.set_ylim(-1, hi)
    ax.set_xlabel("Real cars: median speed lost at full throttle (km/h)", fontsize=9, color=MUTED)
    ax.set_ylabel("Model optimum (km/h)", fontsize=9, color=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.12), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    trade(Path("docs/paper_trade.png"))
    model_vs_real(Path("docs/paper_model_vs_real.png"))
    print("wrote docs/paper_trade.png, docs/paper_model_vs_real.png")
