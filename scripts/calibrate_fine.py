"""Re-solve the calibration's candidate cars on the fine DP grid (research log, 2026-10-20).

The coarse grid in calibrate_car.py has no convergence check and some of its errors are not
monotone in ICE power. Candidates, chosen from the coarse grid.csv only: the best ICE power for
each CdA, the four leave-one-out picks and the overall best. Each is solved at every circuit
with solve_refined (0.125 m/s, 321 charge levels); a solve whose DP prediction and replay differ
by more than CONVERGED_S is flagged. Leave one circuit out is then repeated on the fine errors,
over the candidates. Results are appended to <out>/fine.csv (resumable), report <out>/fine.md.

    python scripts/calibrate_fine.py --out results/calibration_q
"""
import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from calibrate_car import EVENTS, calibration_car, load, run_errors  # noqa: E402
from erslens.optimal import solve_refined  # noqa: E402
from erslens.params import load_car_params  # noqa: E402
from erslens.strategy import CONVERGED_S  # noqa: E402

FIELDS = ["event", "cda", "ice_w", "run_sq_err", "n_runs", "lap_err_s", "dp_minus_sim_s"]


def complete(g):
    g = g.assign(cda=g.cda.round(4), ice_w=g.ice_w.round())
    ok = g.groupby(["cda", "ice_w"]).event.nunique() == len(EVENTS)
    return g.set_index(["cda", "ice_w"]).loc[ok[ok].index].reset_index()


def loo_picks(g):
    return {held: g[g.event != held].groupby(["cda", "ice_w"]).run_sq_err.sum().idxmin() for held in EVENTS}


def candidates(coarse):
    total = coarse.groupby(["cda", "ice_w"]).run_sq_err.sum()
    picks = {(c, total.loc[c].idxmin()) for c in total.index.get_level_values(0).unique()}
    picks |= set(loo_picks(coarse).values()) | {total.idxmin()}
    return sorted(picks)


def report(fine, coarse, out):
    total = fine.groupby(["cda", "ice_w"]).run_sq_err.sum()
    lines = ["# Calibration candidates on the fine DP grid", "",
             "Fine grid 0.125 m/s, 321 charge levels; same target and acceptance rule as the coarse grid.", "",
             "| CdA | ICE wheel power | Fine squared run error | Coarse | Worst DP-replay gap |",
             "|---|---|---|---|---|"]
    ctotal = coarse.groupby(["cda", "ice_w"]).run_sq_err.sum()
    for (c, w), e in total.sort_index().items():
        gap = fine[(fine.cda == c) & (fine.ice_w == w)].dp_minus_sim_s.abs().max()
        flag = "" if gap <= CONVERGED_S else " (not converged)"
        lines.append(f"| {c:.2f} | {w / 1e3:.0f} kW | {e:.2f} | {ctotal[(c, w)]:.2f} | {gap:.3f} s{flag} |")
    lines += ["", "## Leave one circuit out, fine grid", "",
              "| Held-out circuit | CdA | ICE wheel power | Held-out lap error | Within 0.5 s |", "|---|---|---|---|---|"]
    for held, (c, w) in loo_picks(fine).items():
        e = fine[(fine.event == held) & (fine.cda == c) & (fine.ice_w == w)].lap_err_s.iloc[0]
        lines.append(f"| {held} | {c:.2f} | {w / 1e3:.0f} kW | {e:+.3f} s | {'yes' if abs(e) <= 0.5 else 'no'} |")
    best = total.idxmin()
    cdas = sorted(total.index.get_level_values(0).unique())
    edge = best[0] in (cdas[0], cdas[-1])
    lines += ["", f"Best on the fine grid: CdA {best[0]:.2f}, {best[1] / 1e3:.0f} kW"
              + (" (lowest or highest candidate CdA)." if edge else ".")]
    text = "\n".join(lines)
    (out / "fine.md").write_text(text + "\n")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/calibration_q")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--list", action="store_true", help="only print the candidates")
    args = ap.parse_args()
    out = Path(args.out)
    coarse = complete(pd.read_csv(out / "grid.csv"))
    cands = candidates(coarse)
    print(f"{len(cands)} candidates: " + ", ".join(f"{c:.2f}/{w / 1e3:.0f} kW" for c, w in cands), flush=True)
    if args.list:
        return
    path = out / "fine.csv"
    done = set()
    if path.exists():
        done = {(r.event, round(r.cda, 4), round(r.ice_w)) for r in pd.read_csv(path).itertuples()}
    base = load_car_params("configs/car_2026.yaml")
    data = load(args.year)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not done and f.tell() == 0:
            w.writerow(FIELDS)
        for cda, ice_w in cands:
            for event in EVENTS:
                if (event, round(cda, 4), round(ice_w)) in done:
                    continue
                track, spans, dt_real = data[event]
                sol = solve_refined(track, calibration_car(base, event, cda, ice_w))
                per_run, lap_err = run_errors(event, track, spans, dt_real, base, cda, ice_w, sol=sol)
                gap = sol.lap_time_dp_s - sol.lap_time_sim_s
                w.writerow([event, cda, ice_w, sum(e * e for e in per_run), len(per_run), lap_err, gap])
                f.flush()
                print(f"{event:10s} CdA {cda:.2f} ICE {ice_w / 1e3:.0f} kW: model minus real {lap_err:+.3f} s "
                      f"(DP-replay {gap:+.3f} s)", flush=True)
    report(complete(pd.read_csv(path)), coarse, out)


if __name__ == "__main__":
    main()
