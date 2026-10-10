"""Diagnostics for the two failed predictions, S5 and S7 (exploratory; not pre-registered).

S7: compares where the optimum and real cars reach their speed peak on the longest straight,
measured both from the start of full throttle (the pre-registered definition) and from the
braking point, which a corner fixes for the qualifying reference lap and race laps alike.
S5: how bang-bang the optimal deployment is, and, with --tracks, where on the lap the optimum
gains its time over the best tuned simple rule.

    python scripts/strategy_diagnose.py             # S7 and bang-bang shares, committed data only
    python scripts/strategy_diagnose.py --tracks    # adds the S5 time breakdown (needs FastF1 cache)

Writes results/strategy/diagnostics.md.
"""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

from erslens.jaxsim import params_of, simulate_laps, static_track
from erslens.optimal import race_car, start_at_slowest_point
from erslens.params import load_car_params
from erslens.physics import mguk_cap_kmh
from erslens.strategy import race_rules, resolve_runs

EVENTS = ("Australia", "China", "Miami", "Canada")
SITE_ID = {"Australia": "australian", "China": "chinese", "Miami": "miami", "Canada": "canadian"}
AT_CAP, AT_ZERO = 0.98, 0.02
RESULTS = Path("results/strategy")


def observed_traces(event: str) -> list[dict]:
    data = json.loads(Path(f"site/data/races/{SITE_ID[event]}.json").read_text())
    return [lap["trace"] for lap in data["laps"] if lap["clipping"] and "trace" in lap]


def s7_table(nominal: pd.DataFrame) -> list[str]:
    lines = ["## S7: where the speed peak falls on the longest straight", "",
             "Medians over clipping race laps. 'Peak to braking' is measured back from the end of "
             "full throttle, which the corner fixes for the reference lap and race laps alike.", "",
             "| Circuit | Full-throttle run, optimum / real | Start to peak, optimum / real | "
             "Peak to braking, optimum / real | Real cars clip earlier by |", "|---|---|---|---|---|"]
    for e in EVENTS:
        runs, peaks = [], []
        for t in observed_traces(e):
            a, b = t["flat"]
            runs.append(t["d"][b] - t["d"][a])
            peaks.append(t["d"][t["peak"]] - t["d"][a])
        runs, peaks = np.array(runs, float), np.array(peaks, float)
        o = nominal.loc[e]
        opt_tail, obs_tail = o.run_length_m - o.peak_from_start_m, float(np.median(runs - peaks))
        lines.append(f"| {e} | {o.run_length_m:.0f} / {np.median(runs):.0f} m | "
                     f"{o.peak_from_start_m:.0f} / {np.median(peaks):.0f} m | {opt_tail:.0f} / {obs_tail:.0f} m | "
                     f"{obs_tail - opt_tail:.0f} m (n={len(runs)}) |")
    return lines


def bang_bang_table(car) -> list[str]:
    lines = ["", "## S5: how bang-bang the optimum is", "",
             f"Share of full-throttle points with deployment at the MGU-K cap (>= {AT_CAP:.0%} of it), "
             f"at zero (<= {AT_ZERO:.0%}), or in between. The optimizer could choose any level in 25 kW steps.", "",
             "| Circuit | Rules | At cap | Zero | Intermediate |", "|---|---|---|---|---|"]
    for e in EVENTS:
        for label, tag, c in (("original", "", car), ("corrected", "rules_", race_rules(car, e))):
            p = np.load(RESULTS / f"profile_{tag}{e}.npz")
            free = np.isinf(p["v_limit"])
            cap = mguk_cap_kmh(p["v"][free] * 3.6, c)
            dep = p["deploy"][free]
            full, zero = dep >= AT_CAP * cap, dep <= AT_ZERO * cap
            lines.append(f"| {e} | {label} | {full.mean():.0%} | {zero.mean():.0%} | {(~full & ~zero).mean():.0%} |")
    return lines


def zones(free: np.ndarray) -> list[tuple[int, int, bool]]:
    """Contiguous [a, b) runs of full throttle (True) and of speed-limited running (False)."""
    edges = np.flatnonzero(np.diff(free.astype(int))) + 1
    bounds = np.concatenate([[0], edges, [len(free)]])
    return [(int(a), int(b), bool(free[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def time_breakdown(track, car, opt_cmd: np.ndarray, simple_cmd: np.ndarray, top: int = 4) -> list[dict]:
    """Seconds the optimum gains over the simple rule in each zone of the (rotated) lap."""
    rc = race_car(car)
    tr = start_at_slowest_point(track, rc)
    st = static_track(tr, rc)

    def dt(cmd):
        out = simulate_laps(params_of(rc), jnp.asarray(cmd), st, rc, 1, 0.5, float(st.env[-1]))
        return np.asarray(out["dt"][0])

    gain = dt(simple_cmd) - dt(opt_cmd)
    free = np.isinf(tr.v_limit_ms)
    rows = [{"zone": "full throttle" if f else "corner", "start_m": a * tr.ds, "length_m": (b - a) * tr.ds,
             "gain_s": float(gain[a:b].sum())} for a, b, f in zones(free)]
    df = pd.DataFrame(rows)
    straights = df[df.zone == "full throttle"].nlargest(top, "length_m")
    return [{"what": "total", "gain_s": float(gain.sum())},
            {"what": "all corners", "gain_s": float(df[df.zone == "corner"].gain_s.sum())},
            {"what": "all straights", "gain_s": float(df[df.zone == "full throttle"].gain_s.sum())}] + [
        {"what": f"straight {r.length_m:.0f} m at {r.start_m:.0f} m", "gain_s": r.gain_s} for r in straights.itertuples()]


def breakdown_lines(car, year: int) -> list[str]:
    from erslens.ingest import reference_track
    from erslens.simple import tune

    lines = ["", "## S5: where the optimum gains its time (corrected rules)", "",
             "Optimum replayed against the best tuned simple rule on the same lap; positive = optimum faster.", "",
             "| Circuit | Best simple rule | Where | Optimum gains |", "|---|---|---|---|"]
    for e in EVENTS:
        c = race_rules(car, e)
        track = reference_track(year, e)
        best = min(tune(track, c).values(), key=lambda t: t.lap_time_s)
        opt = np.load(RESULTS / f"profile_rules_{e}.npz")["cmd_w"]
        rule = best.family + " " + ", ".join(f"{k}={v / 1e3:.0f} kW" if k.endswith("_w") else f"{k}={v}"
                                             for k, v in best.params.items())
        for r in time_breakdown(track, c, opt, best.cmd_w):
            lines.append(f"| {e} | {rule} | {r['what']} | {r['gain_s']:+.3f} s |")
            rule = ""
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", action="store_true", help="add the S5 time breakdown (loads circuits via FastF1)")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()
    car = load_car_params("configs/car_2026.yaml")
    used, _ = resolve_runs(pd.read_csv(RESULTS / "runs.csv"))
    nominal = used[used.variant == "nominal"].set_index("event")
    lines = ["# Strategy Lab diagnostics (exploratory, run after the verdicts were committed)", ""]
    lines += s7_table(nominal) + bang_bang_table(car)
    if args.tracks:
        lines += breakdown_lines(car, args.year)
    text = "\n".join(lines) + "\n"
    print(text)
    (RESULTS / "diagnostics.md").write_text(text)


if __name__ == "__main__":
    main()
