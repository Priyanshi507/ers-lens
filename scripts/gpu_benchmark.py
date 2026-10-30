"""Strategy Lab on CPU and GPU: the same solves, timed, with results checked for equality.

Step 1 of the GPU work (research log, 2026-10-31). No solver code changes: whatever device JAX
finds runs the existing dynamic program. Two measurements per circuit:
  backward  one backward pass of the dynamic program on the fine grid, after compilation;
            the median of --repeats runs (the inner loop every solve repeats many times)
  solve     one complete solve_refined call, compilation included, as the Strategy Lab runs it

Tracks come from results/tracks (scripts/export_tracks.py), so a machine without the FastF1 cache,
such as a Kaggle notebook, runs the same circuits.

    python scripts/gpu_benchmark.py --out results/gpu_benchmark/cpu_m4.json
    python scripts/gpu_benchmark.py --out results/gpu_benchmark/kaggle_gpu.json      # on Kaggle
    python scripts/gpu_benchmark.py --compare results/gpu_benchmark/cpu_m4.json results/gpu_benchmark/kaggle_gpu.json
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

EVENTS = ("Australia", "China", "Miami", "Canada")
CAR = "configs/car_2026_q_low.yaml"
# Results must agree to this before any speed-up is reported (pre-registered, 2026-10-31).
SAME_LAP_S = 1e-3


def device_info() -> dict:
    import jax

    d = jax.devices()[0]
    return {"platform": d.platform, "device": getattr(d, "device_kind", str(d)), "n_devices": len(jax.devices()),
            "jax": jax.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "processor": platform.processor() or platform.machine()}


def time_backward(track, car, repeats: int) -> dict:
    import jax

    from erslens.optimal import _Compiled, grid_with, race_car, start_at_slowest_point

    rc = race_car(car)
    tr = start_at_slowest_point(track, rc)
    grid = grid_with(rc, 0.125, 321)
    c = _Compiled(tr, rc, grid, True)
    soc0 = 0.5 * rc.es_capacity_j
    t = time.perf_counter()
    values, _ = c.backward(soc0, 0.0)
    jax.block_until_ready(values)
    first = time.perf_counter() - t
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        values, _ = c.backward(soc0, 0.0)
        jax.block_until_ready(values)
        runs.append(time.perf_counter() - t)
    return {"first_call_s": round(first, 3), "median_s": round(statistics.median(runs), 4),
            "runs_s": [round(r, 4) for r in runs],
            "states_per_point": int(len(grid.v_ms) * len(grid.soc_j) * len(grid.cmd_w)),
            "track_points": int(tr.n), "value_checksum": float(np.asarray(values[0]).sum())}


def time_solve(track, car) -> dict:
    from erslens.optimal import solve_refined

    t = time.perf_counter()
    sol = solve_refined(track, car)
    seconds = time.perf_counter() - t
    return {"seconds": round(seconds, 2), "lap_s": round(sol.lap_time_sim_s, 6),
            "dp_pred_s": round(sol.lap_time_dp_s, 6), "harvest_price": sol.harvest_price_s_per_j}


def run(events, repeats: int, skip_solve: bool) -> dict:
    from erslens.ingest import load_track
    from erslens.params import load_car_params

    car = load_car_params(CAR)
    out = {"device": device_info(), "car": CAR, "events": {}}
    for e in events:
        track = load_track(f"results/tracks/{e}.npz")
        print(f"{e}: backward pass ...", flush=True)
        entry = {"backward": time_backward(track, car, repeats)}
        print(f"  {entry['backward']['median_s']} s (first call {entry['backward']['first_call_s']} s)", flush=True)
        if not skip_solve:
            print(f"{e}: full solve ...", flush=True)
            entry["solve"] = time_solve(track, car)
            print(f"  {entry['solve']['seconds']} s, lap {entry['solve']['lap_s']:.3f} s", flush=True)
        out["events"][e] = entry
    return out


def compare(cpu_path: str, gpu_path: str) -> int:
    cpu, gpu = (json.loads(Path(p).read_text()) for p in (cpu_path, gpu_path))
    print(f"CPU: {cpu['device']['processor']} ({cpu['device']['platform']}), GPU: {gpu['device']['device']} "
          f"({gpu['device']['platform']}), JAX {cpu['device']['jax']} / {gpu['device']['jax']}\n")
    print("| Circuit | Backward pass CPU / GPU | Speed-up | Full solve CPU / GPU | Speed-up | Lap time CPU / GPU | Same result |")
    print("|---|---|---|---|---|---|---|")
    same_all = True
    for e, c in cpu["events"].items():
        g = gpu["events"].get(e)
        if g is None:
            continue
        b = c["backward"]["median_s"] / g["backward"]["median_s"]
        row = f"| {e} | {c['backward']['median_s']:.3f} / {g['backward']['median_s']:.4f} s | {b:.1f}x |"
        if "solve" in c and "solve" in g:
            same = abs(c["solve"]["lap_s"] - g["solve"]["lap_s"]) <= SAME_LAP_S
            same_all &= same
            row += (f" {c['solve']['seconds']:.0f} / {g['solve']['seconds']:.0f} s | "
                    f"{c['solve']['seconds'] / g['solve']['seconds']:.1f}x | "
                    f"{c['solve']['lap_s']:.3f} / {g['solve']['lap_s']:.3f} s | {'yes' if same else 'NO'} |")
        else:
            row += " - | - | - | - |"
        print(row)
    if not same_all:
        print("\nResults differ by more than 0.001 s: no speed-up may be reported until the cause is found.")
    return 0 if same_all else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=list(EVENTS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-solve", action="store_true", help="backward-pass timing only")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("CPU_JSON", "GPU_JSON"))
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    if not args.out:
        ap.error("--out is required unless --compare is given")
    data = run(args.events, args.repeats, args.skip_solve)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(data, indent=1))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
