"""What one backward-pass step of the Strategy Lab costs, as XLA compiles it, against what it takes.

XLA's own cost analysis of one step (one track point: every speed x charge x command on the fine
grid) gives the arithmetic and the memory traffic. Dividing by the device's peak rates gives the
least time each could take; the measured time per step comes from a gpu_benchmark.py result.
--parts also times the step's three parts on their own (transition, interpolation, whole step),
to find which one the time goes to. --hlo counts, in the compiled step, the operations that run
once per (state, command) pair, by kind, to show which ones the device repeats.

    python scripts/step_cost.py --bench results/gpu_benchmark/kaggle_gpu.json --parts   # on the GPU
"""
import argparse
import dataclasses
import json
import re
import statistics
import time
from pathlib import Path

# Vendor peak rates: float64 FLOP/s and memory bytes/s. T4: 254.4 GFLOPS (1:32 of FP32), 320 GB/s.
PEAKS = {"Tesla T4": (254.4e9, 320e9), "Tesla P100-PCIE-16GB": (4.7e12, 732e9)}


def _setup(track_path: str, car_path: str, dtype: str = "float64", layout: str = "vus"):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from erslens.ingest import load_track
    from erslens.jaxsim import _step, params_of, static_track
    from erslens.optimal import _interp, _transition, grid_with, race_car, start_at_slowest_point
    from erslens.params import load_car_params

    rc = race_car(load_car_params(car_path))
    tr = start_at_slowest_point(load_track(track_path), rc)
    g = grid_with(rc, 0.125, 321)
    st = static_track(tr, rc)
    p, mass = params_of(rc), rc.mass_kg + rc.fuel_kg
    unlimited = rc.with_(harvest_per_lap_j=np.inf)
    V, S, U = (jnp.asarray(a, dtype=dtype) for a in (g.v_ms, g.soc_j, g.cmd_w))
    p = jax.tree.map(lambda a: a.astype(dtype), p)
    mass, ds = jnp.asarray(mass, dtype=dtype), jnp.asarray(st.ds, dtype=dtype)
    st = dataclasses.replace(st, env=st.env.astype(dtype))

    # vus: arrays shaped (speed, command, charge), the solver's order since 2026-10-31; vsu is the
    # earlier (speed, charge, command), kept to measure the change.
    cmd_axis = {"vsu": -1, "vus": 1}[layout]

    def transition(target, straight):
        if layout == "vus":
            return _transition(p, unlimited, ds, mass, V, S, U, target, straight)
        carry = (V[:, None, None], S[None, :, None], 0.0)
        (v_n, s_n, e_in), (_, _, _, _, dt) = _step(p, unlimited, ds, mass, carry, (U[None, None, :], target, straight))
        return v_n, s_n, e_in, dt

    def interp(value_next, v_n, s_n):
        return _interp(value_next, g, v_n, s_n)

    def step(value_next, target, straight, price):
        v_n, s_n, e_in, dt = transition(target, straight)
        return (dt + price * e_in + interp(value_next, v_n, s_n)).min(axis=cmd_axis)

    value_next = jnp.zeros((len(V), len(S)), dtype=dtype)
    return tr, st, (len(V), len(S), len(U)), transition, interp, step, value_next


def step_cost(track_path: str, car_path: str) -> dict:
    import jax

    tr, st, (nv, ns, nu), _, _, step, value_next = _setup(track_path, car_path)
    c = jax.jit(step).lower(value_next, st.env[0], st.straight[0], 0.0).compile()
    ca = c.cost_analysis()
    ca = ca[0] if isinstance(ca, list) else ca
    hlo = c.as_text()
    return {"hlo": hlo, "pairs": int(nv * ns * nu), "flops": float(ca["flops"]),
            "bytes": float(ca["bytes accessed"]), "transcendentals": float(ca.get("transcendentals", 0.0)),
            "fusions": len(re.findall(r"^\s*\S+ = \S+ fusion\(", hlo, re.M)), "track_points": int(tr.n)}


PER_PAIR_OPS = ("divide", "sqrt", "rsqrt", "power", "exponential", "maximum", "minimum", "clamp",
                "select", "compare", "multiply", "add", "subtract", "negate", "gather", "convert")


def per_pair_ops(hlo: str, pairs: int) -> dict:
    """Instructions in the optimized HLO whose result has one element per (state, command) pair.
    Inside a fused kernel each runs once per pair, so the counts say what each pair costs."""
    counts = {}
    for m in re.finditer(r"= ([a-z]\d+)\[([\d,]*)\]\{?[^ ]* ([a-z-]+)\(", hlo):
        dims = [int(d) for d in m.group(2).split(",") if d]
        n = 1
        for d in dims:
            n *= d
        if n == pairs and m.group(3) in PER_PAIR_OPS:
            key = f"{m.group(1)} {m.group(3)}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _median_s(f, *args, repeats: int) -> float:
    import jax

    jax.block_until_ready(f(*args))
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        runs.append(time.perf_counter() - t)
    return statistics.median(runs)


def same_result_across_layouts(track_path: str, car_path: str) -> float:
    """Largest difference between the two layouts' step results, on a value table that varies with
    both speed and charge (zeros would hide a wrong lookup)."""
    import jax
    import numpy as np

    outs = []
    for layout in ("vsu", "vus"):
        _, st, (nv, ns, _), _, _, step, _ = _setup(track_path, car_path, layout=layout)
        table = np.add.outer(np.linspace(0.0, 30.0, nv), np.cos(np.linspace(0.0, 7.0, ns)))
        outs.append(np.asarray(jax.jit(step)(table, st.env[0], st.straight[0], 1e-7)))
    return float(np.abs(outs[0] - outs[1]).max())


def time_parts(track_path: str, car_path: str, repeats: int = 20, dtype: str = "float64",
               layout: str = "vus") -> dict:
    """Each part as its own compiled function. Run alone, transition and interpolation also write
    their outputs to memory, which the fused step need not do. dtype float32 is for timing only:
    whether float32 keeps the solver's results is a separate, pre-registered test."""
    import jax

    _, st, _, transition, interp, step, value_next = _setup(track_path, car_path, dtype, layout)
    target, straight = st.env[0], st.straight[0]
    v_n, s_n, _, _ = jax.jit(transition)(target, straight)
    if v_n.dtype != dtype:
        raise TypeError(f"transition ran in {v_n.dtype}, not {dtype}")
    return {"transition_s": _median_s(jax.jit(transition), target, straight, repeats=repeats),
            "interp_s": _median_s(jax.jit(interp), value_next, v_n, s_n, repeats=repeats),
            "step_s": _median_s(jax.jit(step), value_next, target, straight, 0.0, repeats=repeats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="gpu_benchmark.py result from this device")
    ap.add_argument("--event", default="Australia")
    ap.add_argument("--car", default="configs/car_2026_q_low.yaml")
    ap.add_argument("--parts", action="store_true", help="also time the step's parts on their own")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                    help="precision for --parts timing (float32: timing only)")
    ap.add_argument("--layout", default="vus", choices=["vus", "vsu"],
                    help="array order for --parts timing: vus (the solver's) or vsu (the earlier one)")
    ap.add_argument("--hlo", help="write the compiled step here and count its per-pair operations")
    ap.add_argument("--out")
    args = ap.parse_args()
    import jax

    bench = json.loads(Path(args.bench).read_text())
    device = jax.devices()[0]
    kind = getattr(device, "device_kind", device.platform)
    cost = step_cost(f"results/tracks/{args.event}.npz", args.car)
    hlo = cost.pop("hlo")
    measured = bench["events"][args.event]["backward"]["median_s"] / cost["track_points"]
    out = {"device": kind, "platform": device.platform, "event": args.event, **cost,
           "measured_s_per_step": measured}
    print(f"{kind}: one step = {cost['pairs']:,} (state, command) pairs, {cost['fusions']} fused kernels")
    print(f"  {cost['flops'] / 1e9:.3f} GFLOP, {cost['bytes'] / 1e9:.3f} GB moved, measured {measured * 1e3:.2f} ms")
    if kind in PEAKS:
        flops_peak, bw_peak = PEAKS[kind]
        out["floor_compute_s"], out["floor_memory_s"] = cost["flops"] / flops_peak, cost["bytes"] / bw_peak
        print(f"  least time from float64 arithmetic {out['floor_compute_s'] * 1e3:.2f} ms "
              f"({100 * out['floor_compute_s'] / measured:.0f}% of measured), "
              f"from memory traffic {out['floor_memory_s'] * 1e3:.2f} ms "
              f"({100 * out['floor_memory_s'] / measured:.0f}% of measured)")
    if args.parts:
        track = f"results/tracks/{args.event}.npz"
        parts = time_parts(track, args.car, dtype=args.dtype, layout=args.layout)
        name = "parts" + ("" if args.dtype == "float64" else f"_{args.dtype}") + f"_{args.layout}"
        out[name] = parts
        out["max_diff_vus_vsu"] = diff = same_result_across_layouts(track, args.car)
        print(f"  vus against vsu, largest difference in the step result: {diff:.3g}")
        print(f"  alone ({args.dtype}, {args.layout}): " + ", ".join(f"{k[:-2]} {v * 1e3:.2f} ms" for k, v in parts.items()))
    if args.hlo:
        Path(args.hlo).parent.mkdir(parents=True, exist_ok=True)
        Path(args.hlo).write_text(hlo)
        out["per_pair_ops"] = per_pair_ops(hlo, cost["pairs"])
        print("  per pair: " + ", ".join(f"{k} x{v}" for k, v in out["per_pair_ops"].items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
