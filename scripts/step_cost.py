"""What one backward-pass step of the Strategy Lab costs, as XLA compiles it, against what it takes.

XLA's own cost analysis of one step (one track point: every speed x charge x command on the fine
grid) gives the arithmetic and the memory traffic. Dividing by the device's peak rates gives the
least time each could take; the measured time per step comes from a gpu_benchmark.py result.
--parts also times the step's three parts on their own (transition, interpolation, whole step),
to find which one the time goes to.

    python scripts/step_cost.py --bench results/gpu_benchmark/kaggle_gpu.json --parts   # on the GPU
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

# Vendor peak rates: float64 FLOP/s and memory bytes/s. T4: 254.4 GFLOPS (1:32 of FP32), 320 GB/s.
PEAKS = {"Tesla T4": (254.4e9, 320e9), "Tesla P100-PCIE-16GB": (4.7e12, 732e9)}


def _setup(track_path: str, car_path: str):
    import jax.numpy as jnp
    import numpy as np

    from erslens.ingest import load_track
    from erslens.jaxsim import params_of, static_track
    from erslens.optimal import _interp, _transition, grid_with, race_car, start_at_slowest_point
    from erslens.params import load_car_params

    rc = race_car(load_car_params(car_path))
    tr = start_at_slowest_point(load_track(track_path), rc)
    g = grid_with(rc, 0.125, 321)
    st = static_track(tr, rc)
    p, mass = params_of(rc), rc.mass_kg + rc.fuel_kg
    unlimited = rc.with_(harvest_per_lap_j=np.inf)
    V, S, U = (jnp.asarray(a) for a in (g.v_ms, g.soc_j, g.cmd_w))

    def transition(target, straight):
        return _transition(p, unlimited, st.ds, mass, V, S, U, target, straight)

    def interp(value_next, v_n, s_n):
        return _interp(value_next, g, v_n, s_n)

    def step(value_next, target, straight, price):
        v_n, s_n, e_in, dt = transition(target, straight)
        return (dt + price * e_in + interp(value_next, v_n, s_n)).min(axis=-1)

    value_next = jnp.zeros((len(V), len(S)))
    return tr, st, (len(V), len(S), len(U)), transition, interp, step, value_next


def step_cost(track_path: str, car_path: str) -> dict:
    import jax

    tr, st, (nv, ns, nu), _, _, step, value_next = _setup(track_path, car_path)
    c = jax.jit(step).lower(value_next, st.env[0], st.straight[0], 0.0).compile()
    ca = c.cost_analysis()
    ca = ca[0] if isinstance(ca, list) else ca
    hlo = c.as_text()
    return {"pairs": int(nv * ns * nu), "flops": float(ca["flops"]),
            "bytes": float(ca["bytes accessed"]), "transcendentals": float(ca.get("transcendentals", 0.0)),
            "fusions": len(re.findall(r"^\s*\S+ = \S+ fusion\(", hlo, re.M)), "track_points": int(tr.n)}


def _median_s(f, *args, repeats: int) -> float:
    import jax

    jax.block_until_ready(f(*args))
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        runs.append(time.perf_counter() - t)
    return statistics.median(runs)


def time_parts(track_path: str, car_path: str, repeats: int = 20) -> dict:
    """Each part as its own compiled function. Run alone, transition and interpolation also write
    their outputs to memory, which the fused step need not do."""
    import jax

    _, st, _, transition, interp, step, value_next = _setup(track_path, car_path)
    target, straight = st.env[0], st.straight[0]
    v_n, s_n, _, _ = jax.jit(transition)(target, straight)
    return {"transition_s": _median_s(jax.jit(transition), target, straight, repeats=repeats),
            "interp_s": _median_s(jax.jit(interp), value_next, v_n, s_n, repeats=repeats),
            "step_s": _median_s(jax.jit(step), value_next, target, straight, 0.0, repeats=repeats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="gpu_benchmark.py result from this device")
    ap.add_argument("--event", default="Australia")
    ap.add_argument("--car", default="configs/car_2026_q_low.yaml")
    ap.add_argument("--parts", action="store_true", help="also time the step's parts on their own")
    ap.add_argument("--out")
    args = ap.parse_args()
    import jax

    bench = json.loads(Path(args.bench).read_text())
    device = jax.devices()[0]
    kind = getattr(device, "device_kind", device.platform)
    cost = step_cost(f"results/tracks/{args.event}.npz", args.car)
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
        parts = time_parts(f"results/tracks/{args.event}.npz", args.car)
        out["parts"] = parts
        print("  alone: " + ", ".join(f"{k[:-2]} {v * 1e3:.2f} ms" for k, v in parts.items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
