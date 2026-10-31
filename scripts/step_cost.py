"""What one backward-pass step of the Strategy Lab costs, as XLA compiles it, against what it takes.

XLA's own cost analysis of one step (one track point: every speed x charge x command on the fine
grid) gives the arithmetic and the memory traffic. Dividing by the device's peak rates gives the
least time each could take; the measured time per step comes from a gpu_benchmark.py result.

    python scripts/step_cost.py --bench results/gpu_benchmark/kaggle_gpu.json   # on the GPU
"""
import argparse
import json
import re
from pathlib import Path

# Vendor peak rates: float64 FLOP/s and memory bytes/s. T4: 254.4 GFLOPS (1:32 of FP32), 320 GB/s.
PEAKS = {"Tesla T4": (254.4e9, 320e9), "Tesla P100-PCIE-16GB": (4.7e12, 732e9)}


def step_cost(track_path: str, car_path: str) -> dict:
    import jax
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

    def step(value_next, target, straight, price):
        v_n, s_n, e_in, dt = _transition(p, unlimited, st.ds, mass, V, S, U, target, straight)
        return (dt + price * e_in + _interp(value_next, g, v_n, s_n)).min(axis=-1)

    c = jax.jit(step).lower(jnp.zeros((len(V), len(S))), st.env[0], st.straight[0], 0.0).compile()
    ca = c.cost_analysis()
    ca = ca[0] if isinstance(ca, list) else ca
    hlo = c.as_text()
    return {"pairs": int(len(V) * len(S) * len(U)), "flops": float(ca["flops"]),
            "bytes": float(ca["bytes accessed"]), "transcendentals": float(ca.get("transcendentals", 0.0)),
            "fusions": len(re.findall(r"^\s*\S+ = \S+ fusion\(", hlo, re.M)), "track_points": int(tr.n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="gpu_benchmark.py result from this device")
    ap.add_argument("--event", default="Australia")
    ap.add_argument("--car", default="configs/car_2026_q_low.yaml")
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
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
