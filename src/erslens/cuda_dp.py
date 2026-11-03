"""The Strategy Lab's backward pass as one hand-written CUDA kernel (CuPy), with a NumPy twin.

One thread per (speed, charge) state loops over the power commands, so the 4.7 million
(state, command) transitions of a track point never leave registers: physics step, bilinear
lookup of the next point's value table, and the minimum, fused. The arithmetic follows
jaxsim._step (with no harvested energy yet this lap and the unlimited-recharge car, as
optimal._Compiled.backward uses it) and jax.scipy.ndimage.map_coordinates(order=1,
mode="nearest"), except that divisions by values fixed per state are done once as reciprocals,
so results match the XLA solver to rounding, not bit for bit.

reference_step is the same loop in NumPy, used to test the arithmetic on machines without a GPU.
"""
from dataclasses import dataclass

import numpy as np

from .jaxsim import V_MIN
from .physics import G

KERNEL = r"""
extern "C" __global__ void dp_step(
    const double* __restrict__ value_next, double* __restrict__ value_out,
    const double* __restrict__ cmd, const double* __restrict__ v_grid, const double* __restrict__ s_grid,
    int nv, int ns, int nu, double v0, double dv, double ds_soc, double target, int straight, double price,
    const double* __restrict__ k, double deploy_cap)
{
    int is = blockIdx.x * blockDim.x + threadIdx.x;
    int iv = blockIdx.y;
    if (is >= ns || iv >= nv) return;
    const double ds = k[0], mass = k[1], rho = k[2], crr = k[3], cda = straight ? k[4] : k[5];
    const double ice = k[6], eh = k[7], ed = k[8], deploy_max = fmin(k[9], deploy_cap), traction = k[10];
    const double superclip = k[11], cap_j = k[12], budget = k[13], vmin = k[14], g = k[15];
    const double mguk = k[16], a1 = k[17], b1 = k[18], knee = k[19], a2 = k[20], b2 = k[21];
    const bool regen_by_command = k[22] != 0.0;

    const double v = v_grid[iv], soc = s_grid[is];
    const double v_eff = fmax(v, vmin);
    const double resist = 0.5 * rho * cda * (v_eff * v_eff) + crr * mass * g;
    const double kmh = v_eff * 3.6;
    const double low = fmin(mguk, a1 - b1 * kmh);
    const double cap_k = fmax(kmh >= knee ? a2 - b2 * kmh : low, 0.0);
    const double space = cap_j - soc;
    const double dt_guess = ds / v_eff;
    const double room = fmin(budget, space) / (eh * dt_guess);
    const double f_needed = mass * (target * target - v * v) / (2.0 * ds) + resist;
    const double p_need = f_needed * v_eff;
    // Divisions hoisted out of the command loop: FP64 division is a long instruction sequence.
    const double inv_veff = 1.0 / v_eff, inv_mass = 1.0 / mass, inv_ed = 1.0 / ed;
    const double soc_limit = soc * ed / dt_guess, inv_dv = 1.0 / dv, inv_dsoc = 1.0 / ds_soc;

    double best = 1e300;  // larger than any lap time or miss penalty
    for (int u = 0; u < nu; ++u) {
        const double c = cmd[u];
        double deploy = fmin(fmin(fmax(c, 0.0), cap_k), deploy_max);
        deploy = fmin(deploy, soc_limit);
        double ice_h = c < 0.0 ? fmin(fmin(fmin(-c, cap_k), ice), room) : 0.0;
        const double p_wheel = ice + deploy - ice_h;
        const double f_drive = fmin(p_wheel * inv_veff, traction);
        const double a = (f_drive - resist) * inv_mass;
        const double v_acc = sqrt(fmax(v * v + 2.0 * a * ds, vmin * vmin));
        const bool accel = v_acc <= target;
        const bool partial = !accel && f_needed >= 0.0;
        const bool braking = !accel && f_needed < 0.0;
        const double v_next = accel ? v_acc : target;
        if (!accel) deploy = 0.0;
        ice_h = accel ? fmin(ice_h, superclip) : (partial ? fmin(ice_h, fmax(0.0, ice - p_need)) : 0.0);
        double brake_h = braking ? fmin(fmin(-p_need, cap_k), room) : 0.0;
        if (regen_by_command) brake_h = fmin(brake_h, fmax(-c, 0.0));
        const double dt = 2.0 * ds / (v + v_next);
        const double e_in = eh * (ice_h + brake_h) * dt;
        const double s_next = fmin(fmax(soc + e_in - deploy * inv_ed * dt, 0.0), cap_j);

        const double cv = (v_next - v0) * inv_dv, cs = s_next * inv_dsoc;
        const double fv = floor(cv), fs = floor(cs);
        const double wv1 = cv - fv, wv0 = 1.0 - wv1, ws1 = cs - fs, ws0 = 1.0 - ws1;
        const int lv = (int)fv, ls = (int)fs;
        const int v_lo = min(max(lv, 0), nv - 1), v_hi = min(max(lv + 1, 0), nv - 1);
        const int s_lo = min(max(ls, 0), ns - 1), s_hi = min(max(ls + 1, 0), ns - 1);
        const double interp = wv0 * ws0 * value_next[v_lo * ns + s_lo] + wv0 * ws1 * value_next[v_lo * ns + s_hi]
                            + wv1 * ws0 * value_next[v_hi * ns + s_lo] + wv1 * ws1 * value_next[v_hi * ns + s_hi];
        best = fmin(best, dt + price * e_in + interp);
    }
    value_out[iv * ns + is] = best;
}
"""


def constants(p: dict, car, ds: float, mass: float) -> np.ndarray:
    """The scalars the kernel reads, in its order. car is the unlimited-recharge car."""
    f = lambda x: float(np.asarray(x))
    return np.array([ds, mass, car.rho, car.crr, f(p["cda_straight"]), f(p["cda_corner"]), f(p["ice_power_w"]),
                     f(p["eta_harvest"]), f(p["eta_deploy"]), car.deploy_max_w, car.traction_max_n,
                     car.superclip_max_w, car.es_capacity_j, max(0.0, car.harvest_per_lap_j), V_MIN, G,
                     car.mguk_power_w, car.taper_a1_w, car.taper_b1_w_per_kmh, car.taper_knee_kmh,
                     car.taper_a2_w, car.taper_b2_w_per_kmh, float(car.regen_by_command)], dtype=np.float64)


def reference_step(value_next, v_grid, s_grid, cmd, target, straight, price, k, deploy_cap=np.inf):
    """The kernel's loop in NumPy over the whole (speed, charge) grid."""
    ds, mass, rho, crr = k[0], k[1], k[2], k[3]
    cda = k[4] if straight else k[5]
    ice, eh, ed, deploy_max, traction, superclip, cap_j, budget, vmin, g = k[6:16]
    deploy_max = min(deploy_max, deploy_cap)
    mguk, a1, b1, knee, a2, b2 = k[16:22]
    regen_by_command = k[22] != 0.0
    nv, ns = value_next.shape
    v0, dv, ds_soc = v_grid[0], v_grid[1] - v_grid[0], s_grid[1] - s_grid[0]
    v = np.asarray(v_grid, dtype=np.float64)[:, None]
    soc = np.asarray(s_grid, dtype=np.float64)[None, :]
    v_eff = np.maximum(v, vmin)
    resist = 0.5 * rho * cda * (v_eff * v_eff) + crr * mass * g
    kmh = v_eff * 3.6
    cap_k = np.maximum(np.where(kmh >= knee, a2 - b2 * kmh, np.minimum(mguk, a1 - b1 * kmh)), 0.0)
    dt_guess = ds / v_eff
    room = np.minimum(budget, cap_j - soc) / (eh * dt_guess)
    f_needed = mass * (target * target - v * v) / (2.0 * ds) + resist
    p_need = f_needed * v_eff
    inv_veff, inv_mass, inv_ed = 1.0 / v_eff, 1.0 / mass, 1.0 / ed
    soc_limit, inv_dv, inv_dsoc = soc * ed / dt_guess, 1.0 / dv, 1.0 / ds_soc
    best = np.full((nv, ns), np.inf)
    for c in cmd:
        deploy = np.minimum(np.minimum(np.maximum(c, 0.0), cap_k), deploy_max)
        deploy = np.minimum(deploy, soc_limit)
        ice_h = np.minimum(np.minimum(np.minimum(-c, cap_k), ice), room) if c < 0.0 else np.zeros_like(room)
        f_drive = np.minimum((ice + deploy - ice_h) * inv_veff, traction)
        a = (f_drive - resist) * inv_mass
        v_acc = np.sqrt(np.maximum(v * v + 2.0 * a * ds, vmin * vmin))
        accel = v_acc <= target
        partial = ~accel & (f_needed >= 0.0)
        braking = ~accel & (f_needed < 0.0)
        v_next = np.where(accel, v_acc, target)
        deploy = np.where(accel, deploy, 0.0)
        ice_h = np.where(accel, np.minimum(ice_h, superclip),
                         np.where(partial, np.minimum(ice_h, np.maximum(0.0, ice - p_need)), 0.0))
        brake_h = np.where(braking, np.minimum(np.minimum(-p_need, cap_k), room), 0.0)
        if regen_by_command:
            brake_h = np.minimum(brake_h, max(-c, 0.0))
        dt = 2.0 * ds / (v + v_next)
        e_in = eh * (ice_h + brake_h) * dt
        s_next = np.minimum(np.maximum(soc + e_in - deploy * inv_ed * dt, 0.0), cap_j)
        cv, cs = (v_next - v0) * inv_dv, s_next * inv_dsoc
        fv, fs = np.floor(cv), np.floor(cs)
        wv1, ws1 = cv - fv, cs - fs
        wv0, ws0 = 1.0 - wv1, 1.0 - ws1
        lv, ls = fv.astype(np.int64), fs.astype(np.int64)
        v_lo, v_hi = np.clip(lv, 0, nv - 1), np.clip(lv + 1, 0, nv - 1)
        s_lo, s_hi = np.clip(ls, 0, ns - 1), np.clip(ls + 1, 0, ns - 1)
        interp = (wv0 * ws0 * value_next[v_lo, s_lo] + wv0 * ws1 * value_next[v_lo, s_hi]
                  + wv1 * ws0 * value_next[v_hi, s_lo] + wv1 * ws1 * value_next[v_hi, s_hi])
        best = np.minimum(best, dt + price * e_in + interp)
    return best


@dataclass
class CudaBackward:
    """optimal._Compiled.backward on the GPU: the same values, one kernel launch per track point."""
    v_grid: np.ndarray
    s_grid: np.ndarray
    cmd: np.ndarray
    env: np.ndarray
    straight: np.ndarray
    k: np.ndarray
    terminal: bool
    miss_penalty: float
    deploy_cap: np.ndarray | None = None
    block: int = 128

    def __post_init__(self):
        import cupy as cp

        self.cp = cp
        self.kernel = cp.RawKernel(KERNEL, "dp_step")
        self.d_cmd, self.d_k = cp.asarray(self.cmd), cp.asarray(self.k)
        self.d_v, self.d_s = cp.asarray(self.v_grid), cp.asarray(self.s_grid)

    def __call__(self, soc0: float, price: float):
        cp = self.cp
        nv, ns, nu = len(self.v_grid), len(self.s_grid), len(self.cmd)
        short = cp.maximum(0.0, soc0 - cp.asarray(self.s_grid))[None, :] * cp.ones((nv, 1))
        v_end = self.miss_penalty * short if self.terminal else cp.zeros((nv, ns))
        values = cp.empty((len(self.env), nv, ns))
        grid, nxt = ((ns + self.block - 1) // self.block, nv), v_end
        cap = np.full(len(self.env), np.inf) if self.deploy_cap is None else self.deploy_cap
        v0, dv, ds_soc = float(self.v_grid[0]), float(self.v_grid[1] - self.v_grid[0]), float(self.s_grid[1] - self.s_grid[0])
        for i in range(len(self.env) - 1, -1, -1):
            self.kernel(grid, (self.block,), (nxt, values[i], self.d_cmd, self.d_v, self.d_s, np.int32(nv), np.int32(ns), np.int32(nu),
                                              v0, dv, ds_soc, float(self.env[i]), np.int32(bool(self.straight[i])),
                                              float(price), self.d_k, float(cap[i])))
            nxt = values[i]
        return values, v_end
