"""Whole-race energy plans: which battery charge to carry from lap to lap (research log, 2026-11-02).

Two levels. Lap level: for every pair of start and end charge on a ladder of levels, the fastest
lap that starts at one and ends at least at the other, with the per-lap recharge limit enforced.
One backward pass of the DP, aimed at an end charge, serves every start charge at once, so the
table costs one backward pass per (end level, harvest price) and a cheap forward rollout per
start level. Every lap in the table is a simulator replay, so its time and energy are measured,
not predicted. Race level: a small DP over laps picks the charge to carry between laps.

Prior work on race-level energy allocation for the 2014-2025 Formula 1 power unit: Duhr,
Buccheri, Balerna, Cerofolini and Onder, IEEE Transactions on Vehicular Technology 72(6), 2023.
"""
from dataclasses import dataclass

import numpy as np

from .optimal import _Compiled, dp_car, grid_with, start_at_slowest_point
from .strategy import CONVERGED_S

END_TOL_J = 5e3
PRICES = (0.0, 2e-8, 5e-8, 1e-7, 1.5e-7, 2e-7, 3e-7, 4e-7, 6e-7, 8e-7, 1.2e-6, 2e-6)


@dataclass
class LapTable:
    levels_j: np.ndarray
    lap_s: np.ndarray        # [start, end]: fastest feasible lap, inf where none was found
    harvest_j: np.ndarray
    end_j: np.ndarray
    price: np.ndarray
    budget_j: float
    unconverged: np.ndarray  # [start, end]: candidates dropped because the DP and its replay disagreed


def lap_table(track, car, levels_j, prices=PRICES, dv: float = 0.125, n_soc: int = 321) -> LapTable:
    rc = dp_car(car)
    tr = start_at_slowest_point(track, rc)
    c = _Compiled(tr, rc, grid_with(rc, dv, n_soc), True)
    n = len(levels_j)
    shape = (n, n)
    lap, harvest, end, price_used = (np.full(shape, np.inf), np.full(shape, np.nan),
                                     np.full(shape, np.nan), np.full(shape, np.nan))
    budget = rc.harvest_per_lap_j
    dropped = np.zeros(shape, dtype=int)
    for j, target in enumerate(levels_j):
        for price in prices:
            values, v_end = c.backward(float(target), float(price))
            for i, start in enumerate(levels_j):
                cmd, start_cost = c.forward(values, v_end, float(start), float(price))
                out = {k: np.asarray(a) for k, a in c.replay(cmd, float(start)).items()}
                h = float((out["harvest"] * out["dt"]).sum() * rc.eta_harvest)
                e = float(out["soc"][-1] + (rc.eta_harvest * out["harvest"][-1]
                                            - out["deploy"][-1] / rc.eta_deploy) * out["dt"][-1])
                t = float(out["dt"].sum())
                if abs(float(start_cost) - price * h - t) > CONVERGED_S:
                    dropped[i, j] += 1
                    continue
                if h <= budget * (1 + 1e-9) and e >= target - END_TOL_J and t < lap[i, j]:
                    lap[i, j], harvest[i, j], end[i, j], price_used[i, j] = t, h, e, price
            del values
    return LapTable(np.asarray(levels_j, float), lap, harvest, end, price_used, budget, dropped)


@dataclass
class RacePlan:
    levels: list           # index of the charge level at the start of each lap, then at the finish
    race_s: float
    neutral_s: float       # every lap from the start level back to it
    start: int

    @property
    def gain_s(self) -> float:
        return self.neutral_s - self.race_s


def race_plan(table: LapTable, n_laps: int, start: int) -> RacePlan:
    """Fastest sequence of lap-start charges over n_laps, starting at level `start`, ending free."""
    lap = table.lap_s
    n = len(table.levels_j)
    cost = np.zeros(n)
    choice = np.zeros((n_laps, n), dtype=int)
    for k in range(n_laps - 1, -1, -1):
        total = lap + cost[None, :]
        choice[k] = np.argmin(total, axis=1)
        cost = total[np.arange(n), choice[k]]
    levels, s = [start], start
    for k in range(n_laps):
        s = int(choice[k, s])
        levels.append(s)
    return RacePlan(levels, float(cost[start]), float(n_laps * lap[start, start]), start)
