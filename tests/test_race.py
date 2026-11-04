import json
import sys

import numpy as np

from erslens.race import LapTable, lap_table, race_plan

sys.path.insert(0, "scripts")
import race_plan as script  # noqa: E402
from race_plan import verdicts  # noqa: E402


def table(lap):
    lap = np.asarray(lap, float)
    z = np.zeros_like(lap)
    return LapTable(np.linspace(0, 4e6, len(lap)), lap, z, z, z, 8.5e6, z.astype(int))


def test_race_plan_spends_the_battery_at_the_end_when_laps_are_identical():
    # Charging costs more than discharging saves, so the whole battery is spent once (when is a tie).
    t = table([[80.0, 81.0, 82.0], [79.7, 80.0, 81.0], [79.3, 79.7, 80.0]])
    p = race_plan(t, 5, 2)
    assert p.levels[-1] == 0
    assert np.isclose(p.race_s, 4 * 80.0 + 79.3) and np.isclose(p.gain_s, 0.7)


def test_race_plan_matches_brute_force():
    rng = np.random.default_rng(3)
    lap = 80 + rng.uniform(-1, 1, (4, 4))
    best = min((sum(lap[a, b] for a, b in zip((1,) + s, s)), s)
               for s in np.ndindex(4, 4, 4))
    assert np.isclose(race_plan(table(lap), 3, 1).race_s, best[0])


def test_lap_table_respects_the_recharge_limit_and_end_charge():
    from erslens.params import load_car_params
    from erslens.track import demo_circuit

    car = load_car_params("configs/car_2026_q_low.yaml")
    t = lap_table(demo_circuit(), car, np.array([2e6, 3e6]), prices=(0.0, 3e-7), dv=0.25, n_soc=81)
    found = np.isfinite(t.lap_s)
    assert found.any()
    assert np.all(t.harvest_j[found] <= t.budget_j * (1 + 1e-9))
    assert np.all(t.end_j[found] >= t.levels_j[None, :].repeat(2, 0)[found] - 5e3)


def test_verdicts_need_the_check_first():
    good = {"check_ok": True, "best_minus_simple_s": 0.02, "gain_s": 0.4, "finish_level_mj": 0.0}
    assert verdicts([good, {**good, "gain_s": 1.5}]) == {"W1": [2, 2, True], "W2": [1, 2, False], "W3": [2, 2, True]}
    assert verdicts([good, {**good, "check_ok": False}]) == {"check": [1, 2, False]}


def test_plans_step_writes_csv_and_json(tmp_path, monkeypatch):
    n = len(script.LEVELS_J)
    lap = np.array([[80 + 0.25 * (j - i) / 2 if j <= i + 2 else np.inf for j in range(n)] for i in range(n)])
    (tmp_path / "tables").mkdir()
    z = np.zeros((n, n))
    np.savez(tmp_path / "tables" / "Australia.npz", levels_j=script.LEVELS_J, lap_s=lap, harvest_j=z, end_j=z,
             price=z, budget_j=7e6, unconverged=z.astype(int), single_lap_s=80.01)
    monkeypatch.setattr(script, "race_laps", lambda year, date: 58)
    script.plans({"Australia": "2026-03-08"}, tmp_path)
    assert json.loads((tmp_path / "verdicts.json").read_text())["W3"] == [1, 1, True]
    assert (tmp_path / "plans.csv").read_text().startswith("event,laps")


def test_the_check_covers_every_level(tmp_path, monkeypatch):
    n = len(script.LEVELS_J)
    lap = np.array([[80 + 0.25 * (j - i) / 2 if j <= i + 2 else np.inf for j in range(n)] for i in range(n)])
    single = np.diag(lap).copy()
    single[1] -= 0.06            # one level's single-lap solve is faster than the table's lap there
    (tmp_path / "tables").mkdir()
    z = np.zeros((n, n))
    np.savez(tmp_path / "tables" / "Australia.npz", levels_j=script.LEVELS_J, lap_s=lap, harvest_j=z, end_j=z,
             price=z, budget_j=7e6, unconverged=z.astype(int), single_lap_s=single, single_ok=np.ones(n, bool))
    monkeypatch.setattr(script, "race_laps", lambda year, date: 58)
    script.plans({"Australia": "2026-03-08"}, tmp_path)
    assert json.loads((tmp_path / "verdicts.json").read_text()) == {"check": [0, 1, False]}
