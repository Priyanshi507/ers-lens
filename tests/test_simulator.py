import numpy as np
import pytest

from erslens.params import CarParams
from erslens.physics import braking_envelope, mguk_power_cap
from erslens.policies import FlatOut, SocTarget, StraightsOnly
from erslens.simulate import lap_times, simulate
from erslens.synth import generate_dataset, sensor_model
from erslens.track import demo_circuit, track_from_reference

CAR = CarParams()
TRACK = demo_circuit()
POLICIES = [FlatOut(), StraightsOnly(), SocTarget()]


@pytest.fixture(scope="module", params=POLICIES, ids=lambda p: p.name)
def run(request):
    return simulate(TRACK, CAR, request.param, n_laps=3)


def test_soc_stays_in_window(run):
    assert run["soc_j"].between(0, CAR.es_capacity_j).all()


def test_harvest_limit_per_lap(run):
    assert (run.groupby("lap")["harvested_lap_j"].max() <= CAR.harvest_per_lap_j + 1).all()


def test_speed_respects_braking_envelope(run):
    env = braking_envelope(TRACK, CAR) * 3.6
    assert (run["speed_kmh"].to_numpy() <= env[run["idx"].to_numpy()] + 1e-6).all()


def test_mguk_cap_respected(run):
    caps = np.array([mguk_power_cap(v / 3.6, CAR) for v in run["speed_kmh"]])
    assert (run["deploy_w"] <= caps + 1).all()
    assert (run["harvest_w"] <= CAR.mguk_power_w + 1).all()


def test_no_deploy_while_braking(run):
    assert (run.loc[run["brake"], "deploy_w"] == 0).all()


def test_lap_times_plausible(run):
    avg_kmh = TRACK.length_m / lap_times(run) * 3.6
    assert avg_kmh.between(150, 280).all()


def test_mguk_taper():
    assert mguk_power_cap(250 / 3.6, CAR) == CAR.mguk_power_w
    assert mguk_power_cap(400 / 3.6, CAR) == 0.0
    assert 0 < mguk_power_cap(320 / 3.6, CAR) < CAR.mguk_power_w


def test_flat_out_drains_battery_more():
    flat = simulate(TRACK, CAR, FlatOut(), n_laps=3)
    managed = simulate(TRACK, CAR, StraightsOnly(), n_laps=3)
    assert flat["soc_j"].iloc[-1] < managed["soc_j"].iloc[-1]


def test_reference_track_marks_full_throttle_as_open():
    d = np.linspace(0, 1000, 201)
    speed = np.where(d < 600, 300.0, 120.0)
    throttle = np.where(d < 600, 100.0, 40.0)
    tr = track_from_reference("t", d, speed, throttle, ds=5.0, min_straight_m=200)
    assert np.isinf(tr.v_limit_ms[:100]).all()
    assert np.isfinite(tr.v_limit_ms[-10:]).all()
    assert tr.straight_mode[:100].all()


def test_sensor_model_shape():
    rng = np.random.default_rng(0)
    sim = simulate(TRACK, CAR, SocTarget(), n_laps=2)
    obs = sensor_model(sim, CAR, rng)
    rate = 1 / np.median(np.diff(obs["time_s"]))
    assert 3.9 < rate < 4.1
    assert obs["label_soc_frac"].between(0, 1).all()
    assert (obs["speed_kmh"] == obs["speed_kmh"].round()).all()


def test_generate_dataset(tmp_path):
    data = generate_dataset(TRACK, CAR, n_episodes=3, out_dir=tmp_path, laps_per_episode=2)
    assert data["episode"].nunique() == 3
    assert (tmp_path / "demo_synthetic.parquet").exists()


def _detected_drop(policy):
    import sys
    sys.path.insert(0, "scripts")
    from derate_analysis import lap_metrics, longest_flat_out_run

    start, end = longest_flat_out_run(TRACK)
    sim = simulate(TRACK, CAR, policy, n_laps=4, soc0_frac=0.5)
    obs = sensor_model(sim, CAR, np.random.default_rng(0))
    obs["distance_m"] -= obs["lap"] * TRACK.length_m
    drops = [lap_metrics(lap.reset_index(drop=True), start, end)["flat_out_drop_kmh"]
             for lap_no, lap in obs.groupby("lap") if lap_no > 0]
    return float(np.mean(drops))


def test_derate_detector_matches_ground_truth():
    from erslens.policies import ClipEndOfStraight

    none = _detected_drop(StraightsOnly())
    mild = _detected_drop(ClipEndOfStraight(clip_from=0.85, clip_w=150e3))
    hard = _detected_drop(ClipEndOfStraight(clip_from=0.5, clip_w=350e3))
    assert none < 2.0
    assert none < mild < hard
    assert hard > 20.0


def test_lap_with_telemetry_gap_is_rejected():
    import sys
    sys.path.insert(0, "scripts")
    import pandas as pd
    from derate_analysis import lap_metrics

    dist = np.arange(0, 1200, 20.0)
    brake = dist > 1000
    speed = np.where(brake, 300 - (dist - 1000) * 1.2, 200 + dist / 10)
    lap = pd.DataFrame({"distance_m": dist, "speed_kmh": speed,
                        "throttle": np.where(brake, 0.0, 100.0), "brake": brake})
    assert lap_metrics(lap, 0, 1000) is not None
    gappy = lap.drop(index=range(30, 36)).reset_index(drop=True)
    assert lap_metrics(gappy, 0, 1000) is None


def test_stale_speed_and_truncated_laps_are_rejected():
    import sys
    sys.path.insert(0, "scripts")
    import pandas as pd
    from derate_analysis import lap_quality

    dist = np.arange(0, 1200, 20.0)
    brake = dist > 1000
    speed = np.where(brake, 320 - (dist - 1000) * 1.2, 150 + dist * 0.17)
    lap = pd.DataFrame({"distance_m": dist, "speed_kmh": speed,
                        "throttle": np.where(brake, 0.0, 100.0), "brake": brake})
    assert lap_quality(lap, 0, 1000)[1] == "ok"

    stale = lap.copy()
    stale.loc[10:14, "speed_kmh"] = stale.loc[10, "speed_kmh"]
    assert lap_quality(stale, 0, 1000)[1] == "stale_speed"

    truncated = lap[lap["distance_m"] < 900].reset_index(drop=True)
    assert lap_quality(truncated, 0, 1000)[1] == "data_ends_before_braking"


def test_single_throttle_blip_does_not_hide_speed_loss():
    import sys
    sys.path.insert(0, "scripts")
    import pandas as pd
    from derate_analysis import lap_quality

    dist = np.arange(0, 1200, 20.0)
    brake = dist > 1000
    speed = np.where(dist < 500, 200 + dist / 5, 300 - (dist - 500) / 25)
    speed = np.where(brake, 280 - (dist - 1000) * 1.2, speed)
    throttle = np.where(brake, 0.0, 100.0)
    throttle[30] = 96.0
    lap = pd.DataFrame({"distance_m": dist, "speed_kmh": speed, "throttle": throttle, "brake": brake})
    metrics, reason = lap_quality(lap, 0, 1000)
    assert reason == "ok"
    assert metrics["flat_out_drop_kmh"] > 15


def test_frozen_lap_without_braking_is_rejected():
    import sys
    sys.path.insert(0, "scripts")
    import pandas as pd
    from derate_analysis import lap_quality

    dist = np.arange(0, 1200, 20.0)
    brake = dist > 1000
    speed = np.where(dist < 100, 142.0, 291.0)
    throttle = np.where(brake, 0.0, 100.0)
    lap = pd.DataFrame({"distance_m": dist, "speed_kmh": speed, "throttle": throttle, "brake": brake})
    assert lap_quality(lap, 0, 1000)[1] != "ok"


def test_gap_between_flat_run_and_braking_is_rejected():
    import sys
    sys.path.insert(0, "scripts")
    import pandas as pd
    from derate_analysis import lap_quality

    dist = np.concatenate([np.arange(0, 1080, 20.0), [1090.0], np.arange(1390, 1500, 20.0)])
    brake = dist >= 1390
    speed = np.where(brake, 325 - (dist - 1390) * 2.0, np.minimum(200 + dist / 4, 325.0))
    throttle = np.where(brake, 0.0, np.where(dist >= 1090, 60.0, 100.0))
    lap = pd.DataFrame({"distance_m": dist, "speed_kmh": speed, "throttle": throttle, "brake": brake})
    assert lap_quality(lap, 0, 1400)[1] == "telemetry_gap"


def test_frozen_speed_after_throttle_closes_is_rejected():
    import sys
    sys.path.insert(0, "scripts")
    import pandas as pd
    from derate_analysis import lap_quality

    dist = np.arange(0, 1500, 20.0)
    speed = np.minimum(200 + dist / 4, 325.0)
    speed = np.where(dist >= 1400, 325 - (dist - 1380) * 2.5, speed)
    throttle = np.where(dist >= 1080, 0.0, 100.0)
    speed[(dist >= 1080) & (dist < 1400)] = 325.0
    lap = pd.DataFrame({"distance_m": dist, "speed_kmh": speed, "throttle": throttle,
                        "brake": dist >= 1400})
    assert lap_quality(lap, 0, 1400)[1] != "ok"


def test_race_summary_measures_clip_time():
    import sys
    sys.path.insert(0, "scripts")
    import pandas as pd
    from derate_analysis import analyse_driver, longest_flat_out_run
    from erslens.policies import ClipEndOfStraight
    from multi_race import race_summary

    start, end = longest_flat_out_run(TRACK)
    rows = []
    for name, pol in [("clip", ClipEndOfStraight(clip_from=0.5, clip_w=350e3)), ("none", StraightsOnly())]:
        sim = simulate(TRACK, CAR, pol, n_laps=6, soc0_frac=0.5)
        obs = sensor_model(sim, CAR, np.random.default_rng(1))
        obs["lap"] += 1
        obs["distance_m"] -= (obs["lap"] - 1) * TRACK.length_m
        obs["time_s"] -= obs.groupby("lap")["time_s"].transform("min")
        obs["pit_in"] = obs["pit_out"] = False
        res = analyse_driver(obs, start, end)
        res.insert(0, "driver", name)
        rows.append(res)
    clip, none = (race_summary(r, len(r), end - start) for r in rows)
    assert none["clip_s_mean"] < 0.5
    # Clipping from half-way down a ~1.1 km straight lasts several seconds.
    assert 3.0 < clip["clip_s_mean"] < 10.0
    assert clip["clip_s_lo"] <= clip["clip_s_mean"] <= clip["clip_s_hi"]


def test_permutation_p_is_exact_and_bounded():
    import sys
    sys.path.insert(0, "scripts")
    from multi_race import permutation_p

    assert permutation_p([0.9, 0.8, 0.85], [0.1, 0.2, 0.15, 0.12]) == 1 / 35
    assert permutation_p([0.1, 0.2, 0.15], [0.9, 0.8, 0.85, 0.7]) == 1.0


def test_electric_power_swing_matches_ground_truth():
    import sys
    sys.path.insert(0, "scripts")
    from energy_validate import run, summarise

    s = summarise(run(n_cars=8, laps=4, seed=3))
    assert s["false_positive_laps"] == 0
    assert s["corr"] > 0.9  # 0.99 on the full 40-car validation; small sample here
    assert s["mae_kw"] < 40


def test_swing_is_none_without_clipping():
    from erslens.energy import electric_power_swing
    import pandas as pd

    t = np.arange(0, 12, 0.25)
    speed = np.minimum(200 + 12 * t, 320.0)
    lap = pd.DataFrame({"time_s": t, "distance_m": np.cumsum(speed / 3.6 * 0.25),
                        "speed_kmh": speed, "throttle": 100.0, "brake": False})
    assert electric_power_swing(lap, 0, 1e6, 800) is None


def test_elevation_correction_removes_gravity_bias():
    """Constant electric power on a level-then-uphill straight: the true swing is zero."""
    import pandas as pd
    from erslens.energy import electric_power_swing

    m, p, cda, grade = 800.0, 450e3, 1.0, 0.10
    dt, v, s, rows = 0.01, 85.0, 0.0, []
    for k in range(int(16 / dt)):
        slope = grade if s >= 600 else 0.0
        a = (p / v - 0.5 * 1.2 * cda * v * v - m * 9.81 * slope) / m
        if k % 25 == 0:
            rows.append((k * dt, s, v * 3.6))
        v += a * dt
        s += v * dt
    t, dist, kmh = map(np.array, zip(*rows))
    lap = pd.DataFrame({"time_s": t, "distance_m": dist, "speed_kmh": kmh,
                        "throttle": 100.0, "brake": False})
    grid = np.arange(0, 2000, 5.0)
    hill = (grid, np.where(grid >= 600, grade * (grid - 600), 0.0))
    level = electric_power_swing(lap, 0, 1e6, m)
    corrected = electric_power_swing(lap, 0, 1e6, m, elevation=hill)
    assert level["swing_w"] > 30e3
    assert abs(corrected["swing_w"]) < 5e3
