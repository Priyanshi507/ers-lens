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
             for _, lap in obs.groupby("lap")]
    return float(np.mean(drops[1:]))


def test_derate_detector_matches_ground_truth():
    from erslens.policies import ClipEndOfStraight

    none = _detected_drop(StraightsOnly())
    mild = _detected_drop(ClipEndOfStraight(clip_from=0.85, clip_w=150e3))
    hard = _detected_drop(ClipEndOfStraight(clip_from=0.5, clip_w=350e3))
    assert none < 2.0
    assert none < mild < hard
    assert hard > 20.0
