import dataclasses

import numpy as np
import pytest

from erslens.optimal import solve
from erslens.params import CarParams
from erslens.sectors import SECTOR_CAP_W, sector_events, sector_mask, with_fia_sectors
from erslens.track import toy_track

TRACK = toy_track([("straight", 600, None), ("corner", 100, 90),
                   ("straight", 300, None), ("corner", 80, 140)], name="short")
CAR = CarParams(superclip_max_w=350e3)


def with_cap(track, cap):
    return dataclasses.replace(track, deploy_cap_w=np.full(track.n, cap))


@pytest.fixture(scope="module")
def base():
    return solve(TRACK, CAR, soc0_fracs=(0.5,))


def test_no_sector_limit_changes_nothing(base):
    same = solve(with_cap(TRACK, np.inf), CAR, soc0_fracs=(0.5,))
    assert same.lap_time_sim_s == base.lap_time_sim_s
    assert np.array_equal(same.cmd_w, base.cmd_w)


def test_a_sector_everywhere_is_the_global_cap():
    everywhere = solve(with_cap(TRACK, SECTOR_CAP_W), CAR, soc0_fracs=(0.5,))
    global_cap = solve(TRACK, CAR.with_(deploy_max_w=SECTOR_CAP_W), soc0_fracs=(0.5,))
    assert everywhere.lap_time_sim_s == global_cap.lap_time_sim_s


def test_partial_sectors_cost_time_between_none_and_everywhere(base):
    half = dataclasses.replace(TRACK, deploy_cap_w=np.where(np.arange(TRACK.n) < TRACK.n // 2, SECTOR_CAP_W, np.inf))
    t_half = solve(half, CAR, soc0_fracs=(0.5,)).lap_time_sim_s
    t_all = solve(with_cap(TRACK, SECTOR_CAP_W), CAR, soc0_fracs=(0.5,)).lap_time_sim_s
    assert base.lap_time_sim_s - 1e-6 <= t_half <= t_all + 1e-6


def test_windows_scale_to_the_track_length():
    inside = sector_mask(TRACK, 2 * TRACK.n * TRACK.ds, [[0.0, TRACK.n * TRACK.ds]])
    assert inside[: TRACK.n // 2].all() and not inside[TRACK.n // 2 + 1:].any()


def test_config_windows_are_ordered_and_inside_the_lap():
    for name, e in sector_events().items():
        flat = [x for w in e["sectors"] for x in w]
        assert flat == sorted(flat) and all(0 <= x <= e["centreline_m"] for x in flat), name
        assert e["read"] in ("pdf", "text"), name


def test_sectors_land_on_real_tracks():
    from erslens.ingest import load_track

    for name in [n for n, e in sector_events().items() if e["sectors"]]:
        t = with_fia_sectors(load_track(f"results/tracks/{name}.npz"), name)
        share = np.isfinite(t.deploy_cap_w).mean()
        assert 0.0 < share < 1.0, name


def test_verdicts_count_only_converged_circuits():
    import sys

    import pandas as pd

    sys.path.insert(0, "scripts")
    from sectors_run import verdicts

    rows = []
    for e, (t_none, t_fia, t_all), conv in (("a", (80.0, 80.1, 80.5), True), ("b", (70.0, 70.4, 70.5), True),
                                             ("c", (60.0, 60.1, 60.2), False)):
        for variant, t, i, o in (("none", t_none, 1.0, 2.0), ("fia", t_fia, 0.8, 2.1), ("everywhere", t_all, 0.7, 1.9)):
            rows.append({"event": e, "variant": variant, "lap_s": t, "converged": conv,
                         "deploy_in_sectors_mj": i, "deploy_outside_mj": o})
    v = verdicts(pd.DataFrame(rows))
    assert v["P1"] == (2, 3, False) and v["P2"] == (1, 3, False) and v["P3"] == (2, 3, False)
