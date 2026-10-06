"""Offline tools driven by a log recorded from a FakeTello flight: tune_replay, sim_vs_real, hoop aggregation."""
import json

import numpy as np
import pytest

from calibration.hoop_locator import aggregate_hoop
from common.config import load_config
from common.fake_tello import FakeOverheadCamera, FakeTello, SimClock
from common.flight_session import FlightSession
from common.logger import FlightLogger
from state_estimation import tune_replay


@pytest.fixture(scope="module")
def flight_log(tmp_path_factory):
    """A real flight through FlightSession with logging on: takeoff, wander, land."""
    cfg = load_config()
    d = tmp_path_factory.mktemp("flights")
    clock = SimClock()
    fake = FakeTello(cfg, "typical", seed=7, clock=clock)
    cam = FakeOverheadCamera(fake, seed=8)
    fake.connect()
    logger = FlightLogger("t", directory=d)
    s = FlightSession(cfg, fake, cam, hoop=np.array([0.4, 0.2, 1.0]), logger=logger)
    for _ in range(60):
        s.tick()
    assert s.request_takeoff() == []
    for _ in range(300):
        if s.tick().phase == "flying":
            break
    rng = np.random.default_rng(0)
    a = np.zeros(4)
    for i in range(420):
        if i % 45 == 0:
            a = np.array([*rng.uniform(-1, 1, 3) * [1, 1, 0.3], 0.8])
        s.tick(manual_action=a)
    s.request_land()
    for _ in range(400):
        if s.tick().phase == "landed":
            break
    s.close()
    return logger.path


def test_log_holds_everything_needed_for_replay(flight_log):
    ticks = tune_replay.load_ticks(flight_log)
    assert any(t["tele"] for t in ticks) and any(t["cam"] for t in ticks)
    fl = [t for t in ticks if t["phase"] == "flying" and t["obs"]]
    assert fl and len(fl[0]["obs"]) == 75 and len(fl[0]["executed"]) == 4


def test_tune_replay_scores_and_ranks(flight_log):
    cfg = load_config()
    ticks = tune_replay.load_ticks(flight_log)
    cur, n = tune_replay.replay_score(ticks, cfg)
    assert n > 50 and 0.0 < cur < 0.15                                    # a sane filter predicts held-out frames within cm
    bad = json.loads(json.dumps(cfg))
    bad["estimator"].update(q_acc=0.001, sigma_vel=0.001, sigma_cam_xy=0.5)    # trusts biased velocity, ignores camera
    worse, _ = tune_replay.replay_score(ticks, bad)
    assert worse > cur
    rows = tune_replay.tune(ticks, cfg, tune_replay.GRIDS["small"])
    assert len(rows) == 12 and rows[0][0] <= rows[-1][0]


def test_sim_vs_real_reports_a_finite_gap(flight_log):
    pytest.importorskip("pybullet")
    pytest.importorskip("gym_pybullet_drones")
    from sim_check import sim_vs_real
    r = sim_vs_real.compare(flight_log, plot=True)
    assert r["steps"] > 100 and np.isfinite(r["rms_m"]) and r["rms_m"] < 3.0
    assert r["plot"].endswith(".png") and __import__("os").path.getsize(r["plot"]) > 1000


def test_hoop_aggregate_accepts_clean_rejects_noisy_and_out_of_range():
    rng = np.random.default_rng(0)
    clean = np.array([0.30, -0.20]) + rng.normal(0, 0.003, (30, 2))
    xyz, std, problems = aggregate_hoop(clean, 1.0)
    assert xyz is not None and np.allclose(xyz, [0.30, -0.20, 1.0], atol=0.01) and std < 0.01 and not problems
    noisy = np.array([0.3, -0.2]) + rng.normal(0, 0.03, (30, 2))
    assert aggregate_hoop(noisy, 1.0, max_std=0.02)[0] is None
    assert aggregate_hoop(np.array([1.3, 0.0]) + rng.normal(0, 0.002, (30, 2)), 1.0)[0] is None       # outside +-0.75
    assert aggregate_hoop(clean, 1.9)[0] is None                                                       # z out of range
    assert aggregate_hoop(clean[:4], 1.0)[0] is None                                                   # too few frames
    # a few stray detections (another marker) are discarded, not averaged in
    stray = np.vstack([clean, [[0.9, 0.9], [-0.7, 0.1]]])
    xyz2, _, _ = aggregate_hoop(stray, 1.0)
    assert np.allclose(xyz2[:2], [0.30, -0.20], atol=0.01)
