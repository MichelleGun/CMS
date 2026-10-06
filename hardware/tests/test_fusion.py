"""Filter + estimator on synthetic trajectories: accuracy, dropouts, latency, yaw health, out-of-sequence."""
import math

import numpy as np

from common.config import load_config
from common.fake_tello import FakeOverheadCamera, FakeTello, SimClock
from state_estimation.est_types import CameraDetection, DEGRADED, LOST, OK
from state_estimation.estimator import StateEstimator
from state_estimation.fusion_kf import FusionKF, YawEstimator
from state_estimation.latency import LatencyTracker

CFG = load_config()
E = CFG["estimator"]


def test_kf_fuses_and_beats_dead_reckoning_with_biased_velocity():
    rng = np.random.default_rng(0)
    kf = FusionKF(E, 0.0)
    kf.update_height(0.0, 1.0)
    kf.update_camera(0.0, (0.0, 0.0))
    dt, t, true = 0.1, 0.0, np.zeros(2)
    dead = np.zeros(2)
    vtrue = np.array([0.2, 0.1])
    for i in range(300):
        t += dt
        true += vtrue * dt
        v_meas = vtrue + 0.05 + rng.normal(0, 0.02, 2)           # biased velocity sensor
        dead += v_meas * dt
        kf.update_velocity(t, [v_meas[0], v_meas[1], 0.0])
        kf.update_height(t, 1.0)
        if i % 3 == 0:
            kf.update_camera(t, true + rng.normal(0, 0.02, 2))
    err_fused = np.linalg.norm(kf.x[0:2] - true)
    err_dead = np.linalg.norm(dead - true)
    assert err_fused < 0.08
    assert err_dead > 3 * err_fused


def test_kf_dropout_grows_uncertainty_then_recovers():
    kf = FusionKF(E, 0.0)
    kf.update_height(0, 1.0)
    kf.update_camera(0.0, (0, 0))
    t = 0.0
    for i in range(20):
        t += 0.1
        kf.update_velocity(t, [0, 0, 0])
        kf.update_camera(t, (0, 0))
    std0 = kf.pos_std(t)
    for i in range(60):                                          # 6 s without camera
        t += 0.1
        kf.update_velocity(t, [0, 0, 0])
    std1 = kf.pos_std(t)
    assert std1 > 3 * std0
    kf.update_camera(t + 0.1, (0.05, 0.0))
    assert kf.pos_std(t + 0.1) < std1


def test_gate_rejects_outlier_but_reacquires_after_persistent_offset():
    kf = FusionKF(E, 0.0)
    kf.update_height(0, 1.0)
    kf.update_camera(0.0, (0, 0))
    t = 0.0
    for _ in range(30):
        t += 0.1
        kf.update_velocity(t, [0, 0, 0])
        kf.update_camera(t, (0, 0))
    t += 0.1
    assert kf.update_camera(t, (2.0, 2.0)) is False              # single bad detection (another marker)
    assert np.linalg.norm(kf.x[0:2]) < 0.1
    for _ in range(8):                                           # the truth really did move
        t += 0.1
        kf.update_camera(t, (0.8, 0.0))
    assert abs(kf.x[0] - 0.8) < 0.15


def test_out_of_sequence_camera_matches_in_order_result():
    def build(order):
        kf = FusionKF(E, 0.0)
        kf.update_height(0, 1.0)
        kf.update_camera(0.0, (0, 0))
        ms = []
        for i in range(1, 11):
            t = i * 0.1
            ms.append(("v", t, [0.2, 0.0, 0.0]))
        ms.append(("c", 0.55, (0.11, 0.0)))
        if order == "late":
            ms = sorted(ms, key=lambda m: (m[0] == "c", m[1]))   # camera arrives after everything else
        else:
            ms = sorted(ms, key=lambda m: m[1])
        for k, t, z in ms:
            kf.update_velocity(t, z) if k == "v" else kf.update_camera(t, z)
        return kf.x.copy()

    assert np.allclose(build("late"), build("inorder"), atol=1e-6)


def test_yaw_offset_rate_limited_and_health_flags():
    y = YawEstimator(E)
    y.update_tello(0.0, 0.0)
    y.update_camera(0.0, math.radians(30))                       # initialises offset
    assert abs(y.offset - math.radians(30)) < 1e-9 and y.health == OK
    # camera suddenly disagrees by 40 deg -> LOST, offset untouched
    y.update_camera(0.05, math.radians(70))
    assert y.health == LOST and abs(y.offset - math.radians(30)) < 1e-9
    # slow genuine drift is tracked but never faster than 5 deg/s
    y2 = YawEstimator(E)
    y2.update_tello(0.0, 0.0)
    y2.update_camera(0.0, 0.0)
    t = 0.0
    for i in range(100):
        t += 0.1
        y2.update_tello(t, 0.0)
        y2.update_camera(t, math.radians(8.0))                    # 8 deg error: inside the 15 deg tolerance
    assert 0 < math.degrees(y2.offset) <= 5.0 * t + 1e-6
    assert y2.health == OK


def test_yaw_degraded_freezes_offset():
    y = YawEstimator(E)
    y.update_tello(0.0, 0.0)
    y.update_camera(0.0, 0.0)
    t = 0.0
    for _ in range(40):                                           # persistent 20 deg disagreement (> 15, < 30)
        t += 0.1
        y.update_tello(t, 0.0)
        y.update_camera(t, math.radians(20))
    assert y.health == DEGRADED and abs(y.offset) < 1e-9


def test_latency_tracker_converges_near_true_lag():
    rng = np.random.default_rng(1)
    lt = LatencyTracker(E)
    true_lag = 0.12
    t = 0.0
    for i in range(30 * 20):
        t += 1 / 30
        vx = 0.25 * math.sin(2 * math.pi * 0.35 * t) + 0.1 * math.sin(2 * math.pi * 0.9 * t)
        vy = 0.2 * math.cos(2 * math.pi * 0.3 * t)
        lt.add_tello_velocity(t, vx + rng.normal(0, 0.01), vy + rng.normal(0, 0.01))
        # camera position integrates the true velocity, observed `true_lag` late
        tc = t - true_lag
        px = 0.25 / (2 * math.pi * 0.35) * (1 - math.cos(2 * math.pi * 0.35 * tc)) + \
             0.1 / (2 * math.pi * 0.9) * (1 - math.cos(2 * math.pi * 0.9 * tc))
        py = 0.2 / (2 * math.pi * 0.3) * math.sin(2 * math.pi * 0.3 * tc)
        lt.add_camera(t, px + rng.normal(0, 0.003), py + rng.normal(0, 0.003))
        lt.maybe_update(t)
    assert abs(lt.latency - true_lag) < 0.035, lt.latency
    assert lt.sigma < 0.06


def _fly(profile, occlusions=(), seed=3, seconds=14, manual=None):
    """FakeTello + camera + estimator, hand-flown through takeoff then a fixed velocity pattern."""
    clock = SimClock()
    fake = FakeTello(CFG, profile, seed=seed, clock=clock, cam_occlusions=occlusions)
    cam = FakeOverheadCamera(fake, seed=seed + 1)
    est = StateEstimator(CFG, t0=clock.now())
    fake.connect()
    errs, healths = [], []
    for step in range(int(seconds * 30)):
        now_pre = clock.now()
        if step == 15:
            fake.takeoff()
        if fake.phase == "hover" and step > 150:
            t = step / 30
            fake.send_rc(int(15 * math.sin(t)), int(15 * math.cos(0.7 * t)), 0, 0)
        fake.advance(1 / 30)
        now = clock.now()
        est.on_raw_telemetry(fake.poll_telemetry(), now)
        for d in cam.poll(now):
            est.on_camera(d, now)
        e = est.estimate(now)
        if fake.phase == "hover":
            errs.append(np.linalg.norm(e.pos - fake.pos))
            healths.append(e.pos_health)
    return errs, healths, est, fake


def test_estimator_tracks_fake_tello_typical():
    errs, healths, est, fake = _fly("typical")
    assert len(errs) > 100
    assert np.percentile(errs, 90) < 0.12
    assert "LOST" not in healths


def test_estimator_survives_camera_blackout_then_recovers():
    errs, healths, est, fake = _fly("typical", occlusions=((8.0, 10.0),), seconds=16)
    assert "DEGRADED" in healths                                   # noticed the blackout
    assert "LOST" not in healths                                   # but dead-reckoning bridged 2 s
    assert errs[-1] < 0.15                                         # camera back, error recovered
    assert healths[-1] == "OK"


def test_estimator_harsh_profile_still_bounded():
    errs, healths, est, fake = _fly("harsh", seconds=16)
    assert np.percentile(errs, 90) < 0.25
