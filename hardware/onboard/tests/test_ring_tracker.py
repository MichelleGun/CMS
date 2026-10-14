"""The ring tracker (obs.RingTracker): a dead-reckoned belief of where the ring is while it is out
of view, appended as 4 extra inputs (spec.TRACKER_FIELDS) only when asked for."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))

from onboard import spec  # noqa: E402
from onboard.gym_env import OnboardHoopEnv  # noqa: E402
from onboard.obs import NOMINAL_FWD_MPS, RingTracker, add_tracker, refresh_obs  # noqa: E402

DT = spec.CTRL_DT


def seen(tr, cx=0.0, cy=0.0, rng=1.0, yaw=0.0, h=1.0):
    tr.step(True, cx, cy, rng, yaw, h, 0.0, 0.0, DT)


def blind(tr, yaw=0.0, h=1.0, fb=0.0, lr=0.0, dt=DT):
    tr.step(False, 0.0, 0.0, float("inf"), yaw, h, fb, lr, dt)


def test_nothing_until_the_ring_has_been_seen():
    tr = RingTracker()
    blind(tr, fb=1.0)
    assert not tr.valid and np.all(tr.fields() == 0)


def test_a_detection_straight_ahead_is_straight_ahead():
    tr = RingTracker()
    seen(tr, rng=1.0)
    assert np.allclose(tr.p, [1.0, 0.0, 0.0])
    f = tr.fields()
    assert np.allclose(f[:3], [1.0 / spec.RANGE_NORM, 0, 0]) and f[3] == 1.0


def test_image_offsets_follow_the_pinhole_camera():
    tr = RingTracker()
    seen(tr, cx=0.5, cy=-0.5, rng=2.0)               # right of centre, above it
    k = np.tan(np.radians(spec.CAMERA_HFOV_DEG) / 2)
    assert np.allclose(tr.p, [2.0, 0.5 * k * 2.0, 0.5 * k * 2.0])


def test_turning_clockwise_moves_the_ring_to_the_left():
    tr = RingTracker()
    seen(tr, rng=1.0, yaw=0.0)
    blind(tr, yaw=90.0)                              # clockwise-positive, like the Tello
    assert np.allclose(tr.p, [0.0, -1.0, 0.0], atol=1e-9)


def test_flying_forward_brings_the_ring_closer_and_climbing_lowers_it():
    tr = RingTracker()
    seen(tr, rng=1.0, h=1.0)
    steps = int(round(0.5 / (NOMINAL_FWD_MPS * DT)))
    for _ in range(steps):
        blind(tr, fb=1.0, h=1.2)
    assert tr.p[0] == pytest.approx(1.0 - steps * NOMINAL_FWD_MPS * DT)
    assert tr.p[2] == pytest.approx(-0.2)


def test_a_fresh_sighting_resets_the_drift():
    tr = RingTracker()
    seen(tr, rng=1.0)
    for _ in range(30):
        blind(tr, fb=1.0, lr=0.5, yaw=10.0)
    seen(tr, rng=0.7, yaw=10.0)
    assert np.allclose(tr.p, [0.7, 0.0, 0.0])


def test_the_default_env_is_unchanged_and_the_tracker_is_opt_in():
    base = OnboardHoopEnv(seed=1, profile="measured", laps=0, lap_dir="same", start_mode="zones")
    trk = OnboardHoopEnv(seed=1, profile="measured", laps=0, lap_dir="same", start_mode="zones",
                         ring_tracker=True)
    assert base.observation_space.shape == (spec.OBS_DIM,)
    assert trk.observation_space.shape == (spec.TRACKER_OBS_DIM,)
    o1, _ = base.reset(seed=3)
    o2, _ = trk.reset(seed=3)
    for _ in range(40):
        a = np.array([0.0, 0.3, 0.0, 0.4], np.float32)
        o1, *_ = base.step(a)
        o2, *_ = trk.step(a)
        assert np.allclose(o1, o2[:spec.OBS_DIM])    # the first 37 inputs are exactly as before


def test_replaying_a_recorded_episode_matches_the_live_tracker():
    env = OnboardHoopEnv(seed=2, profile="measured", laps=0, lap_dir="same", start_mode="zones",
                         ring_tracker=True)
    obs, _ = env.reset(seed=11)
    rows, ts = [obs], [env.sim.t]
    rng = np.random.default_rng(0)
    for _ in range(300):
        obs, _, term, trunc, _ = env.step(rng.uniform(-0.6, 0.6, 4).astype(np.float32))
        rows.append(obs)
        ts.append(env.sim.t)
        if term or trunc:
            break
    live = np.array(rows)
    replay = add_tracker(refresh_obs(live[:, :spec.OBS_DIM], np.zeros(len(live), int), np.array(ts)),
                         np.zeros(len(live), int), np.array(ts))
    sl = spec.TRACKER_SLICES["ring_est"]
    # the replay integrates the yaw-rate column instead of reading yaw: a few cm apart at most
    err = np.abs(replay[:, sl] - live[:, sl]) * spec.RANGE_NORM
    assert np.median(err) < 0.02 and np.percentile(err, 95) < 0.10
    assert np.array_equal(replay[:, spec.TRACKER_SLICES["est_valid"]], live[:, spec.TRACKER_SLICES["est_valid"]])


def test_a_widened_37_input_model_acts_exactly_as_before(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from vec_util import load_for_resume
    from onboard.train_onboard import _widen_index
    kw = dict(profile="measured", laps=1, start_mode="ring")
    small = PPO("MlpPolicy", DummyVecEnv([lambda: OnboardHoopEnv(**kw)]), seed=0)
    small.save(tmp_path / "m37.zip")
    big, _ = load_for_resume(PPO, tmp_path / "m37.zip",
                             DummyVecEnv([lambda: OnboardHoopEnv(ring_tracker=True, **kw)]), widen_at=_widen_index)
    assert big.observation_space.shape == (spec.TRACKER_OBS_DIM,)
    o = np.random.default_rng(1).normal(size=(20, spec.TRACKER_OBS_DIM)).astype(np.float32)
    a37, _ = small.predict(o[:, :spec.OBS_DIM], deterministic=True)
    a41, _ = big.predict(o, deterministic=True)
    assert np.allclose(a37, a41, atol=1e-6)


def test_a_37_input_policy_reads_a_41_wide_observation(tmp_path):
    import json
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from onboard.policy import load_policy
    m = PPO("MlpPolicy", DummyVecEnv([lambda: OnboardHoopEnv(profile="measured")]), seed=0)
    m.save(tmp_path / "best_model.zip")
    (tmp_path / "meta.json").write_text(json.dumps({"algo": "ppo", "task": spec.TASK_NAME,
                                                    "yaw_convention": spec.YAW_CONVENTION}))
    fn = load_policy(tmp_path / "best_model.zip")
    o = np.random.default_rng(2).normal(size=spec.TRACKER_OBS_DIM).astype(np.float32)
    assert fn.obs_width == spec.OBS_DIM
    assert np.allclose(fn(o), fn(o[:spec.OBS_DIM]))
