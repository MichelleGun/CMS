"""The PyBullet -> onboard teacher bridge (make_teacher_demos.py), demo source weights, and the
BC-regularised PPO trainer (train_onboard.py).

    python -m pytest hardware/onboard/tests/test_teacher.py -q
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HW = Path(__file__).resolve().parents[2]
CMS = HW.parent
for p in (HW, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# flake8: noqa: E402
from imitation.record_demo import DemoRecorder, load_demos, parse_source_weights, sample_weights
from onboard import spec
from onboard.make_teacher_demos import (PYB_SPEED_LIMIT, TEACHER_OBS_DIM, SimTeacher,
                                        _rotation_between)
from onboard.sim import MAX_SPEED, OnboardSim


class _Space:
    def __init__(self, shape):
        self.shape = shape


class FakeModel:
    """Records the observation it is given and returns a fixed action."""
    def __init__(self, action=(1.0, 0.0, 0.0, 1.0)):
        self.observation_space = _Space((1, TEACHER_OBS_DIM))
        self.action = np.asarray(action, dtype=np.float32).reshape(1, 4)
        self.seen = None

    def predict(self, obs, deterministic=True):
        self.seen = obs
        return self.action, None


def _sim(seed=3):
    return OnboardSim(profile="ideal", seed=seed)


# ---------------------------------------------------------------- the observation bridge

def test_world_frame_obs_matches_hoopnetaviary_layout():
    """Field order must be HoopNetAviary._computeObs's: kin 12 (pos, rpy, vel, ang vel), a
    15-step action buffer, then the pass-direction normal and ring - pos."""
    t = SimTeacher(FakeModel(), frame="world")
    sim = _sim()
    sim.vel = np.array([0.1, -0.05, 0.02])
    o = t.teacher_obs(sim).reshape(-1)
    assert o.shape == (78,)
    assert np.allclose(o[0:3], sim.pos)
    assert np.allclose(o[3:6], 0) and np.allclose(o[9:12], 0)
    assert np.allclose(o[6:9], sim.vel)
    assert np.allclose(o[12:72], 0)                         # empty action buffer at the start
    n = o[72:75]
    assert np.isclose(np.linalg.norm(n), 1)
    assert np.dot(sim.ring.centre - sim.pos, n) > 0          # it points the way to fly through
    assert np.allclose(o[75:78], sim.ring.centre - sim.pos, atol=1e-6)


def test_obs_layout_against_the_real_env():
    """The real HoopNetAviary puts the same quantities in the same slots (PyBullet needed)."""
    pytest.importorskip("pybullet")
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType
    from hoop_net_env import HoopNetAviary, ROOM
    env = HoopNetAviary(obs=ObservationType.KIN, act=ActionType.VEL, room=ROOM, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        o = obs.reshape(-1)
        assert o.shape == (TEACHER_OBS_DIM,)
        st = env._getDroneStateVector(0)
        assert np.allclose(o[0:3], st[0:3], atol=1e-5)
        assert np.allclose(o[6:9], st[10:13], atol=1e-5)
        assert np.allclose(o[72:75], env.ring_n * env._dir, atol=1e-5)
        assert np.allclose(o[75:78], env.ring_c - st[0:3], atol=1e-5)
    finally:
        env.close()


def test_action_buffer_fills_at_pybullet_rate():
    m = FakeModel((0.2, 0.3, 0.4, 0.5))
    t = SimTeacher(m, frame="world")
    sim = _sim()
    t.act(sim)
    buf = t.teacher_obs(sim).reshape(-1)[12:72].reshape(15, 4)
    # 30 Hz teacher, 15 Hz onboard: one tick = two buffer slots, newest last
    assert np.allclose(buf[-2:], [0.2, 0.3, 0.4, 0.5])
    assert np.allclose(buf[:-2], 0)


# ---------------------------------------------------------------- the frame alignment

def test_rotation_between():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.normal(size=3), rng.normal(size=3)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        R = _rotation_between(a, b)
        assert np.allclose(R @ a, b, atol=1e-9)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    R = _rotation_between([1, 0, 0], [-1, 0, 0])
    assert np.allclose(R @ np.array([1, 0, 0]), [-1, 0, 0])


def test_canonical_frame_puts_the_ring_at_the_nominal_pose():
    pytest.importorskip("pybullet")
    from hoop_net_env import DEFAULT_SPEC, ring_normal
    t = SimTeacher(FakeModel(), frame="canonical")
    sim = _sim()
    o = t.teacher_obs(sim).reshape(-1)
    nominal = ring_normal(DEFAULT_SPEC.normal_yaw_deg, DEFAULT_SPEC.normal_tilt_deg)
    assert np.allclose(o[72:75], nominal, atol=1e-6)
    # the ring is where the teacher was trained to find it, and distances are preserved
    assert np.allclose(o[0:3] + o[75:78], DEFAULT_SPEC.centre, atol=1e-6)
    assert np.isclose(np.linalg.norm(o[75:78]), np.linalg.norm(sim.ring.centre - sim.pos), atol=1e-6)


def test_canonical_command_along_the_axis_flies_along_the_real_axis():
    """'Fly through along the normal' in the teacher's frame must come out as flying along the
    onboard ring's own axis, whatever the frame rotation."""
    pytest.importorskip("pybullet")
    from hoop_net_env import DEFAULT_SPEC, ring_normal
    nominal = ring_normal(DEFAULT_SPEC.normal_yaw_deg, DEFAULT_SPEC.normal_tilt_deg)
    t = SimTeacher(FakeModel(), frame="canonical")
    sim = _sim()
    rc = t.action_to_rc(np.r_[nominal, 1.0], sim)
    v = (rc[spec.ACT_FB] * sim.forward + rc[spec.ACT_LR] * sim.right
         + np.array([0, 0, rc[spec.ACT_UD]])) * MAX_SPEED
    n = SimTeacher.pass_dir(sim)
    assert np.dot(v, n) / np.linalg.norm(v) > 0.999


# ---------------------------------------------------------------- velocity -> rc

@pytest.mark.parametrize("yaw", [0.0, 37.0, -120.0])
def test_velocity_to_rc_reproduces_the_commanded_velocity(yaw):
    t = SimTeacher(FakeModel(), frame="world")
    sim = _sim()
    sim.yaw_deg = yaw
    d = np.array([0.3, -0.5, 0.2])
    rc = t.action_to_rc(np.r_[d, 0.6], sim)
    want = PYB_SPEED_LIMIT * 0.6 * d / np.linalg.norm(d)
    got = (rc[spec.ACT_FB] * sim.forward + rc[spec.ACT_LR] * sim.right
           + np.array([0, 0, rc[spec.ACT_UD]])) * MAX_SPEED
    assert np.allclose(got, want, atol=1e-9)
    assert rc[spec.ACT_YAW] == 0.0


def test_yaw_turns_the_nose_towards_the_ring():
    sim = _sim()
    d = sim.ring.centre - sim.pos
    bearing = math.degrees(math.atan2(d[1], d[0]))
    sim.yaw_deg = bearing + 30.0                 # nose 30 deg left of the ring
    a = SimTeacher.yaw_to_ring(sim)
    before = sim.yaw_deg
    sim.step([0, 0, 0, a])
    sim.step([0, 0, 0, a])
    assert abs(((sim.yaw_deg - bearing) + 180) % 360 - 180) < abs(before - bearing)


# ---------------------------------------------------------------- demo source weights

def test_source_weights():
    assert parse_source_weights(["human=3", "sim_teacher=0.5"]) == {"human": 3.0, "sim_teacher": 0.5}
    with pytest.raises(ValueError):
        parse_source_weights(["human"])
    w = sample_weights(np.array(["human", "sim_teacher", "sim_teacher", "x"]), {"human": 3.0})
    assert np.isclose(w.mean(), 1.0)
    assert np.isclose(w[0] / w[1], 3.0) and np.isclose(w[1], w[3])


def test_load_demos_tags_every_sample_with_its_source(tmp_path):
    paths = []
    for src, n in (("human", 3), ("sim_teacher", 5)):
        rec = DemoRecorder(meta={"source": src}, obs_dim=spec.OBS_DIM, task=spec.TASK_NAME,
                           obs_shape=(spec.OBS_DIM,))
        for i in range(n):
            rec.add(np.zeros(spec.OBS_DIM), np.zeros(4), np.zeros(4), i * 0.1)
        paths.append(rec.save(tmp_path / f"{src}.npz"))
    d = load_demos(paths)
    assert d["source"].tolist() == ["human"] * 3 + ["sim_teacher"] * 5


# ---------------------------------------------------------------- the guided trainer

@pytest.mark.slow
def test_bc_regularised_ppo_trains_and_saves_a_plain_ppo(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from onboard.gym_env import OnboardHoopEnv
    from onboard.policy import load_policy
    from onboard.train_onboard import BCRegularizedPPO

    venv = DummyVecEnv([lambda: OnboardHoopEnv(seed=0, max_episode_sec=3.0)])
    model = BCRegularizedPPO("MlpPolicy", venv, n_steps=64, batch_size=32, n_epochs=2,
                             device="cpu", seed=0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(200, spec.OBS_DIM)).astype(np.float32)
    act = np.tile([0.0, 0.5, 0.0, 0.0], (200, 1)).astype(np.float32)
    model.configure(obs, act, np.ones(200, np.float32), bc_coef=5.0, bc_coef_min=5.0,
                    bc_decay_steps=10_000, critic_warmup_steps=0)
    import torch
    with torch.no_grad():
        before = float(((model.policy.get_distribution(torch.as_tensor(obs)).distribution.mean
                         - torch.as_tensor(act)) ** 2).mean())
    model.learn(64 * 6)
    with torch.no_grad():
        after = float(((model.policy.get_distribution(torch.as_tensor(obs)).distribution.mean
                        - torch.as_tensor(act)) ** 2).mean())
    assert after < before                     # the demo term pulls the policy towards the demos

    model.save(str(tmp_path / "best_model.zip"))
    (tmp_path / "meta.json").write_text(json.dumps({"algo": "ppo", "task": spec.TASK_NAME}))
    plain = PPO.load(str(tmp_path / "best_model.zip"), device="cpu")
    assert getattr(plain, "_bc_obs", None) is None   # the demos are not stored in the zip
    fn = load_policy(tmp_path / "best_model.zip")
    assert fn(np.zeros(spec.OBS_DIM)).shape == (4,)


@pytest.mark.slow
def test_critic_warmup_freezes_the_actor():
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from onboard.gym_env import OnboardHoopEnv
    from onboard.train_onboard import BCRegularizedPPO

    venv = DummyVecEnv([lambda: OnboardHoopEnv(seed=0, max_episode_sec=3.0)])
    model = BCRegularizedPPO("MlpPolicy", venv, n_steps=64, batch_size=32, n_epochs=2,
                             device="cpu", seed=0)
    model.configure(critic_warmup_steps=10_000)
    actor0 = [p.detach().clone() for p in model._actor_params()]
    critic0 = [p.detach().clone() for p in model.policy.value_net.parameters()]
    model.learn(128)
    assert all(torch.equal(a, b) for a, b in zip(actor0, model._actor_params()))
    assert not all(torch.equal(a, b) for a, b in zip(critic0, model.policy.value_net.parameters()))
