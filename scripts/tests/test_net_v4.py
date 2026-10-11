"""MDP v4 for the PyBullet net env (2026-10-11 night): hover starts, the prop-wash ring push,
the lap "armed" flag in the observation, budget + shaping, and widening an older policy."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoop_net_env import ARMED_INDEX, HoopNetAviary, env_kwargs_from_meta  # noqa: E402


def _step_to(env, pos):
    env._prev_pos = env.pos[0].copy()
    env.pos[0] = pos
    env.vel[0] = 0
    env._post_physics()
    return env._reward, dict(env._ev)


def _seed_at(env, pos):
    env.pos[0] = pos
    env._fresh = True
    env._post_physics()


@pytest.mark.parametrize("seed", range(5))
def test_hover_start_height(seed):
    e = HoopNetAviary(seed=seed, start_z=0.8)
    e.reset(seed=seed)
    z = e._getDroneStateVector(0)[2]
    assert 0.7 - 1e-6 <= z <= 0.9 + 1e-6
    for _ in range(30):                                   # 1 s of "hold still": no crash, no big drop
        e.step(np.array([[0, 0, 0, 0.0]]))
    assert e._getDroneStateVector(0)[2] > 0.5
    e.close()


def test_floor_start_is_still_the_default():
    e = HoopNetAviary(seed=0)
    e.reset(seed=0)
    assert e._getDroneStateVector(0)[2] < 0.1
    e.close()


def test_armed_flag_in_obs_follows_the_lap_rules():
    e = HoopNetAviary(seed=0, laps=0, wind=0.0, reverse_prob=0.0, armed_obs=True)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (1, 83) and obs[0, ARMED_INDEX] == 1.0
    assert np.allclose(obs[0, -3:], e.ring_c - e._getDroneStateVector(0)[:3], atol=1e-5)
    e._dir = 1.0
    n, c = e.ring_n, e.ring_c
    _seed_at(e, c - n * 0.1)
    _, ev = _step_to(e, c + n * 0.05)                     # pass: no longer armed
    assert ev["passed"] and e._computeObs()[0, ARMED_INDEX] == 0.0
    _step_to(e, c + n * 0.2)
    _step_to(e, c + n * 0.4)                              # flown clear on the exit side
    assert e._computeObs()[0, ARMED_INDEX] == 1.0
    assert e.observation_space.shape == (1, 83)
    e.close()


def test_env_kwargs_from_meta_restores_the_armed_layout():
    assert env_kwargs_from_meta({"obs_dim": 83})["armed_obs"] is True
    assert "armed_obs" not in env_kwargs_from_meta({"obs_dim": 82})
    kw = env_kwargs_from_meta({"wash_deg": 15.0, "start_z": 0.8})
    assert kw == {"wash_deg": 15.0, "start_z": 0.8}


def test_wash_pushes_the_ring_bottom_away_and_it_settles_back():
    e = HoopNetAviary(seed=0, wind=0.0, wash_deg=15.0, reverse_prob=0.0)
    e.reset(seed=0)
    n0, c0 = e.ring_n0.copy(), e.ring_c0.copy()
    dt = 1.0 / e.CTRL_FREQ
    e._drone_pos = c0 - n0 * 0.15                         # just in front of the ring
    for _ in range(90):
        e._update_ring(dt)
    assert e.wash_ang > 3.0                                # tilted
    assert float((e.ring_c - c0) @ n0) > 0.01              # the centre moved away from the drone
    e._drone_pos = c0 + n0 * 0.15                          # now on the other side
    for _ in range(120):
        e._update_ring(dt)
    assert e.wash_ang < -3.0 and float((e.ring_c - c0) @ n0) < -0.01
    e._drone_pos = np.array([0.9, 0.9, 0.3])               # far away: swings back to rest
    for _ in range(300):
        e._update_ring(dt)
    assert abs(e.wash_ang) < 0.5 and np.linalg.norm(e.ring_c - c0) < 0.01
    e.close()


def test_no_wash_draws_nothing_extra():
    """wash_deg=0 keeps old runs' random streams: same ring as before the change."""
    a = HoopNetAviary(seed=3)
    b = HoopNetAviary(seed=3, wash_deg=0.0)
    a.reset(seed=11)
    b.reset(seed=11)
    assert np.allclose(a.ring_c0, b.ring_c0) and np.allclose(a.ring_n0, b.ring_n0)
    a.close()
    b.close()


def test_budget_shaping_pays_progress_but_not_the_budget():
    e = HoopNetAviary(seed=0, wind=0.0, reward_mode="budget", reverse_prob=0.0,
                      reward_weights={"budget_shape": 2.0})
    e.reset(seed=0)
    e._dir = 1.0
    n, c = e.ring_n, e.ring_c
    _seed_at(e, c - n * 0.9)
    r, _ = _step_to(e, c - n * 0.6)                       # 0.3 m closer to the approach point
    assert r > 0.5
    assert abs(e.budget_score + 0.005) < 1e-9             # the budget only saw the time cost
    r, _ = _step_to(e, c - n * 0.9)                       # and back: shaping gives it all back
    assert r < -0.5
    assert abs(e.budget_score + 0.010) < 1e-9
    e.close()


def test_budget_shaping_leads_out_after_a_pass():
    """After a pass (not re-armed) the potential rises as the drone flies out on the exit side."""
    e = HoopNetAviary(seed=0, wind=0.0, reward_mode="budget", reverse_prob=0.0,
                      reward_weights={"budget_shape": 2.0})
    e.reset(seed=0)
    e._dir = 1.0
    n, c = e.ring_n, e.ring_c
    _seed_at(e, c - n * 0.1)
    _step_to(e, c + n * 0.05)                              # pass
    r_out, _ = _step_to(e, c + n * 0.20)
    assert r_out > 0
    e.close()


def test_widened_policy_acts_exactly_like_the_old_one(tmp_path):
    from stable_baselines3 import PPO
    from vec_util import load_for_resume
    old_env = HoopNetAviary(seed=0)
    new_env = HoopNetAviary(seed=0, armed_obs=True)
    model = PPO("MlpPolicy", old_env, n_steps=64, batch_size=32, seed=0, device="cpu")
    model.save(tmp_path / "old.zip")
    widened, _ = load_for_resume(PPO, tmp_path / "old.zip", new_env, widen_at=ARMED_INDEX + 1)
    assert widened.observation_space.shape == (1, 83)
    rng = np.random.default_rng(0)
    for _ in range(5):
        o82 = rng.normal(size=(1, 82)).astype(np.float32)
        for flag in (0.0, 1.0):
            o83 = np.insert(o82, 79, flag, axis=1)
            a_old, _ = model.predict(o82, deterministic=True)
            a_new, _ = widened.predict(o83, deterministic=True)
            assert np.allclose(a_old, a_new, atol=1e-6)
    old_env.close()
    new_env.close()


def test_teacher_bridge_accepts_83d(tmp_path):
    hw = Path(__file__).resolve().parents[2] / "hardware"
    sys.path.insert(0, str(hw))
    from stable_baselines3 import PPO
    from onboard.make_teacher_demos import SimTeacher
    from onboard.sim import OnboardSim
    env = HoopNetAviary(seed=0, armed_obs=True)
    t = SimTeacher(PPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cpu"))
    o = t.teacher_obs(OnboardSim(seed=0))
    assert o.shape == (1, 83) and o[0, ARMED_INDEX] == 1.0
    env.close()
