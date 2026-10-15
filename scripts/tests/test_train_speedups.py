"""TRAINING_SPEEDUPS.md: the PyBullet env fixes (tip-over is a penalised failure, action repeat, a
shorter action history, post-pass starts, fixed evaluation layouts, failure-weighted zones, budget
ramp) and the shared callbacks (timed checkpoints, exploration watch, zone focus, reward norm)."""
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import train_speedups as speed  # noqa: E402
from hoop_net_env import (HoopNetAviary, DEFAULT_REWARD_WEIGHTS, env_kwargs_from_meta,  # noqa: E402
                          OBS_EXTRA)

LAPS = dict(reward_mode="budget", budget=10.0, start_z=0.8, armed_obs=True, streak_obs=True,
            lap_dir="same", start_mode="zones", wind=0.0, wash_deg=0.0)


def _tilt(env, roll):
    i = env.DRONE_IDS[0]
    pos, _ = p.getBasePositionAndOrientation(i, physicsClientId=env.CLIENT)
    p.resetBasePositionAndOrientation(i, pos, p.getQuaternionFromEuler([roll, 0, 0]),
                                      physicsClientId=env.CLIENT)


@pytest.mark.parametrize("mode", ["shaped", "budget"])
def test_tipping_over_ends_the_episode_as_a_penalised_failure(mode):
    env = HoopNetAviary(seed=0, **dict(LAPS, reward_mode=mode))
    env.reset(seed=1)
    _tilt(env, 1.2)
    _, r, term, trunc, info = env.step(np.zeros((1, 4), np.float32))
    env.close()
    assert term and not trunc and info["tipped"]
    if mode == "shaped":
        assert r <= -DEFAULT_REWARD_WEIGHTS["tip_penalty"] + 1.0
    else:                                    # drains the budget, like the net
        assert env.budget_score <= -env.budget + 1e-6


def test_action_repeat_holds_each_action_for_n_control_steps():
    a = HoopNetAviary(seed=0, **LAPS)
    b = HoopNetAviary(seed=0, action_repeat=2, **LAPS)
    a.reset(seed=3)
    b.reset(seed=3)
    act = np.array([[1.0, 0.0, 0.0, 0.8]], np.float32)
    ra = sum(a.step(act)[1] for _ in range(2))
    rb = b.step(act)[1]
    assert b.step_counter == a.step_counter          # one policy step = two control steps
    assert np.allclose(a.pos[0], b.pos[0], atol=1e-6)
    assert rb == pytest.approx(ra, abs=1e-6)
    a.close()
    b.close()


def test_a_shorter_action_history_keeps_the_newest_actions_and_the_layout():
    full = HoopNetAviary(seed=0, **LAPS)
    short = HoopNetAviary(seed=0, act_hist=4, **LAPS)
    assert full.observation_space.shape[-1] == 72 + OBS_EXTRA + 2
    assert short.observation_space.shape[-1] == 12 + 16 + OBS_EXTRA + 2
    full.reset(seed=4)
    short.reset(seed=4)
    for k in range(6):
        act = np.array([[k / 6, 0.2, 0.0, 0.5]], np.float32)
        of = full.step(act)[0].reshape(-1)
        os_ = short.step(act)[0].reshape(-1)
    assert np.allclose(of[:12], os_[:12])                    # the drone state
    assert np.allclose(of[72 - 16:72], os_[12:28])          # the newest 4 actions
    assert np.allclose(of[72:], os_[28:])                   # ring, flags, vector to the ring
    kw = env_kwargs_from_meta({"act_hist": 4, "obs_dim": 12 + 16 + OBS_EXTRA + 2})
    assert kw["act_hist"] == 4 and kw["armed_obs"] and kw["streak_obs"]
    full.close()
    short.close()


def test_post_pass_starts_begin_after_one_pass_on_the_exit_side():
    env = HoopNetAviary(seed=0, post_pass_p=1.0, **LAPS)
    for s in range(5):
        env.reset(seed=s)
        assert env.post_pass_start and env.passes == 1 and not env._armed and not env._returned
        side = float((env.pos[0] - env.ring_c) @ env.ring_n) * env._dir
        assert side > 0.1                                   # past the ring, on its exit side
    single = HoopNetAviary(seed=0, post_pass_p=1.0, **dict(LAPS, reward_mode="shaped"), laps=1)
    single.reset(seed=1)
    assert not single.post_pass_start                       # one-pass episodes: never
    env.close()
    single.close()


def test_fixed_evaluation_layouts_repeat():
    env = HoopNetAviary(seed=0, **LAPS)
    seen = []
    for _ in range(2):
        env.set_episode_seed(900_000_000)
        env.set_zone_index(0)
        lay = []
        for _ in range(3):
            env.reset()
            lay.append((env.start_zone, tuple(np.round(env.ring_c0, 6)), tuple(np.round(env.pos[0], 6))))
        seen.append(lay)
    env.close()
    assert seen[0] == seen[1]
    assert [z for z, _, _ in seen[0]] == [1, 2, 3]


def test_zone_weights_and_budget_hooks():
    env = HoopNetAviary(seed=0, **LAPS)
    w = np.zeros(9)
    w[4] = 1.0
    env.set_zone_weights(w)
    for s in range(4):
        env.reset(seed=s)
        assert env.start_zone == 5
    env.set_budget(3.0)
    assert env.budget == 3.0
    env.close()


def test_budget_ramp_values():
    r = speed.BudgetRamp(3.0, 10.0, 1000)
    assert r.value(0) == 3.0 and r.value(500) == 6.5 and r.value(5000) == 10.0


# ---- shared callbacks, on a tiny env ----------------------------------------------------------

class TinyEnv:
    """Minimal gymnasium env with zones, for the callbacks."""
    def __new__(cls, *a, **k):
        import gymnasium as gym
        from gymnasium import spaces

        class _E(gym.Env):
            observation_space = spaces.Box(-1, 1, (2,), np.float32)
            action_space = spaces.Box(-1, 1, (1,), np.float32)

            def __init__(self):
                self.t, self.zone, self.weights = 0, 0, None

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.t = 0
                self.zone = self.zone % 9 + 1
                return np.zeros(2, np.float32), {}

            def step(self, a):
                self.t += 1
                done = self.t >= 5
                info = {"start_zone": self.zone, "passes": 1 if self.zone == 1 else 0}
                return np.zeros(2, np.float32), float(a[0]), done, False, info

            def set_zone_weights(self, w):
                self.weights = w
        return _E()


def test_callbacks_run_inside_ppo(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([TinyEnv, TinyEnv])
    model = PPO("MlpPolicy", venv, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    with torch_no_grad():
        model.policy.log_std.fill_(math.log(0.01))       # collapsed exploration
    ck = speed.TimedCheckpoint(tmp_path, every_min=0.0001, keep=2)
    watch = speed.ExplorationWatch(0.05, tmp_path)
    zones = speed.ZoneFocus(strength=2.0, window=10, every=9)
    model.learn(600, callback=[ck, watch, zones])
    assert (tmp_path / "latest_model.zip").is_file()
    assert len(list((tmp_path / "checkpoints").glob("step_*.zip"))) == 2
    assert watch.warned and (tmp_path / "exploration_warning.txt").is_file()
    w = venv.get_attr("weights")[0]
    assert w is not None and np.isclose(sum(w), 1.0)
    assert w[0] == min(w)                                  # zone 1 passes: drawn least


def test_reward_normalisation_keeps_evaluation_raw():
    from stable_baselines3.common.vec_env import DummyVecEnv
    train, ev = speed.wrap_reward_norm(DummyVecEnv([TinyEnv]), DummyVecEnv([TinyEnv]), 0.99)
    ev.reset()
    _, r, _, _ = ev.step(np.array([[0.7]], np.float32))
    assert r[0] == pytest.approx(0.7)
    assert train.norm_reward and not ev.norm_reward


class torch_no_grad:
    def __enter__(self):
        import torch
        self._c = torch.no_grad()
        return self._c.__enter__()

    def __exit__(self, *a):
        return self._c.__exit__(*a)
