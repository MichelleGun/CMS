"""Training speed-ups on the onboard task (TRAINING_SPEEDUPS.md): parallel evaluation scores the same
episodes as one copy, start zones can be rewound and weighted, the stall cut-off truncates."""
import sys
from pathlib import Path

import numpy as np
import pytest

CMS = Path(__file__).resolve().parents[3]
for p in (CMS / "hardware", CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.evaluation import evaluate_policy  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

import train_speedups as speed  # noqa: E402
from onboard import spec  # noqa: E402
from onboard.gym_env import OnboardHoopEnv  # noqa: E402

KW = dict(profile="measured", dr=1.0, laps=0, lap_dir="same", start_mode="zones", ring_tracker=True,
          max_episode_sec=3.0)
SEED = 900_000_000


class FixedPolicy:
    """A deterministic obs -> action map with SB3's predict() signature."""
    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.atleast_2d(obs)
        a = np.zeros((len(obs), spec.ACT_DIM), np.float32)
        a[:, spec.ACT_FB] = 0.6
        a[:, spec.ACT_YAW] = np.clip(obs[:, 1] * 2.0, -1, 1)
        a[:, spec.ACT_UD] = np.clip(-obs[:, 2] * 2.0, -1, 1)
        return a, None


def _evaluate(n_envs, n_episodes, times=2):
    env = make_vec_env(OnboardHoopEnv, n_envs=n_envs, seed=1, env_kwargs=dict(KW, seed=SEED),
                       vec_env_cls=DummyVecEnv)
    out = []
    for _ in range(times):
        speed.reset_eval_layouts(env, SEED, n_episodes)
        zones = []

        def cb(locals_, _globals):
            if locals_["done"]:
                zones.append(locals_["info"].get("start_type"))
        rewards, lengths = evaluate_policy(FixedPolicy(), env, n_eval_episodes=n_episodes,
                                           deterministic=True, return_episode_rewards=True,
                                           callback=cb)
        out.append(sorted(zip(zones, np.round(rewards, 6), lengths)))
    env.close()
    return out


def test_eval_plan_matches_how_sb3_hands_out_episodes():
    assert speed.eval_plan(36, 1) == [(0, 36)]
    assert speed.eval_plan(36, 4) == [(0, 9), (9, 9), (18, 9), (27, 9)]
    assert speed.eval_plan(10, 3) == [(0, 3), (3, 3), (6, 4)]
    assert sum(c for _, c in speed.eval_plan(37, 4)) == 37


def test_parallel_evaluation_flies_the_same_episodes():
    # (a fixed, non-network policy: a network's batched maths rounds ~1e-7 differently, see
    # train_speedups' docstring, so only the layouts are guaranteed equal for a trained policy)
    one = _evaluate(1, 9)
    three = _evaluate(3, 9)
    assert one[0] == three[0]
    assert one[0] == one[1] == three[1]          # and the next evaluation is the same again
    assert sorted(z for z, _, _ in one[0]) == [f"zone_{k}" for k in range(1, 10)]


def test_default_eval_copies_follow_the_vec_choice():
    assert speed.default_eval_envs(8, "subproc") == 4
    assert speed.default_eval_envs(2, "subproc") == 2
    assert speed.default_eval_envs(8, "dummy") == 1


def test_zone_index_rewinds_the_round_robin():
    env = OnboardHoopEnv(seed=3, **KW)
    env.set_zone_index(4)
    env.reset()
    assert env.sim.start_type == "zone_5"
    env.reset()
    assert env.sim.start_type == "zone_6"


def test_zone_weights_draw_from_the_weak_zones_and_leave_other_draws_alone():
    env = OnboardHoopEnv(seed=3, **KW)
    w = np.zeros(9)
    w[[2, 7]] = 0.5                             # zones 3 and 8 only
    env.set_zone_weights(w)
    seen = set()
    for _ in range(20):
        env.reset()
        seen.add(env.sim.start_type)
    assert seen == {"zone_3", "zone_8"}
    env.set_zone_weights(None)                  # back to the round-robin
    env.reset()
    assert env.sim.start_type.startswith("zone_")
    # the weights draw from their own stream: the ring layout of a seed does not change
    a = OnboardHoopEnv(seed=5, **KW)
    b = OnboardHoopEnv(seed=5, **KW)
    b.set_zone_weights(np.full(9, 1 / 9))
    a.reset(seed=11)
    b.reset(seed=11)
    assert np.allclose(a.sim.ring.centre, b.sim.ring.centre)


def test_zone_weights_favour_the_failing_zones():
    means = {z: 1.0 for z in range(1, 10)}
    means[3] = 0.0
    means[7] = None                             # no data yet: gets tried
    w = speed.zone_weights(means, strength=2.0)
    assert np.isclose(sum(w), 1.0)
    assert w[2] == pytest.approx(3 * w[0])      # zone 3: 1 + 2 vs the best zones' 1
    assert w[6] == pytest.approx(w[2])
    assert min(w) > 0                           # nothing is dropped
    assert np.allclose(speed.zone_weights({z: 0.0 for z in range(1, 10)}, 2.0), 1 / 9)


def test_stall_cut_truncates_after_seconds_without_a_pass():
    env = OnboardHoopEnv(seed=3, **dict(KW, max_episode_sec=90.0), stall_sec=2.0)
    env.reset()
    for t in range(1, 200):
        _, _, term, trunc, info = env.step(np.zeros(spec.ACT_DIM, np.float32))   # hover: no pass
        if term or trunc:
            break
    assert trunc and not term and info.get("stalled")
    assert t == int(2.0 * spec.CTRL_HZ)
    plain = OnboardHoopEnv(seed=3, **dict(KW, max_episode_sec=90.0))
    plain.reset()
    for _ in range(40):
        _, _, term, trunc, _ = plain.step(np.zeros(spec.ACT_DIM, np.float32))
        assert not (term or trunc)
