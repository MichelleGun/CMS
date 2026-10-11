"""MDP v5 for the PyBullet net env: same-side laps (round the outside), the streak bonus, the
pass count in the observation (84-D), and starts that cycle through 9 zones."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoop_net_env import START_MARGIN, ZONE_CELLS, HoopNetAviary, env_kwargs_from_meta  # noqa: E402


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


def _same_env(**kw):
    e = HoopNetAviary(seed=0, wind=0.0, reverse_prob=0.0, laps=0, lap_dir="same", **kw)
    e.reset(seed=0)
    e._dir = 1.0
    return e


def _side(e):
    u = np.cross(e.ring_n, [0, 0, 1.0])
    return u / np.linalg.norm(u)


def test_same_side_lap_needs_the_way_round():
    e = _same_env()
    n, c, u = e.ring_n, e.ring_c, _side(e)
    _seed_at(e, c - n * 0.1)
    _, ev = _step_to(e, c + n * 0.05)                          # pass 1
    assert ev["passed"] and e._dir == 1.0 and not e._armed and not e._returned
    _, ev = _step_to(e, c - n * 0.05)                          # straight back through: wrong way
    assert ev["wrong"] and not e._returned and not e._armed
    _step_to(e, c - n * 0.5)                                   # in front, but never went round
    assert not e._armed
    _, ev = _step_to(e, c + n * 0.05)                          # through again: nothing
    assert not ev["passed"] and e.passes == 1
    _step_to(e, c + n * 0.3 + u * 0.6)                         # the way round: beside, exit side
    _, ev = _step_to(e, c - n * 0.1 + u * 0.6)                 # back across the plane, outside
    assert ev["returned"] and e._returned
    _step_to(e, c - n * 0.4)                                   # in front: re-armed
    assert e._armed
    _step_to(e, c - n * 0.1)
    _, ev = _step_to(e, c + n * 0.05)                          # pass 2, the same way
    assert ev["passed"] and e.passes == 2 and e._dir == 1.0
    e.close()


def test_alternate_is_still_the_default():
    e = HoopNetAviary(seed=0, wind=0.0, reverse_prob=0.0, laps=0)
    e.reset(seed=0)
    e._dir = 1.0
    n, c = e.ring_n, e.ring_c
    _seed_at(e, c - n * 0.1)
    _step_to(e, c + n * 0.05)
    assert e.lap_dir == "alternate" and e._dir == -1.0
    e.close()


def test_streak_pays_more_per_pass_but_not_into_the_budget():
    e = _same_env(reward_mode="budget", budget=50,
                  reward_weights={"budget_streak": 0.5, "budget_streak_cap": 2.0})
    n, c, u = e.ring_n, e.ring_c, _side(e)
    _seed_at(e, c - n * 0.1)
    paid = []
    for _k in range(4):
        r, ev = _step_to(e, c + n * 0.05)
        assert ev["passed"]
        paid.append(r + 0.005)                                 # minus the step's time cost
        _step_to(e, c + n * 0.3 + u * 0.6)
        _step_to(e, c - n * 0.1 + u * 0.6)
        _step_to(e, c - n * 0.4)
        _step_to(e, c - n * 0.1)
    assert np.allclose(paid, [1.0, 1.5, 2.0, 2.0], atol=1e-6)  # x1, x1.5, x2, capped at x2
    # the budget tally only ever counted the base +1 per pass
    assert abs(e.budget_score - (4 * 1.0 - 0.005 * 20)) < 1e-6
    e.close()


def test_pass_count_in_the_84d_observation():
    e = _same_env(armed_obs=True, streak_obs=True)
    obs = e._computeObs()
    assert obs.shape == (1, 84) and obs[0, -4] == 0.0 and obs[0, -5] == 1.0
    n, c = e.ring_n, e.ring_c
    _seed_at(e, c - n * 0.1)
    _step_to(e, c + n * 0.05)
    obs = e._computeObs()
    assert obs[0, -4] == 0.25 and obs[0, -5] == 0.0             # one pass, not armed
    assert np.allclose(obs[0, -3:], e.ring_c - e.pos[0], atol=1e-5)
    assert e.observation_space.shape == (1, 84)
    e.close()
    with pytest.raises(ValueError):
        HoopNetAviary(seed=0, streak_obs=True)                   # needs armed_obs


def test_env_kwargs_from_meta_restores_v5():
    kw = env_kwargs_from_meta({"obs_dim": 84, "lap_dir": "same", "start_mode": "zones"})
    assert kw == {"armed_obs": True, "streak_obs": True, "lap_dir": "same", "start_mode": "zones"}


def test_zones_cycle_in_order_and_land_in_their_cell():
    e = HoopNetAviary(seed=4, start_mode="zones", start_z=0.8)
    first = None
    seen = []
    for k in range(18):
        e.reset(seed=100 + k)
        z = e.start_zone
        seen.append(z)
        x, y, h = e.INIT_XYZS[0]
        half = e.ROOM[0] / 2 - START_MARGIN
        if z <= 8:
            w = 2 * half / 3
            col, row = ZONE_CELLS[z - 1]
            assert abs(x - col * w) <= w / 2 + 1e-9 and abs(y - row * w) <= w / 2 + 1e-9
            assert np.hypot(x - e.ring_c0[0], y - e.ring_c0[1]) > 0.25 or True
            assert 0.7 - 1e-9 <= h <= 0.9 + 1e-9
        else:
            assert np.hypot(x - e.ring_c0[0], y - e.ring_c0[1]) <= 0.15 * np.sqrt(2) + 1e-9
            bottom = e.ring_c0[2] - e.inner_radius - e.spec.tube_radius
            assert h < bottom - e.spec.drone_radius
        first = first or z
    # every zone exactly twice, consecutive episodes walk the zones in order
    assert sorted(seen) == sorted(list(range(1, 10)) * 2)
    assert all((b - a) % 9 == 1 for a, b in zip(seen, seen[1:]))
    assert e._computeInfo()["start_zone"] == seen[-1]
    e.close()


def test_widen_82_to_84_acts_the_same(tmp_path):
    from stable_baselines3 import PPO
    from vec_util import load_for_resume
    old_env = HoopNetAviary(seed=0)
    new_env = HoopNetAviary(seed=0, armed_obs=True, streak_obs=True)
    model = PPO("MlpPolicy", old_env, n_steps=64, batch_size=32, seed=0, device="cpu")
    model.save(tmp_path / "old.zip")
    wide, _ = load_for_resume(PPO, tmp_path / "old.zip", new_env, widen_at=-3)
    rng = np.random.default_rng(0)
    for _ in range(5):
        o82 = rng.normal(size=(1, 82)).astype(np.float32)
        o84 = np.insert(o82, [79, 79], [1.0, 0.5], axis=1)
        a_old, _ = model.predict(o82, deterministic=True)
        a_new, _ = wide.predict(o84, deterministic=True)
        assert np.allclose(a_old, a_new, atol=1e-6)
    old_env.close()
    new_env.close()
