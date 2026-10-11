"""Onboard v4 (2026-10-11 night): failures drain the rest of the episode, the facing ("find")
potential, the `pushed` odometer and its demo upgrade, the prop-wash ring, the lap expert, and
widening a 33-input policy."""
import numpy as np
import pytest

from onboard import spec
from onboard.gym_env import (OnboardHoopEnv, R_OUT_OF_CAGE, R_STRUCK, W_TIME)
from onboard.obs import ObsBuilder, upgrade_obs
from onboard.ring_detect import RingObs
from onboard.sim import OnboardSim, Ring


def _env(**kw):
    env = OnboardHoopEnv(ring=Ring(np.array([0.0, 0.0, 1.0]), 0.0), profile="ideal", laps=0, **kw)
    env.reset(seed=0)
    env.sim.pos = np.array([-0.5, 0.0, 1.0])
    env.sim.yaw_deg = 0.0
    env.sim.vel[:] = 0.0
    env._dir, env._armed = 1, True
    env._reset_shaping()
    return env


def test_out_of_cage_is_never_cheaper_than_flying_on():
    """Hovering to the time limit costs W_TIME per tick; flying into the net early must cost
    at least that much more (it used to be -10 against up to -18 for waiting)."""
    env = _env()
    env.sim.pos = np.array([0.0, 0.97, 1.0])
    env.sim.yaw_deg = 90.0
    env._reset_shaping()
    left = env.max_ticks
    total = 0.0
    for _ in range(50):
        _o, r, term, _tr, info = env.step(np.array([0.0, 1.0, 0.0, 0.0]))
        total += r
        if term:
            break
    assert info["phase"] == "out_of_cage"
    assert total < R_OUT_OF_CAGE - W_TIME * (left - 50) + 1.0
    env.close()


def test_strike_costs_the_fixed_penalty_plus_the_drain():
    env = _env()
    env.sim.pos = np.array([-0.3, 0.0, 1.0 + 0.18])      # aimed at the noodle
    env._reset_shaping()
    for _ in range(100):
        _o, r, term, _tr, info = env.step(np.array([0.0, 1.0, 0.0, 0.0]))
        if term:
            break
    assert info["phase"] == "struck"
    assert r < R_STRUCK - W_TIME * (env.max_ticks - env._ticks) + 1.0
    env.close()


def test_facing_potential_pays_turning_towards_the_ring():
    env = _env()
    env.sim.yaw_deg = 180.0                                # facing away
    env._reset_shaping()
    gained = 0.0
    for _ in range(140):                                  # turn round on the spot
        _o, r, _te, _tr, _i = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
        gained += r
        if abs(((env.sim.yaw_deg + 180) % 360) - 180) < 10:
            break
    assert gained > 1.0
    env.close()


def test_spinning_on_the_spot_earns_nothing():
    """Potential-based: a full turn back to the start heading pays only the time penalty."""
    env = _env()
    total, n = 0.0, 0
    y0 = env.sim.yaw_deg
    turned = 0.0
    while turned < 360.0 and n < 1000:
        prev = env.sim.yaw_deg
        _o, r, _te, _tr, _i = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
        turned += abs(((env.sim.yaw_deg - prev + 180) % 360) - 180)
        total += r
        n += 1
    assert total < -W_TIME * n + 1.0                       # nothing gained beyond a sliver
    assert abs(((env.sim.yaw_deg - y0 + 180) % 360) - 180) < 30
    env.close()


def test_pushed_counts_forward_flight_since_the_last_sighting():
    b = ObsBuilder()
    seen = RingObs(found=True, fresh=True, cx=0.0, cy=0.0, scale=0.3, axis_ratio=1.0,
                   range_m=0.4, age=0)
    lost = RingObs(found=False, fresh=False, age=spec.LOCK_MAX_AGE, range_m=float("inf"))
    sl = spec.OBS_SLICES["pushed"].start
    o = b.build(seen, {"vgx": 20}, 0.0)
    assert o[sl] == 0.0
    t = 0.0
    for _ in range(15):                                    # 1 s at 20 cm/s, blind
        t += spec.CTRL_DT
        o = b.build(lost, {"vgx": 20}, t)
    assert abs(o[sl] * spec.RANGE_NORM - 0.20) < 0.02
    o = b.build(seen, {"vgx": 20}, t + spec.CTRL_DT)       # seen again: back to zero
    assert o[sl] == 0.0


def test_upgraded_demos_match_the_builder():
    """A 33-wide recording upgraded on load gets the same `pushed` the builder would have made."""
    env = OnboardHoopEnv(seed=3, dr=1.0)
    rows, eps, ts = [], [], []
    for ep in range(2):
        obs, _ = env.reset(seed=50 + ep)
        for k in range(120):
            rows.append(obs.copy())
            eps.append(ep)
            ts.append(env.sim.t)
            obs, _r, term, trunc, _i = env.step(np.array([0.0, 0.8, 0.1, 0.2]))
            if term or trunc:
                break
    full = np.array(rows)
    for width in (33, 34):                               # before `pushed`, before the heading memory
        legacy = full[:, spec.legacy_columns(width)]
        assert legacy.shape[1] == width
        up = upgrade_obs(legacy, np.array(eps), np.array(ts))
        assert up.shape == full.shape
        assert np.allclose(up, full, atol=2e-3)
    env.close()


def test_load_demo_upgrades_old_onboard_files(tmp_path):
    import json
    from imitation.record_demo import load_demo
    n = 20
    p = tmp_path / "old.npz"
    np.savez_compressed(p, obs=np.zeros((n, spec.LEGACY_OBS_DIM), np.float32),
                        intent=np.zeros((n, 4), np.float32), executed=np.zeros((n, 4), np.float32),
                        t=np.arange(n) / 15.0, episode=np.zeros(n, np.int64),
                        meta=json.dumps({"task": "onboard_hoop", "source": "human"}))
    d = load_demo(p)
    assert d["obs"].shape == (n, spec.OBS_DIM)
    assert d["meta"]["upgraded_from_obs_dim"] == spec.LEGACY_OBS_DIM


def test_wash_pushes_the_ring_away_and_tilts_it_up():
    sim = OnboardSim(seed=0, wash_deg=15.0)
    c0, n0 = sim.ring0.centre.copy(), sim.ring0.normal
    sim.pos = c0 - n0 * 0.2                                # just in front, facing it
    sim.yaw_deg = sim.ring0.normal_deg
    for _ in range(45):
        sim._update_ring()
        sim.t += spec.CTRL_DT
    assert sim.wash_tilt_deg > 3.0
    assert float(np.dot(sim.ring.centre - c0, n0)) > 0.01  # moved away from the drone
    assert sim.ring.normal[2] > 0.05                       # face tilted up
    sim.pos = np.array([-0.9, -0.9, 0.5])                  # gone: settles back
    for _ in range(200):
        sim._update_ring()
    assert abs(sim.wash_tilt_deg) < 0.5


def test_no_wash_keeps_the_random_stream():
    a, b = OnboardSim(seed=7, dr=1.0, hard_start_p=0.5), OnboardSim(seed=7, dr=1.0, hard_start_p=0.5,
                                                                     wash_deg=0.0)
    for s in range(5):
        a.reset(seed=s)
        b.reset(seed=s)
        assert np.allclose(a.pos, b.pos) and np.allclose(a.ring.centre, b.ring.centre)


def test_lap_expert_flies_more_than_one_pass():
    from onboard.make_demos import generate
    rec, tally = generate(episodes=10, seed=0, dr=0.0, laps=0, verbose=False)
    assert tally["max_passes"] >= 2 and tally["mean_passes"] >= 1.2
    assert len(rec) > 0


def test_widened_onboard_policy_acts_like_the_old_one(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from vec_util import load_for_resume
    from onboard.policy import load_policy
    from onboard.train_onboard import _widen_index

    for width in (33, 34):
        class Old(OnboardHoopEnv):
            def __init__(self):
                super().__init__(seed=0)
                self.observation_space = spaces.Box(-np.inf, np.inf, shape=(width,),
                                                    dtype=np.float32)
        old = PPO("MlpPolicy", Old(), n_steps=64, batch_size=32, seed=0, device="cpu")
        old.save(tmp_path / f"old{width}.zip")
        new_env = OnboardHoopEnv(seed=0)
        wide, _ = load_for_resume(PPO, tmp_path / f"old{width}.zip", new_env, widen_at=_widen_index)
        legacy_fn = load_policy(tmp_path / f"old{width}.zip", algo="ppo")   # old models still fly
        keep = spec.legacy_columns(width)
        rng = np.random.default_rng(1)
        for _ in range(5):
            o = rng.normal(size=spec.OBS_DIM).astype(np.float32)
            a_wide, _ = wide.predict(o, deterministic=True)
            a_old, _ = old.predict(o[keep], deterministic=True)
            assert np.allclose(a_wide, a_old, atol=1e-6)
            assert np.allclose(legacy_fn(o), np.clip(a_old, -1, 1), atol=1e-6)
        new_env.close()
