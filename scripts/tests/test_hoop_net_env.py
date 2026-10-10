import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoop_net_env import START_MARGIN, HoopNetAviary, HoopSpec, START_Z, ring_normal  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = HoopNetAviary(seed=0)
    yield e
    e.close()


def _fly(env, policy, steps):
    info = {}
    for _ in range(steps):
        obs, r, te, tr, info = env.step(policy(env))
        if te or tr:
            break
    return info, te, tr


def _expert(env):
    pos = env._getDroneStateVector(0)[:3]
    n, c = env.ring_n * env._dir, env.ring_c
    d = pos - c
    s = d @ n
    lat = np.linalg.norm(d - s * n)
    tgt = c + n * 0.6 if (-0.4 < s < 0.2 and lat < 0.06) else c - n * 0.35
    v = tgt - pos
    return np.array([[*(v / (np.linalg.norm(v) + 1e-9)), 1.0]])


def test_obs_shape_and_layout(env):
    obs, _ = env.reset(seed=1)
    assert obs.shape == (1, 82)
    assert np.allclose(obs[0, -3:], env.ring_c - env._getDroneStateVector(0)[:3], atol=1e-5)
    assert np.isclose(np.linalg.norm(obs[0, -10:-7]), 1.0, atol=1e-5)
    assert np.allclose(obs[0, -7:-4], env.ring_vel, atol=1e-5)
    assert np.isclose(obs[0, -4], env.inner_radius, atol=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_always_starts_on_floor_and_ring_jitter_in_range(env, seed):
    env.reset(seed=seed)
    z = env._getDroneStateVector(0)[2]
    assert z < 0.1
    sp = env.spec
    assert np.all(np.abs(env.ring_c0 - np.array(sp.centre)) <= np.array(sp.centre_jitter) + 1e-9)
    assert 1 - sp.radius_jitter_frac - 1e-9 <= env.ring_r_factor <= 1 + sp.radius_jitter_frac + 1e-9
    # episode facing within the yaw/tilt jitter of the nominal (scan-measured) facing, or of its
    # mirrored lean when the wind holds the ring over the other way this episode
    lean = -sp.normal_tilt_deg if env._lean_flipped else sp.normal_tilt_deg
    nominal = ring_normal(sp.normal_yaw_deg, lean)
    max_dev = np.radians(sp.yaw_jitter_deg + sp.tilt_jitter_deg)
    assert np.degrees(np.arccos(np.clip(env.ring_n0 @ nominal, -1, 1))) <= np.degrees(max_dev) + 1e-6
    # live pose stays within the wind sway of the episode pose
    assert np.linalg.norm(env.ring_c - env.ring_c0) <= _max_centre_sway(env) + 1e-9


def test_start_override_is_clamped_to_floor():
    e = HoopNetAviary(seed=0, start_pos=[0.3, -0.2, 1.5])
    e.reset(seed=0)
    assert e._getDroneStateVector(0)[2] < 0.1
    e.close()


def test_net_bounce_keeps_drone_inside():
    env = HoopNetAviary(seed=0, net_terminate=False)
    env.reset(seed=2)
    for direction in ([0, 0, 1, 1], [1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1]):
        env.reset(seed=2)
        hit = False
        for _ in range(500):
            _, _, te, tr, info = env.step(np.array([direction], dtype=np.float32))
            hit |= info["net_contact"]
            pos = env._getDroneStateVector(0)[:3]
            assert abs(pos[0]) <= 1.0 and abs(pos[1]) <= 1.0 and pos[2] <= 2.0
            if te or tr:
                break
        if direction[2] == 1:
            assert hit
    env.close()


def test_ceiling_contact_is_penalised_not_terminal_without_net_terminate():
    env = HoopNetAviary(seed=0, net_terminate=False)
    env.reset(seed=3)
    info, te, tr = _fly(env, lambda e: np.array([[0, 0, 1, 1.0]]), 450)
    assert info["net_contacts"] > 0 and not te
    env.close()


def test_net_contact_ends_the_episode_by_default(env):
    env.reset(seed=3)
    info, te, tr = _fly(env, lambda e: np.array([[0, 0, 1, 1.0]]), 450)
    assert te and info["net_ended"] and info["net_contacts"] == 1 and not info["success"]


def test_expert_passes_through_ring(env):
    wins = 0
    for seed in range(4):
        env.reset(seed=seed)
        info, te, tr = _fly(env, _expert, 700)
        wins += bool(info["success"] and te)
    assert wins >= 3


def test_pass_geometry_synthetic(env):
    """Drive _post_physics with synthetic positions: forward / reverse / miss."""
    sp = env.spec
    env.set_wind(0.0)              # a still ring, so the synthetic positions stay exact

    def cross(from_s, to_s, lateral):
        env.reset(seed=0)
        env._dir = 1.0                 # this episode is front -> back (along +ring_n)
        n, c = env.ring_n, env.ring_c
        u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u)
        env._prev_pos = c + n * from_s + u * lateral
        env._prev_s = from_s
        env.pos[0] = c + n * to_s + u * lateral
        env.vel[0] = 0
        env._post_physics()
        return dict(env._ev)

    ok = env.inner_radius - sp.drone_radius
    assert cross(-0.05, 0.05, 0.0)["passed"]
    assert not cross(-0.05, 0.05, ok + 0.05)["passed"]       # outside the clear opening
    ev = cross(0.05, -0.05, 0.0)
    assert ev["wrong"] and not ev["passed"]
    env.set_wind(1.0)


def test_rim_contact_bounces(env):
    env.set_wind(0.0)
    env.reset(seed=0)
    n, c = env.ring_n, env.ring_c
    u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u)
    R = env.inner_radius + env.spec.tube_radius
    env._prev_pos = env.pos[0].copy()
    env.pos[0] = c + u * R          # right on the tube
    env.vel[0] = 0
    env.quat[0] = [0, 0, 0, 1]
    env._post_physics()
    assert env._ev["rim"]
    env.set_wind(1.0)


def test_no_wind_keeps_ring_still():
    e = HoopNetAviary(seed=0, wind=0.0)
    e.reset(seed=4)
    c0 = e.ring_c.copy()
    for _ in range(60):
        e.step(np.array([[0, 0, 0, 0.0]]))
    assert np.allclose(e.ring_c, c0) and np.allclose(e.ring_c, e.ring_c0)
    e.close()


def test_wind_sway_is_bounded_and_seen_in_obs(env):
    env.reset(seed=5)
    env._gust = 1.0                 # full gust for this episode
    cs = []
    for _ in range(150):            # 5 s, longer than the slowest swing
        obs, *_ = env.step(np.array([[0, 0, 0, 0.0]]))
        cs.append(env.ring_c.copy())
        assert np.allclose(obs[0, -3:], env.ring_c - env._getDroneStateVector(0)[:3], atol=1e-5)
        assert np.isclose(np.linalg.norm(env.ring_n), 1.0)
    dev = np.linalg.norm(np.array(cs) - env.ring_c0, axis=1)
    assert dev.max() <= _max_centre_sway(env) + 1e-9
    assert dev.max() > 0.5 * _max_centre_sway(env)      # it really swings


def _max_centre_sway(env):
    """Knot drift plus the arc the centre travels about the knot at full gust."""
    r_out = env.inner_radius + env.spec.tube_radius
    return env.sway_top_amp + r_out * np.radians(env.sway_deg)


def _swing_trace(env, steps=150):
    """Full gust, no shove: the live top and bottom points over `steps` control steps."""
    env._gust = 1.0
    r_out = env.inner_radius + env.spec.tube_radius
    tops, bottoms = [], []
    for _ in range(steps):
        env.step(np.array([[0, 0, 0, 0.0]]))
        env.push_off[:] = 0
        env._update_ring(0.0)                           # re-pose without the shove
        top = env.ring_knot
        tops.append(top)
        bottoms.append(2 * env.ring_c - top)
    return np.array(tops), np.array(bottoms), r_out


def test_wind_swings_about_top_knot(env):
    env.reset(seed=5)
    top0 = env.ring_top(env.ring_c0, env.ring_n0)
    bottom0 = 2 * env.ring_c0 - top0
    tops, bottoms, _ = _swing_trace(env)
    top_dev = np.linalg.norm(tops - top0, axis=1).max()
    bottom_dev = np.linalg.norm(bottoms - bottom0, axis=1).max()
    assert top_dev <= env.sway_top_amp + 1e-6           # the knot only drifts a little
    assert bottom_dev > 3 * top_dev                     # the bottom swings much further


def test_sway_deg_option():
    peaks = {}
    for deg in (0.0, 20.0, 40.0):
        e = HoopNetAviary(seed=0, sway_deg=deg)
        e.reset(seed=5)
        e._gust = 1.0
        cs = []
        for _ in range(150):
            e.step(np.array([[0, 0, 0, 0.0]]))
            cs.append(e.ring_c.copy())
        peaks[deg] = np.linalg.norm(np.array(cs) - e.ring_c0, axis=1).max()
        e.close()
    assert peaks[0.0] < 1e-6                            # no swing: the ring stays put
    assert peaks[40.0] > peaks[20.0] > 0.02
    e = HoopNetAviary(seed=0)
    assert e.sway_deg == e.spec.sway_tilt_deg
    e.set_sway_deg(30)
    assert e.sway_deg == 30.0
    e.close()


def test_rim_hit_shoves_ring_and_it_swings_back(env):
    env.reset(seed=0)
    env._gust = 0.0                 # isolate the shove from the sway
    n, c = env.ring_n, env.ring_c
    u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u)
    R = env.inner_radius + env.spec.tube_radius
    env._prev_pos = env.pos[0].copy()
    env.pos[0] = c + u * (R - 0.05)                     # just inside the tube, moving outwards into it
    env.vel[0] = u * 0.3
    env.quat[0] = [0, 0, 0, 1]
    env._post_physics()
    assert env._ev["rim"]
    peak = 0.0
    for _ in range(10):
        env._update_ring(1 / 30)
        peak = max(peak, float(env.push_off @ u))
    assert peak > 0.01                                    # pushed the way the drone was going
    for _ in range(300):
        env._update_ring(1 / 30)
    assert np.linalg.norm(env.push_off) < 0.1 * peak      # and settled back to its episode pose


def test_lap_mode_requires_return_pass():
    e = HoopNetAviary(seed=0, laps=2, reverse_prob=0.0)
    e.reset(seed=0)
    info, te, tr = _fly(e, _expert, 700)
    assert info["passes"] >= 1
    assert (te and info["passes"] == 2) or (not te and info["passes"] == 1)
    assert np.allclose(info["hoop_normal"], -e.ring_n)       # required direction flipped after lap 1
    e.close()


# ---------------------------------------------------------------- episode variety

def test_fresh_starts_cover_the_whole_floor():
    e = HoopNetAviary(seed=0, carry_start=0.0)
    xs = []
    for i in range(300):
        e.reset(seed=1000 + i)
        xs.append(e.INIT_XYZS[0].copy())
    xs = np.array(xs)
    lim = e.ROOM[0] / 2 - START_MARGIN
    assert np.all(np.abs(xs[:, :2]) <= lim + 1e-9) and np.all(xs[:, 2] < 0.1)
    for sx in (-1, 1):                                   # every quadrant, out near the corners
        for sy in (-1, 1):
            q = xs[(np.sign(xs[:, 0]) == sx) & (np.sign(xs[:, 1]) == sy)]
            assert len(q) > 30
    assert (np.abs(xs[:, 0]) > 0.8).any() and (np.abs(xs[:, 1]) > 0.8).any()
    under = np.linalg.norm(xs[:, :2] - e.ring_c0[:2], axis=1) < 0.25
    assert under.any()                                   # under the ring is allowed now
    e.close()


def test_carry_start_begins_where_the_last_episode_ended():
    e = HoopNetAviary(seed=0, carry_start=1.0)
    e.reset(seed=1)
    for _ in range(3):
        for _ in range(40):                              # fly somewhere for a bit
            e.step(np.array([[0.6, -0.8, 1.0, 1.0]], dtype=np.float32))
        end = e._getDroneStateVector(0)[:3].copy()
        e.reset()
        lim = e.ROOM[0] / 2 - START_MARGIN
        assert np.allclose(e.INIT_XYZS[0][:2], np.clip(end[:2], -lim, lim))
        assert e.INIT_XYZS[0][2] < 0.1 and e.carried_start
        assert np.allclose(e._getDroneStateVector(0)[:2], e.INIT_XYZS[0][:2], atol=1e-3)
    e.close()


def test_reversed_episodes_flip_the_required_direction():
    e = HoopNetAviary(seed=0, reverse_prob=1.0, wind=0.0)
    obs, info = e.reset(seed=2)
    assert e._dir == -1.0
    assert np.allclose(obs[0, -10:-7], -e.ring_n, atol=1e-5)
    wins = 0
    for seed in range(3):
        e.reset(seed=seed)
        info, te, tr = _fly(e, _expert, 700)
        assert info["reversed"]
        wins += bool(info["success"] and te)
    assert wins >= 2                                     # the expert gets through back to front
    e.close()


def test_mirrored_lean_points_the_face_up():
    for p, up in ((1.0, True), (0.0, False)):
        e = HoopNetAviary(seed=0, lean_flip_prob=p, wind=0.0)
        for i in range(10):
            e.reset(seed=i)
            assert (e.ring_n0[2] > 0) == up
        e.close()


def test_variety_probabilities():
    e = HoopNetAviary(seed=0)                            # defaults: 0.7 / 0.5 / 0.3
    n, carried, rev, flip = 400, 0, 0, 0
    e.reset(seed=7)
    for _ in range(n):
        e.reset()
        carried += e.carried_start
        rev += e._dir < 0
        flip += e._lean_flipped
    assert abs(carried / n - 0.7) < 0.07
    assert abs(rev / n - 0.5) < 0.07
    assert abs(flip / n - 0.3) < 0.07
    e.close()


def test_ring_velocity_in_obs():
    e = HoopNetAviary(seed=0, wind=0.0)
    e.reset(seed=4)
    for _ in range(10):
        obs, *_ = e.step(np.array([[0, 0, 0, 0.0]]))
    assert np.allclose(obs[0, -7:-4], 0.0, atol=1e-6)          # still ring
    e.close()
    e = HoopNetAviary(seed=0, wind=1.0)
    e.reset(seed=5)
    e._gust = 1.0
    prev_c, speeds = e.ring_c.copy(), []
    for _ in range(60):
        obs, *_ = e.step(np.array([[0, 0, 0, 0.0]]))
        assert np.allclose(obs[0, -7:-4], (e.ring_c - prev_c) * e.CTRL_FREQ, atol=1e-4)
        prev_c = e.ring_c.copy()
        speeds.append(np.linalg.norm(obs[0, -7:-4]))
    assert max(speeds) > 0.02                                   # swinging ring: seen moving
    e.close()


def _synthetic_step(env, pos):
    env._prev_pos = env.pos[0].copy()
    env.pos[0] = pos
    env.vel[0] = 0
    env._post_physics()
    return env._reward, dict(env._ev)


def test_shaping_telescopes_over_a_closed_path():
    """Fly a closed loop that never crosses the opening: shaping sums to ~zero (only the time
    penalty remains), so circling cannot farm reward."""
    e = HoopNetAviary(seed=0, wind=0.0)
    e.reset(seed=0)
    e._dir = 1.0
    n, c = e.ring_n, e.ring_c
    u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u)
    start = c - n * 0.6 + u * 0.05
    e.pos[0] = start
    e._fresh = True
    e._post_physics()                                           # seed the potential here
    path = [c - n * s + u * l for s, l in [(0.4, 0.1), (0.3, 0.0), (0.2, 0.05), (0.45, 0.2), (0.6, 0.3)]]
    total = sum(_synthetic_step(e, p)[0] for p in path + [start])
    assert abs(total + 6 * 0.005) < 1e-6
    e.close()


def test_endless_laps_need_rearming():
    """laps=0: a pass flips the direction; dithering straight back through does not count until
    the drone has flown clear on the exit side, and the episode never terminates on passes."""
    e = HoopNetAviary(seed=0, laps=0, wind=0.0, reverse_prob=0.0)
    e.reset(seed=0)
    e._dir = 1.0
    n, c = e.ring_n, e.ring_c
    e.pos[0] = c - n * 0.1
    e._fresh = True
    e._post_physics()
    _, ev = _synthetic_step(e, c + n * 0.05)                    # pass 1
    assert ev["passed"] and e.passes == 1 and e._dir == -1.0 and not e._armed
    _, ev = _synthetic_step(e, c - n * 0.05)                    # straight back: not armed
    assert not ev["passed"] and e.passes == 1
    _, ev = _synthetic_step(e, c + n * 0.05)                    # and through again the old way
    assert ev["wrong"]
    _synthetic_step(e, c + n * 0.2)
    _synthetic_step(e, c + n * 0.4)                             # flown clear: re-armed
    assert e._armed
    _synthetic_step(e, c + n * 0.1)
    _, ev = _synthetic_step(e, c - n * 0.05)                    # pass 2, back to front
    assert ev["passed"] and e.passes == 2 and e._dir == 1.0
    assert not e._computeTerminated() and e._computeInfo()["success"]
    e.close()


def test_centred_pass_pays_more(env):
    env.set_wind(0.0)
    rewards = []
    for lateral in (0.0, 0.06):
        env.reset(seed=0)
        env._dir = 1.0
        n, c = env.ring_n, env.ring_c
        u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u)
        env.pos[0] = c - n * 0.05 + u * lateral
        env._fresh = True
        env._post_physics()
        r, ev = _synthetic_step(env, c + n * 0.05 + u * lateral)
        assert ev["passed"]
        rewards.append(r)
    assert rewards[0] > rewards[1] + 3
    env.set_wind(1.0)


# ---------------------------------------------------------------- budget reward (MDP v3)

def _budget_env(**kw):
    e = HoopNetAviary(seed=0, wind=0.0, reward_mode="budget", reverse_prob=0.0, **kw)
    e.reset(seed=0)
    return e


def test_budget_mode_forces_endless_laps_and_net_end():
    e = HoopNetAviary(seed=0, reward_mode="budget", laps=1, net_terminate=False)
    assert e.laps == 0 and e.net_terminate and e.EPISODE_LEN_SEC == 120.0 and e.budget == 3.0
    e.close()
    with pytest.raises(ValueError):
        HoopNetAviary(seed=0, reward_mode="nope")


def test_budget_hovering_runs_out_at_minus_budget():
    e = _budget_env(budget=1.0)
    total, steps = 0.0, 0
    for _ in range(400):
        _, r, te, tr, info = e.step(np.array([[0, 0, 0, 0.0]]))     # sit on the floor
        total += r
        steps += 1
        if te or tr:
            break
    assert te and not tr
    assert steps == 200                                         # 1 / 0.005
    assert abs(total + 1.0) < 1e-6 and abs(info["score"] + 1.0) < 1e-6
    e.close()


def test_budget_pass_rim_and_net():
    e = _budget_env()
    n, c = e.ring_n, e.ring_c
    e.pos[0] = c - n * 0.1
    e._fresh = True
    e._post_physics()
    r, ev = _synthetic_step(e, c + n * 0.05)                    # a pass: +1, keeps going
    assert ev["passed"] and abs(r - (1.0 - 0.005)) < 1e-9
    assert not e._computeTerminated() and e.passes == 1 and abs(e.score - 0.995) < 1e-9
    u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u)
    e.quat[0] = [0, 0, 0, 1]
    r, ev = _synthetic_step(e, c + u * (e.inner_radius + e.spec.tube_radius))   # on the rim
    assert ev["rim"] and abs(r - (-0.005 - 0.05)) < 1e-9
    score_before = e.score
    r, ev = _synthetic_step(e, np.array([0.99, 0.0, 1.0]))      # into the side net
    assert ev["net"] and e._computeTerminated()
    assert abs(r - (-3.0 - score_before)) < 1e-9 and abs(e.score + 3.0) < 1e-9
    e.close()


def test_env_kwargs_from_meta():
    from hoop_net_env import env_kwargs_from_meta
    meta = {"sway_deg": 35.0, "reverse_prob": 0.5, "lean_flip_prob": 0.3, "laps": 0,
            "net_terminate": True, "episode_sec": 120.0, "reward_mode": "budget", "budget": 3.0,
            "algo": "ppo", "carry_start": None}
    kw = env_kwargs_from_meta(meta)
    assert kw == {"sway_deg": 35.0, "reverse_prob": 0.5, "lean_flip_prob": 0.3, "laps": 0,
                  "net_terminate": True, "episode_sec": 120.0, "reward_mode": "budget", "budget": 3.0}
    e = HoopNetAviary(seed=0, **kw)
    assert e.reward_mode == "budget" and e.sway_deg == 35.0
    e.close()
    assert env_kwargs_from_meta({"sway_deg": 20.0}) == {"sway_deg": 20.0}     # older runs


def test_reward_weights_default_to_the_constants_and_reject_unknown_names():
    import hoop_net_env as hn
    w = hn.merge_reward_weights()
    assert w["pass_bonus"] == hn.PASS_BONUS and w["rim_penalty"] == hn.RIM_PENALTY
    assert w["budget_pass"] == hn.BUDGET_PASS
    assert hn.merge_reward_weights({"pass_bonus": 30})["pass_bonus"] == 30.0
    with pytest.raises(ValueError):
        hn.merge_reward_weights({"no_such_term": 1})


def test_reward_weights_change_the_reward(env):
    ev = {"net": False, "rim": False, "wrong": False, "passed": True}
    default = env._shaped_reward(ev, env._prev_phi, 0.0, 0.1)
    e = HoopNetAviary(seed=0, reward_weights={"pass_bonus": 50.0, "time_penalty": 0.0})
    try:
        e.reset(seed=0)
        custom = e._shaped_reward(ev, e._prev_phi, 0.0, 0.1)
    finally:
        e.close()
    import hoop_net_env as hn
    assert default == pytest.approx(hn.PASS_BONUS + hn.CENTRE_BONUS - hn.TIME_PENALTY)
    assert custom == pytest.approx(50.0 + hn.CENTRE_BONUS)


def test_info_reports_this_episodes_swing(env):
    env.set_wind(0.5)
    env.reset(seed=4)
    _, _, _, _, info = env.step(np.zeros((1, 4)))
    assert info["wind_scale"] == 0.5
    assert info["swing_deg"] == pytest.approx(0.5 * env._gust * env.sway_deg)
    assert 1.5 <= info["swing_period_s"] <= 3.0
    env.set_wind(1.0)


def test_env_kwargs_from_meta_carries_the_reward_weights():
    from hoop_net_env import env_kwargs_from_meta
    kw = env_kwargs_from_meta({"reward_mode": "shaped", "reward_weights": {"pass_bonus": 25.0}})
    assert kw["reward_weights"] == {"pass_bonus": 25.0}
