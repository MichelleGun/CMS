import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoop_net_env import HoopNetAviary, HoopSpec, START_Z, ring_normal  # noqa: E402


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
    assert obs.shape == (1, 78)
    assert np.allclose(obs[0, -3:], env.ring_c - env._getDroneStateVector(0)[:3], atol=1e-5)
    assert np.isclose(np.linalg.norm(obs[0, -6:-3]), 1.0, atol=1e-5)


@pytest.mark.parametrize("seed", range(8))
def test_always_starts_on_floor_and_ring_jitter_in_range(env, seed):
    env.reset(seed=seed)
    z = env._getDroneStateVector(0)[2]
    assert z < 0.1
    sp = env.spec
    assert np.all(np.abs(env.ring_c0 - np.array(sp.centre)) <= np.array(sp.centre_jitter) + 1e-9)
    assert 1 - sp.radius_jitter_frac - 1e-9 <= env.ring_r_factor <= 1 + sp.radius_jitter_frac + 1e-9
    # episode facing within the yaw/tilt jitter of the nominal (scan-measured) facing
    nominal = ring_normal(sp.normal_yaw_deg, sp.normal_tilt_deg)
    max_dev = np.radians(sp.yaw_jitter_deg + sp.tilt_jitter_deg)
    assert np.degrees(np.arccos(np.clip(env.ring_n0 @ nominal, -1, 1))) <= np.degrees(max_dev) + 1e-6
    # live pose stays within the wind sway of the episode pose
    assert np.linalg.norm(env.ring_c - env.ring_c0) <= _max_centre_sway(env) + 1e-9


def test_start_override_is_clamped_to_floor():
    e = HoopNetAviary(seed=0, start_pos=[0.3, -0.2, 1.5])
    e.reset(seed=0)
    assert e._getDroneStateVector(0)[2] < 0.1
    e.close()


def test_net_bounce_keeps_drone_inside(env):
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


def test_ceiling_contact_is_penalised_not_terminal(env):
    env.reset(seed=3)
    info, te, tr = _fly(env, lambda e: np.array([[0, 0, 1, 1.0]]), 450)
    assert info["net_contacts"] > 0 and not te


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
    e = HoopNetAviary(seed=0, laps=2)
    e.reset(seed=0)
    info, te, tr = _fly(e, _expert, 700)
    assert info["passes"] >= 1
    assert (te and info["passes"] == 2) or (not te and info["passes"] == 1)
    assert np.allclose(info["hoop_normal"], -e.ring_n)       # required direction flipped after lap 1
    e.close()
