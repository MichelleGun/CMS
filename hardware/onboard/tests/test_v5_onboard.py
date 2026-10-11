"""Onboard v5: start zones, same-side laps (round the outside), the streak bonus, and the heading
memory (pass detection from telemetry alone)."""
import numpy as np

from onboard import spec
from onboard.gym_env import OnboardHoopEnv, R_PASSED
from onboard.obs import ObsBuilder
from onboard.ring_detect import RingObs
from onboard.sim import CAGE_XY, ZONE_CELLS, OnboardSim, Ring


def test_zones_cycle_in_order_and_land_in_their_cell():
    sim = OnboardSim(seed=5, start_mode="zones")
    seen = []
    for k in range(18):
        sim.reset(seed=200 + k)
        z = int(sim.start_type.split("_")[1])
        seen.append(z)
        x, y, h = sim.pos
        lim = CAGE_XY - 0.12
        if z <= 8:
            w = 2 * lim / 3
            col, row = ZONE_CELLS[z - 1]
            assert abs(x - col * w) <= w / 2 + 1e-9 and abs(y - row * w) <= w / 2 + 1e-9
        else:
            assert np.hypot(x - sim.ring.centre[0], y - sim.ring.centre[1]) <= 0.22
            assert h < sim.ring.centre[2] - sim.ring.outer_m / 2 - sim.drone_radius_m
        assert sim.rim_distance() > 0                      # never starts touching the noodle
    assert sorted(seen) == sorted(list(range(1, 10)) * 2)
    assert all((b - a) % 9 == 1 for a, b in zip(seen, seen[1:]))


def _lap_env():
    env = OnboardHoopEnv(ring=Ring(np.array([0.0, 0.0, 1.0]), 0.0), profile="ideal", laps=0,
                         lap_dir="same")
    env.reset(seed=0)
    env.sim.pos = np.array([-0.5, 0.0, 1.0])
    env.sim.yaw_deg = 0.0
    env.sim.vel[:] = 0.0
    env._dir, env._armed = 1, True
    env._reset_shaping()
    return env


def _fly(env, fb, ticks, until):
    out = []
    for _ in range(ticks):
        _o, r, term, trunc, info = env.step(np.array([0.0, fb, 0.0, 0.0]))
        out.append((r, info))
        if term or trunc or until(info):
            break
    return out


def _go_round(env):
    """Teleport the way round: beside the ring on the exit side, then across the plane outside."""
    sim = env.sim
    sim.pos = np.array([0.3, 0.6, 1.0])
    sim.vel[:] = 0.0
    env.step(np.zeros(4))
    prev = sim.pos.copy()
    sim.pos = np.array([-0.2, 0.6, 1.0])
    sim._check_crossing(prev)                              # crossed the plane clear of the noodle
    env.step(np.zeros(4))


def test_same_side_lap_needs_the_way_round_and_pays_a_streak():
    env = _lap_env()
    steps = _fly(env, 1, 200, until=lambda i: i["phase"] == "passed")
    r1 = steps[-1][0]
    assert steps[-1][1]["passes"] == 1 and env._dir == 1 and not env._returned
    _fly(env, 1, 60, until=lambda i: env.sim.pos[0] > 0.4)
    assert not env._armed
    _go_round(env)
    assert env._returned
    env.sim.pos = np.array([-0.5, 0.0, 1.0])               # in front, facing it again
    env.sim.yaw_deg = 0.0
    env.sim.vel[:] = 0.0
    env.step(np.zeros(4))
    assert env._armed
    steps = _fly(env, 1, 200, until=lambda i: i["phase"] == "passed")
    r2 = steps[-1][0]
    assert steps[-1][1]["passes"] == 2 and env.streak_mult == 1.5
    assert r2 - r1 > 0.4 * R_PASSED                        # the 2nd pass pays x1.5
    env.close()


def test_same_side_back_through_the_opening_is_wrong_way():
    env = _lap_env()
    _fly(env, 1, 200, until=lambda i: i["phase"] == "passed")
    _fly(env, 1, 60, until=lambda i: env.sim.pos[0] > 0.4)
    env.sim.yaw_deg = 180.0                                # turned round, camera on the ring
    env.sim.vel[:] = 0.0
    steps = _fly(env, 1, 300, until=lambda i: i["phase"] != "flying")
    assert steps[-1][1]["phase"] == "wrong_way" and env.passes == 1
    assert not env._returned and not env._armed
    env.close()


def test_heading_memory_flips_over_a_turn_round():
    b = ObsBuilder()
    s = spec.OBS_SLICES
    close = RingObs(found=True, fresh=True, cx=0.0, cy=0.0, scale=0.5, axis_ratio=1.0,
                    range_m=0.4, age=0)
    lost = RingObs(found=False, fresh=False, age=spec.LOCK_MAX_AGE, range_m=float("inf"))
    t = 0.0
    o = b.build(close, {"vgx": 24, "yaw": 10}, t)
    assert np.allclose(o[s["pass_heading"]], 0) and o[s["streak"]] == 0
    for _ in range(40):                                    # blind push well past 0.4 + 0.15 m
        t += spec.CTRL_DT
        o = b.build(lost, {"vgx": 24, "yaw": 10}, t)
    assert o[s["streak"]] == 0.25 and np.allclose(o[s["pass_heading"]], (1, 0), atol=1e-6)
    t += spec.CTRL_DT
    o = b.build(lost, {"vgx": 0, "yaw": -170}, t)          # turned round
    assert np.allclose(o[s["pass_heading"]], (-1, 0), atol=1e-6)


def test_detected_passes_match_the_sim():
    """The telemetry-only pass detector agrees with the simulator's own count on expert flights."""
    from onboard.expert import ExpertConfig, ExpertPolicy
    env = OnboardHoopEnv(seed=3, laps=0, lap_dir="same", start_mode="zones")
    pol = ExpertPolicy(ExpertConfig(laps=True, lap_same=True))
    agree = total = 0
    for e in range(9):
        obs, _ = env.reset(seed=300 + e)
        pol.reset()
        for _ in range(900):
            obs, _r, te, tr, info = env.step(pol(obs))
            if te or tr:
                break
        total += 1
        agree += int(env.builder.passes.passes == env.passes)
    assert agree / total >= 0.8


def test_same_side_expert_strings_passes_together():
    from onboard.make_demos import generate
    _rec, tally = generate(episodes=18, seed=0, laps=0, lap_dir="same", start_mode="zones",
                           verbose=False)
    assert tally["max_passes"] >= 2 and tally["mean_passes"] >= 0.8
    assert sum(z["passed"] > 0 for z in tally["by_start"].values()) >= 7   # most zones pass
