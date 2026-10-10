"""Robustness of the onboard task: domain randomisation, a swinging ring, hard starts (behind /
beside / below / anywhere), the open ring plane, and the demo augmentation of the trainer."""
import numpy as np
import pytest

from onboard import spec
from onboard.gym_env import OnboardHoopEnv
from onboard.sim import CAGE_XY, PROFILES, OnboardSim, Ring


def test_no_randomisation_means_the_measured_task():
    sim = OnboardSim(seed=3)
    assert sim.profile is PROFILES["typical"] and sim.speed_scale == 1.0
    assert sim.start_type == "front" and sim._swing is None
    c0 = sim.ring.centre.copy()
    for _ in range(30):
        sim.step([0, 0, 0, 0])
    assert np.array_equal(sim.ring.centre, c0)                 # a still ring


def test_randomised_drones_differ_between_episodes():
    sim = OnboardSim(seed=0, dr=1.0)
    seen = set()
    for s in range(6):
        sim.reset(seed=s)
        seen.add((round(sim.speed_scale, 3), round(sim.profile.tau_accel, 3)))
        assert 0.75 <= sim.speed_scale <= 1.25
    assert len(seen) == 6


def test_the_ring_swings_on_its_rope():
    sim = OnboardSim(seed=0, dr=1.0)
    for s in range(10):                                        # every windy episode, not a lucky one
        sim.reset(seed=s)
        sim.pos = np.array([-0.9, -0.9, 0.5])                  # out of the ring's way
        centres = []
        for _ in range(45):                                    # 3 s, at least one full period
            sim.step([0, 0, 0, 0])
            centres.append(sim.ring.centre.copy())
        centres = np.array(centres)
        sideways = np.linalg.norm(centres[:, :2] - sim.ring0.centre[:2], axis=1).max()
        assert sideways > spec.RING_CLEARANCE_M / 2             # matters against the 8 cm budget
        assert sideways < 0.20
        assert np.all(centres[:, 2] >= sim.ring0.centre[2] - 1e-9)  # a pendulum only rises


def test_a_ring_swinging_past_a_hovering_drone_is_never_a_pass():
    """Regression: the plane crossing used to compare the drone's side against the ring's
    PREVIOUS pose, so parking in the opening while the ring swung scored +20 'passes'."""
    sim = OnboardSim(seed=0, dr=1.0, profile="ideal")
    for s in range(30):
        sim.reset(seed=s)
        sim.profile = PROFILES["ideal"]
        sim.pos = sim.ring0.centre.copy()                      # parked right in the opening
        for _ in range(90):
            sim.step([0, 0, 0, 0])
            sim.vel[:] = 0.0
            sim.pos = sim.ring0.centre.copy()
            assert not sim.outcome.passed
            if sim.outcome.done:
                break


def test_every_windy_episode_has_a_real_gust():
    from onboard.sim import SWING_GUST_MIN
    sim = OnboardSim(seed=0, dr=1.0)
    for s in range(30):
        sim.reset(seed=s)
        assert sim._swing["gust"] >= SWING_GUST_MIN


def test_the_swinging_ring_can_hit_a_hovering_drone():
    """A drone holding still right beside the noodle is struck when the ring swings into it."""
    hit = tried = 0
    sim = OnboardSim(seed=0, dr=1.0, profile="ideal", open_plane=True)
    for s in range(60):
        sim.reset(seed=s)
        sim.profile = PROFILES["ideal"]
        c, n = sim.ring0.centre, sim.ring0.normal
        u = np.array([-n[1], n[0], 0.0])
        wind = np.array([np.cos(sim._swing["dir"]), np.sin(sim._swing["dir"]), 0.0])
        if abs(wind @ u) < 0.8:                                 # swinging in the ring's own plane
            continue
        tried += 1
        side = np.sign(wind @ u)
        sim.pos = c + side * u * (sim.ring.radius + sim.ring.noodle_m + sim.drone_radius_m + 0.02)
        sim._prev_side = sim._plane_side()
        for _ in range(60):
            sim.step([0, 0, 0, 0])
            sim.vel[:] = 0.0
            if sim.outcome.done:
                break
        if sim.outcome.struck:
            hit += 1
    assert tried >= 5 and hit >= 0.9 * tried


@pytest.mark.parametrize("kind", ["behind", "beside", "below", "anywhere"])
def test_hard_starts_land_where_they_should(kind):
    sim = OnboardSim(seed=0, hard_start_p=1.0)
    found = 0
    for s in range(200):
        sim.reset(seed=s)
        if sim.start_type != kind:
            continue
        found += 1
        c, n = sim.ring.centre, sim.ring.normal
        d = sim.pos - c
        axial = float(d @ n)
        assert abs(sim.pos[0]) <= CAGE_XY and abs(sim.pos[1]) <= CAGE_XY
        if kind == "behind":
            assert axial > 0.2 or abs(sim.pos[0]) >= CAGE_XY - 0.06   # unless the net clipped it
        elif kind == "beside":
            assert abs(axial) <= 0.21
        elif kind == "below":
            assert sim.pos[2] <= 0.9 and abs(axial) <= 0.26
        else:
            assert np.hypot(*d[:2]) > 0.24
    assert found > 10


def test_hard_starts_get_longer_episodes():
    env = OnboardHoopEnv(seed=0, hard_start_p=1.0)
    env.reset(seed=0)
    assert env.max_ticks == int(40 * spec.CTRL_HZ)
    env = OnboardHoopEnv(seed=0)
    env.reset(seed=0)
    assert env.max_ticks == int(30 * spec.CTRL_HZ)


def _fly_under(open_plane):
    ring = Ring(np.array([0.0, 0.0, 1.2]), 0.0)
    sim = OnboardSim(ring=ring, profile="ideal", start_pos=[-0.4, 0.0, 0.6], start_yaw_deg=0.0,
                     open_plane=open_plane)
    for _ in range(120):
        sim.step([0, 1, 0, 0])
        if sim.outcome.crossed:
            break
    return sim


def test_flying_under_the_ring_is_no_outcome_with_an_open_plane():
    assert _fly_under(open_plane=False).outcome.missed          # the old infinite wall
    sim = _fly_under(open_plane=True)
    assert not sim.outcome.crossed and sim.pos[0] > 0.0          # went under it and flew on


def test_a_pass_from_behind_counts():
    ring = Ring(np.array([0.0, 0.0, 1.0]), 0.0)
    sim = OnboardSim(ring=ring, profile="ideal", start_pos=[0.5, 0.0, 1.0], start_yaw_deg=180.0,
                     open_plane=True)
    for _ in range(200):
        sim.step([0, 1, 0, 0])
        if sim.outcome.crossed:
            break
    assert sim.outcome.passed


def test_the_search_bonus_is_paid_once():
    from onboard.gym_env import R_FOUND
    env = OnboardHoopEnv(seed=0, hard_start_p=1.0)
    for s in range(100):                       # an episode that starts facing away
        env.reset(seed=s)
        if not env._found_once:
            break
    assert not env._found_once
    # put it squarely in front of the ring, facing it: the ring comes into view
    sim = env.sim
    sim.pos = sim.ring.centre - 0.9 * sim.ring.normal
    sim.yaw_deg = float(sim.ring.normal_deg)
    sim._prev_pose = (sim.pos.copy(), sim.yaw_deg)
    sim._prev_side = sim._plane_side()
    env._reset_shaping()
    rewards = [env.step(np.zeros(4))[1] for _ in range(10)]
    assert env._found_once                                      # seen within 10 ticks
    assert sum(r > R_FOUND / 2 for r in rewards) == 1           # paid once, on the first sighting


def test_mirror_augmentation_flips_exactly_the_lateral_terms():
    from onboard.train_onboard import mirror_signs
    obs, act = mirror_signs()
    sl = spec.OBS_SLICES
    flipped = set(np.flatnonzero(obs < 0))
    want = {sl["centre"].start, sl["last_centre"].start, sl["vel"].start + 1,
            sl["att"].start + 1, sl["att"].start + 2}
    want |= {sl["actions"].start + k * spec.ACT_DIM + j
             for k in range(spec.ACT_BUFFER) for j in (spec.ACT_LR, spec.ACT_YAW)}
    assert flipped == want
    assert set(np.flatnonzero(act < 0)) == {spec.ACT_LR, spec.ACT_YAW}
    assert np.all(obs * obs == 1)                                # its own inverse


def test_ramp_counts_from_this_runs_start():
    from onboard.train_onboard import RampCallback

    class Vec:
        num_envs = 1
        calls = []

        def env_method(self, name, value):
            self.calls.append(value)

    vec = Vec()
    cb = RampCallback(horizon=1000, every=1)
    cb.model = type("M", (), {"get_env": lambda self: vec, "num_timesteps": 1_700_000})()
    cb.num_timesteps = 1_700_000                    # continued from a 1.7M-step run
    cb._on_training_start()
    assert vec.calls[-1] == 0.0                     # starts easy, not at full strength
    cb.num_timesteps += 500
    cb._on_step()
    assert vec.calls[-1] == 0.5
