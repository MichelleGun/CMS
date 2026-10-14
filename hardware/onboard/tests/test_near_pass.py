"""Near-pass starts (training curriculum for the loop): where they put the drone, and that they
change nothing when off."""
import numpy as np

from onboard.gym_env import OnboardHoopEnv
from onboard.sim import CAGE_XY

KW = dict(profile="measured", dr=1.0, laps=0, lap_dir="same", start_mode="zones", wash_deg=15.0)


def test_a_near_pass_start_is_lined_up_in_front_of_the_ring_and_inside_the_cage():
    env = OnboardHoopEnv(seed=3, near_pass_p=1.0, **KW)
    for _ in range(20):
        env.reset()
        s = env.sim
        assert s.start_type == "near_pass"
        d = s.pos - s.ring.centre
        n = s.ring.normal
        axial = float(np.dot(d, n))
        radial = float(np.linalg.norm(d - axial * n))
        assert 0.3 < abs(axial) < 0.6 and radial < 0.1
        assert np.all(np.abs(s.pos[:2]) <= CAGE_XY)
        assert float(np.dot(s.forward[:2], -np.sign(axial) * n[:2])) > 0.9   # facing through it
        assert env._dir == (1 if axial < 0 else -1)                          # this way through counts
        assert env.max_ticks == env.hard_ticks


def test_off_by_default_and_the_layouts_are_unchanged():
    a, b = OnboardHoopEnv(seed=5, **KW), OnboardHoopEnv(seed=5, near_pass_p=0.0, **KW)
    for _ in range(5):
        oa, _ = a.reset()
        ob, _ = b.reset()
        assert np.allclose(oa, ob) and np.allclose(a.sim.pos, b.sim.pos)
        assert a.sim.start_type != "near_pass"


def test_a_share_of_episodes_only():
    env = OnboardHoopEnv(seed=9, near_pass_p=0.5, **KW)
    kinds = []
    for _ in range(60):
        env.reset()
        kinds.append(env.sim.start_type == "near_pass")
    assert 0.3 < np.mean(kinds) < 0.7
