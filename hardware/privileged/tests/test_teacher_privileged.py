"""The privileged teacher and the crossing fix (hardware/privileged/)."""
import copy
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[2]
if str(HW) not in sys.path:
    sys.path.insert(0, str(HW))

from onboard import spec  # noqa: E402
from onboard.gym_env import OnboardHoopEnv  # noqa: E402
from onboard.sim import OnboardSim  # noqa: E402
from privileged.sim_patches import fixed_crossing  # noqa: E402
from privileged.teacher import PrivilegedTeacher  # noqa: E402

RING = dict(sway_deg=5.0, wash_deg=15.0, dr=1.0)


def _env(**kw):
    return OnboardHoopEnv(seed=11, profile="measured", laps=1, start_mode="ring", **{**RING, **kw})


def test_the_teacher_is_stateless():
    env, t = _env(), PrivilegedTeacher()
    env.reset(seed=3)
    for _ in range(40):
        env.step(t(env))
    a1, a2 = t(env), t(env)                       # same state, asked twice
    np.testing.assert_array_equal(a1, a2)
    other = PrivilegedTeacher()                    # a fresh instance agrees too
    np.testing.assert_array_equal(other(env), a1)


def test_commands_are_in_range_smooth_and_never_ignored():
    env, t = _env(), PrivilegedTeacher()
    from onboard.sim import PROFILES
    prof = PROFILES["measured"]
    acts, steps = [], []
    for ep in range(3):
        env.reset(seed=20 + ep)
        ep_acts = []
        while True:
            a = t(env)
            acts.append(a)
            ep_acts.append(a)
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        steps.append(np.abs(np.diff(np.array(ep_acts), axis=0)).max())   # within an episode
    acts = np.array(acts)
    assert np.all(np.abs(acts) <= 1.0)
    assert max(steps) <= 0.25 + 1e-6
    yaw = acts[:, spec.ACT_YAW]
    assert not np.any((yaw != 0) & (np.abs(yaw) < prof.yaw_deadband - 1e-6))
    others = acts[:, [spec.ACT_LR, spec.ACT_FB, spec.ACT_UD]]
    assert not np.any((others != 0) & (np.abs(others) < prof.deadband - 1e-6))


def test_the_teacher_flies_through_on_the_measured_profile():
    with fixed_crossing():
        env, t = _env(), PrivilegedTeacher()
        passed = 0
        for ep in range(5):
            env.reset(seed=1000 + ep)
            while True:
                _, _, term, trunc, info = env.step(t(env))
                if term or trunc:
                    passed += info["phase"] == "passed"
                    break
    assert passed >= 4


# the crossing tests now live in onboard/tests/test_crossing.py (the fix is in sim.py)
