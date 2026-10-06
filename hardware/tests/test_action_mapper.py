import math

import numpy as np

from common.action_mapper import ActionMapper, discrete_to_action
from common.config import load_config
from common.constants import SPEED_LIMIT

CFG = load_config()
M = ActionMapper(CFG)


def test_zero_action_gives_zero_translation():
    rc = M.to_rc(np.zeros(4), 0.0)
    assert rc.as_tuple()[:3] == (0, 0, 0)


def test_world_velocity_roundtrip_at_various_yaw():
    for yaw in (0.0, 0.7, -2.0, math.pi / 2):
        a = np.array([0.6, -0.3, 0.2, 0.9])
        want = M.world_velocity(a)
        rc = M.to_rc(a, yaw)
        got = M.from_rc(rc, yaw)
        assert np.allclose(got, want, atol=1.2 / CFG["control"]["rc_per_mps"] + 1e-6), (yaw, got, want)


def test_full_speed_forward_matches_sim_speed_limit():
    rc = M.to_rc(np.array([1.0, 0, 0, 1.0]), 0.0)
    v = SPEED_LIMIT * CFG["control"]["rc_per_mps"]
    assert abs(rc.fb - round(v)) <= 1
    assert abs(rc.fb) <= CFG["control"]["rc_max"]


def test_sign_of_speed_fraction_ignored_like_sim():
    a = M.world_velocity(np.array([1.0, 0, 0, -0.7]))
    b = M.world_velocity(np.array([1.0, 0, 0, 0.7]))
    assert np.allclose(a, b)


def test_clamp_and_nan_safety():
    big = ActionMapper({**CFG, "control": {**CFG["control"], "rc_per_mps": 1000.0}})
    rc = big.to_rc(np.array([1, 1, 1, 1.0]), 0.0)
    assert all(abs(c) <= CFG["control"]["rc_max"] for c in rc.as_tuple()[:3])
    rc2 = M.to_rc(np.array([np.nan, 0, 0, 1.0]), 0.0)
    assert all(isinstance(c, int) for c in rc2.as_tuple())


def test_yaw_hold_is_damped_with_deadband_and_clamped():
    assert M.to_rc(np.zeros(4), math.radians(2.0)).yaw == 0                    # inside deadband
    big = M.to_rc(np.zeros(4), math.radians(80.0))
    assert abs(big.yaw) == CFG["control"]["yaw_rc_max"]                         # clamped
    # heading 10 deg CCW of the reference -> rotate clockwise (rc yaw positive = clockwise)
    a = M.to_rc(np.zeros(4), math.radians(10.0), 0.0).yaw
    assert a > 0
    # already turning back (clockwise, negative CCW rate) -> damping shrinks the command
    damped = M.to_rc(np.zeros(4), math.radians(10.0), math.radians(-40.0)).yaw
    assert 0 <= damped < a
    # drifting further away (CCW) -> command grows
    assert M.to_rc(np.zeros(4), math.radians(10.0), math.radians(40.0)).yaw > a


def test_discrete_actions_map_to_seven_moves():
    for i in range(7):
        a = discrete_to_action(i)
        assert a.shape == (4,)
    assert discrete_to_action(6)[3] == 0.0           # hover
