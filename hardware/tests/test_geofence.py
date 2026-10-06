import numpy as np

from common.config import load_config
from common.constants import SPEED_LIMIT
from common.geofence import Geofence, action_to_velocity

CFG = load_config()


def test_inside_cage_action_untouched():
    g = Geofence(CFG["fence"])
    a = np.array([1.0, 0.0, 0.0, 1.0])
    r = g.apply(a, [0.0, 0.0, 1.0])
    assert not r.clamped and r.zone == "ok"
    assert np.allclose(r.action, a)


def test_soft_zone_tapers_outward_motion_only():
    g = Geofence(CFG["fence"])
    a = np.array([1.0, 0.0, 0.0, 1.0])
    r = g.apply(a, [0.80, 0.0, 1.0])               # 20 cm from the +x wall: halfway through the taper
    assert r.clamped and r.zone == "soft"
    v = action_to_velocity(r.action)
    assert 0 < v[0] < SPEED_LIMIT * 0.6
    assert np.isclose(v[0], SPEED_LIMIT * 0.5, atol=1e-6)
    # at the zero-speed line outward motion is fully stopped
    assert abs(action_to_velocity(g.apply(a, [0.90, 0.0, 1.0]).action)[0]) < 1e-9
    # moving inward is never clamped
    r2 = g.apply(np.array([-1.0, 0.0, 0.0, 1.0]), [0.93, 0.0, 1.0])
    assert not r2.clamped


def test_outward_component_removed_at_wall_keeps_tangential():
    g = Geofence(CFG["fence"])
    a = np.array([1.0, 1.0, 0.0, 1.0])
    r = g.apply(a, [1.0, 0.0, 1.0])                # exactly on the +x wall
    v = action_to_velocity(r.action)
    assert v[0] <= 0 and abs(v[0]) <= CFG["fence"]["push_max"] + 1e-9   # never outward; nudged back in
    assert v[1] > 0                                                      # tangential motion survives


def test_hard_zone_beyond_wall():
    g = Geofence(CFG["fence"])
    assert g.apply(np.zeros(4), [1.2, 0.0, 1.0]).hard
    assert not g.apply(np.zeros(4), [1.05, 0.0, 1.0]).hard


def test_clamped_action_is_a_valid_sim_action():
    g = Geofence(CFG["fence"])
    r = g.apply(np.array([0.3, 0.9, -0.2, 0.8]), [0.95, 0.95, 0.2])
    assert r.action.shape == (4,) and np.all(np.abs(r.action) <= 1.0 + 1e-9)
    assert np.all(np.isfinite(r.action))


def test_drifting_past_zero_line_gets_pushed_back_even_with_zero_command():
    g = Geofence(CFG["fence"])
    r = g.apply(np.zeros(4), [0.97, 0.0, 1.0])       # drifting toward the +x wall while hovering
    v = action_to_velocity(r.action)
    assert r.clamped and v[0] < 0                    # nudged inward
    assert abs(v[0]) <= CFG["fence"]["push_max"] + 1e-9
    r2 = g.apply(np.zeros(4), [-0.97, 0.0, 1.0])
    assert action_to_velocity(r2.action)[0] > 0      # and symmetric on the other wall
    assert not g.apply(np.zeros(4), [0.5, 0.0, 1.0]).clamped
