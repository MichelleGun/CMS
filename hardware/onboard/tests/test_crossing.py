"""The ring-plane crossing check (onboard/sim.py _check_crossing), fixed 2026-10-13: a centred
pass while the prop-washed ring swings back towards the drone used to go unrecorded."""
import copy

import numpy as np

from onboard.sim import OnboardSim


def _sim():
    s = OnboardSim(seed=0, profile="ideal", start_pos=[-0.004, 0.0, 1.14], start_yaw_deg=0.0)
    s.ring = s.ring0 = copy.deepcopy(s.ring)
    s.ring.centre[:] = [0.0, 0.0, s.pos[2]]
    s.ring.normal_deg = 0.0
    return s


def _step(s, drone_dx, ring_dx):
    prev = s.pos.copy()
    s._ring_before = copy.deepcopy(s.ring)          # what _update_ring stashes every step
    s.pos = s.pos + np.array([drone_dx, 0.0, 0.0])
    s.ring.centre[0] += ring_dx
    s._check_crossing(prev)
    return s.outcome


def test_a_centred_pass_is_recorded_while_the_ring_swings_back():
    o = _step(_sim(), 0.008, -0.006)                 # drone 8 mm through, ring 6 mm back
    assert o.done and o.passed


def test_a_ring_sweeping_past_a_still_drone_is_not_a_pass():
    assert not _step(_sim(), 0.0, -0.02).done


def test_a_ring_sweeping_past_a_retreating_drone_is_not_a_pass():
    assert not _step(_sim(), -0.002, -0.02).done


def test_an_ordinary_pass_with_a_still_ring():
    o = _step(_sim(), 0.008, 0.0)
    assert o.done and o.passed and o.direction == 1


def test_a_stress_profile_keeps_its_stressed_lag_through_domain_randomisation():
    """At dr 1 every non-kept field is re-drawn, so stress tests name what must survive."""
    from dataclasses import replace
    import numpy as np
    from onboard.sim import PROFILES, sample_profile
    base = PROFILES["measured"]
    lag = replace(base, tau_accel=9.0, tau_decel=9.0, keep=("tau_accel", "tau_decel"))
    s = sample_profile(np.random.default_rng(0), 1.0, lag)
    assert s.tau_accel == 9.0 and s.tau_decel == 9.0
    plain = sample_profile(np.random.default_rng(0), 1.0, replace(base, tau_accel=9.0))
    assert plain.tau_accel != 9.0          # without keep, dr 1 re-draws it (the old silent behaviour)
