import numpy as np

from common.config import load_config
from common.safety import LandingSequence, Level, SafetyInputs, SafetyMonitor, preflight_problems
from common.smoothing import BatteryFilter
from state_estimation.est_types import StateEstimate

CFG = load_config()


def est(**kw):
    d = dict(t=0, pos=np.array([0, 0, 1.0]), vel=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0, yaw_rate=0.0,
             pos_std=0.03, pos_health="OK", yaw_health="OK", tele_age=0.05, cam_age=0.03, ready=True,
             z_smooth=1.0, fields={"bat": 80, "temph": 60})
    d.update(kw)
    return StateEstimate(**d)


def run(mon, seq, dt=0.033, **inp_kw):
    """seq: list of StateEstimate. Returns last decision."""
    t, dec = 0.0, None
    for e in seq:
        t += dt
        dec = mon.evaluate(SafetyInputs(now=t, est=e, airborne=True, **inp_kw))
    return dec


def test_nominal_is_none():
    assert run(SafetyMonitor(CFG), [est()] * 30).level == Level.NONE


def test_telemetry_stale_hold_then_lost_land():
    m = SafetyMonitor(CFG)
    assert run(m, [est(tele_age=0.8)]).level == Level.HOLD
    assert run(SafetyMonitor(CFG), [est(tele_age=2.5)]).level == Level.LAND


def test_hold_fault_promoted_to_land_if_it_persists():
    m = SafetyMonitor(CFG)
    dec = run(m, [est(pos_health="LOST")] * 40)       # 1.3 s: still a hold
    assert dec.level == Level.HOLD
    dec = run(m, [est(pos_health="LOST")] * 80)       # > hold_to_land_s total
    assert dec.level == Level.LAND and dec.latched


def test_hold_clears_when_cause_clears():
    m = SafetyMonitor(CFG)
    assert run(m, [est(tele_age=0.8)]).level == Level.HOLD
    assert run(m, [est()] * 3).level == Level.NONE


def test_tilt_and_fence_hard_and_time_cap_land():
    assert run(SafetyMonitor(CFG), [est(roll=0.6)]).level == Level.LAND
    assert run(SafetyMonitor(CFG), [est()], fence_hard=True).level == Level.LAND
    assert run(SafetyMonitor(CFG), [est()], flight_elapsed=1e3).level == Level.LAND


def test_battery_spike_does_not_trigger_but_sustained_low_does():
    m = SafetyMonitor(CFG)
    seq = [est(fields={"bat": 80, "temph": 60})] * 20
    seq += [est(fields={"bat": 12, "temph": 60})]              # one-sample dip (sag / glitch)
    seq += [est(fields={"bat": 80, "temph": 60})] * 20
    assert run(m, seq).level == Level.NONE
    m2 = SafetyMonitor(CFG)
    assert run(m2, [est(fields={"bat": 15, "temph": 60})] * 150).level == Level.LAND   # 5 s below 20 %


def test_battery_filter_ignores_upward_jumps_and_single_dips():
    f = BatteryFilter(7)
    for v in [80, 80, 80, 80, 80, 80, 80]:
        f.update(v)
    assert f.update(5) == 80                                    # median rejects one low outlier
    for v in [79, 79, 79]:
        f.update(v)
    assert f.value <= 80
    f.update(95)
    assert f.value <= 80                                        # no upward jumps


def test_critical_battery_needs_two_consecutive_readings():
    m = SafetyMonitor(CFG)
    assert run(m, [est(fields={"bat": 8, "temph": 60})]).level == Level.NONE
    assert run(m, [est(fields={"bat": 8, "temph": 60})]).level == Level.LAND


def test_human_kill_land_and_hold_toggle():
    m = SafetyMonitor(CFG)
    m.request(Level.HOLD, "space")
    assert run(m, [est()]).level == Level.HOLD
    m.request(Level.HOLD, "space")
    assert run(m, [est()]).level == Level.NONE                  # SPACE toggles
    m.request(Level.KILL, "x")
    d = run(m, [est()])
    assert d.level == Level.KILL and d.latched
    assert run(m, [est()] * 5).level == Level.KILL              # latched


def test_loop_overrun_and_bad_action_and_axis_mismatch_hold():
    m = SafetyMonitor(CFG)
    last = None
    for _ in range(4):
        last = m.evaluate(SafetyInputs(now=_ * 0.2 + 1, est=est(), airborne=True, loop_dt=0.2))
    assert last.level == Level.HOLD
    assert run(SafetyMonitor(CFG), [est()], action_finite=False).level == Level.HOLD
    assert run(SafetyMonitor(CFG), [est()], axis_mismatch=True).level == Level.HOLD


def test_policy_fighting_fence_triggers_hold():
    m = SafetyMonitor(CFG)
    t, level = 0.0, Level.NONE
    for _ in range(90):
        t += 0.033
        level = m.evaluate(SafetyInputs(now=t, est=est(), airborne=True, fence_clamped=True,
                                        policy_active=True)).level
    assert level >= Level.HOLD


def test_ground_faults_do_not_land_a_grounded_drone():
    m = SafetyMonitor(CFG)
    d = m.evaluate(SafetyInputs(now=1.0, est=est(tele_age=9.0, pos_health="LOST"), airborne=False))
    assert d.level == Level.NONE


# -- landing escalation ----------------------------------------------------------------
def test_landing_normal_completes_without_emergency():
    L = LandingSequence(CFG)
    assert L.start(0.0) == ["land"]
    acts, t = [], 0.0
    for i in range(300):
        t += 0.033
        h = max(0.0, 1.0 - 0.3 * t)
        acts += L.update(t, h, -0.3 if h > 0 else 0.0)
    assert L.landed and "emergency" not in acts


def test_landing_reissues_land_and_never_cuts_motors_high_up():
    L = LandingSequence(CFG)
    L.start(0.0)
    acts, t = [], 0.0
    for i in range(1000):                                       # 33 s stuck at 1.5 m
        t += 0.033
        acts += L.update(t, 1.5, 0.0)
    assert acts.count("land") >= 5
    assert "emergency" not in acts


def test_landing_cuts_motors_when_low_and_stuck():
    L = LandingSequence(CFG)
    L.start(0.0)
    acts, t = [], 0.0
    for i in range(300):
        t += 0.033
        acts += L.update(t, 0.25, 0.2)                          # hovering at 25 cm, never settles
    assert acts.count("emergency") == 1


def test_landing_blind_height_uses_time_only():
    L = LandingSequence(CFG)
    L.start(0.0)
    t, acts = 0.0, []
    for i in range(300):                                        # 10 s: not yet
        t += 0.033
        acts += L.update(t, None, 0.0)
    assert "emergency" not in acts
    for i in range(200):
        t += 0.033
        acts += L.update(t, None, 0.0)
    assert acts.count("emergency") == 1


def test_noisy_height_does_not_declare_landed_early():
    rng = np.random.default_rng(0)
    L = LandingSequence(CFG)
    L.start(0.0)
    t = 0.0
    for i in range(60):                                         # 2 s at 0.9 m with a few zero glitches
        t += 0.033
        h = 0.0 if i in (10, 11) else 0.9 + rng.normal(0, 0.02)
        L.update(t, h, 0.0)
    assert not L.landed


# -- preflight --------------------------------------------------------------------------
def test_preflight_refuses_uncalibrated_real_hardware_and_low_battery():
    e = est()
    assert any("calibrated" in p for p in preflight_problems(CFG, e, True, [80] * 5))
    assert preflight_problems(CFG, e, False, [80] * 5) == []
    assert any("battery" in p for p in preflight_problems(CFG, e, False, [20] * 5))
    assert any("5 battery" in p for p in preflight_problems(CFG, e, False, [80] * 2))
    assert any("not ready" in p for p in preflight_problems(CFG, est(ready=False), False, [80] * 5))


def test_grounded_drone_is_declared_landed_even_with_noisy_height_and_velocity():
    """Regression: a landed drone was motor-cut because the filter's noisy vz never settled."""
    rng = np.random.default_rng(1)
    L = LandingSequence(CFG)
    L.start(0.0)
    acts, t = [], 0.0
    for i in range(400):
        t += 0.033
        h = max(0.0, 1.0 - 0.3 * t) + (abs(rng.normal(0, 0.03)) if t > 3.3 else 0.0)   # on the ground by 3.3 s, noisy
        acts += L.update(t, h * 0.6 if t > 3.3 else h, vz=rng.normal(0, 0.15))
    assert L.landed and "emergency" not in acts
