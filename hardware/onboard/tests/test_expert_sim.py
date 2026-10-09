"""
The scripted expert, the simulator and the Gym environment.

The headline test is that the expert flies through the ring. The rest pin down the
behaviours that make it safe to put in the air, and one regression for a control bug that
would otherwise be invisible.
"""
import numpy as np
import pytest

from onboard import spec
from onboard.expert import ABORT, COMMIT, DONE, SEARCH, Expert, ExpertConfig
from onboard.obs import ObsBuilder
from onboard.ring_detect import RingObs
from onboard.sim import PROFILES, OnboardSim, Ring


def fly(seed=0, profile="typical", cfg=None, render=False, max_ticks=500):
    sim = OnboardSim(profile=profile, seed=seed, render=render)
    ex = Expert(cfg)
    ring = sim.observe()
    for i in range(max_ticks):
        action, _info = ex(ring, sim.state())
        ring = sim.step(action)
        if ex.finished or sim.out_of_cage() or sim.outcome.crossed:
            break
    return sim, ex, i


def pass_rate(profile="typical", n=25, cfg=None, render=False):
    return sum(fly(s, profile, cfg, render)[0].outcome.passed for s in range(n)) / n


# --------------------------------------------------------------- the headline

@pytest.mark.parametrize("profile,floor", [("ideal", 0.9), ("typical", 0.85)])
def test_the_expert_flies_through_the_ring(profile, floor):
    assert pass_rate(profile) >= floor


def test_the_expert_still_mostly_works_on_the_harsh_profile():
    """Deliberately brutal: 180 ms command delay, a big dead-band, 15 % detection misses.
    A lower bar, but a total collapse here would mean the controller only works on a
    drone that behaves perfectly, which no real one does."""
    assert pass_rate("harsh", n=25) >= 0.4


def test_it_never_strikes_the_ring_more_often_than_it_passes():
    outcomes = [fly(s)[0].outcome for s in range(25)]
    assert sum(o.struck for o in outcomes) < sum(o.passed for o in outcomes)


def test_the_real_detector_in_the_loop_also_passes():
    """render=True puts the actual OpenCV detector and tracker in the control loop, so
    this covers the thresholds and the clipped-ring path, not just the analytic model."""
    assert pass_rate(n=6, render=True) >= 0.5


def test_it_stays_inside_the_cage():
    assert not any(fly(s)[0].out_of_cage() for s in range(25))


# --------------------------------------------------------------- regressions

def test_the_forward_setpoint_is_strictly_closer_than_the_commit_threshold():
    """Regression. When these were equal, the proportional forward term decayed to zero
    exactly at the commit threshold, so the drone asymptoted just outside it and only
    committed when noise nudged the estimate across. The measured symptom was 0/30 on the
    noise-free profile and 14/30 on the noisy one -- better sensors made it worse."""
    c = ExpertConfig()
    assert c.approach_target_m < c.commit_range_m


def test_the_noise_free_profile_is_not_worse_than_the_noisy_one():
    """The direct behavioural form of the same regression."""
    assert pass_rate("ideal", n=15) >= pass_rate("typical", n=15)


# --------------------------------------------------------------- phases

def _ring(**kw):
    base = dict(found=True, fresh=True, cx=0.0, cy=0.0, major_px=150.0, minor_px=140.0,
                axis_ratio=0.93, scale=0.3, range_m=1.0, age=0)
    base.update(kw)
    return RingObs(**base)


def test_without_a_lock_it_searches_by_yawing():
    ex = Expert()
    action, info = ex(RingObs(found=False), {})
    assert info.phase == SEARCH
    assert abs(action[spec.ACT_YAW]) > 0
    assert action[spec.ACT_FB] == 0, "never drive forward while blind"


def test_searching_gives_up_rather_than_spinning_for_ever():
    ex = Expert(ExpertConfig(search_max_ticks=5))
    for _ in range(7):
        _a, info = ex(RingObs(found=False), {})
    assert info.phase == ABORT


def test_repeated_loss_of_lock_aborts():
    ex = Expert(ExpertConfig(max_aborts=1, search_max_ticks=10_000))
    for _ in range(400):
        ex(_ring(), {})
        for _ in range(spec.LOCK_MAX_AGE + 2):
            _a, info = ex(RingObs(found=False), {})
        if info.phase == ABORT:
            return
    pytest.fail("never aborted despite losing the lock repeatedly")


def test_commit_is_blind_and_bounded():
    """Once committed it must ignore detections entirely, and it must stop."""
    ex = Expert()
    # Derived from the config, not hardcoded: this test asserted a 0.5 m range, which
    # stopped triggering a commit the moment commit_range_m was recalibrated to the
    # measured ring. The test then failed for a reason that had nothing to do with
    # whether the push is blind and bounded, which is what it is actually about.
    inside = ex.cfg.commit_range_m - 0.05
    for _ in range(3):
        ex(_ring(range_m=inside, cx=0.0, cy=0.0, axis_ratio=0.95), {})
    assert ex.phase == COMMIT

    seen = []
    for _ in range(ex.cfg.commit_ticks_max + ex.cfg.settle_ticks + 5):
        a, info = ex(_ring(cx=0.9, cy=0.9, range_m=0.1), {})   # wildly off: ignored
        seen.append(a[spec.ACT_FB])
        if info.phase == DONE:
            break
    assert ex.phase == DONE, "the blind push must terminate"
    assert max(seen) == pytest.approx(ex.cfg.commit_fb)
    assert seen[-1] == 0.0, "it settles to a hover before declaring done"


def test_commit_duration_follows_the_measured_range_and_assumed_speed():
    """Derived, not hardcoded: recalibrating commit_speed_mps must change the push."""
    slow = Expert(ExpertConfig(commit_speed_mps=0.12))
    fast = Expert(ExpertConfig(commit_speed_mps=0.40))
    assert slow._commit_ticks_for(0.55) > fast._commit_ticks_for(0.55)
    # Compared at the default speed: at 0.12 m/s both of these saturate the 5 s cap.
    mid = Expert()
    assert mid._commit_ticks_for(0.9) > mid._commit_ticks_for(0.3)


def test_commit_duration_is_clamped_both_ways():
    ex = Expert()
    assert ex._commit_ticks_for(0.0) >= ex.cfg.commit_ticks_min
    assert ex._commit_ticks_for(99.0) <= ex.cfg.commit_ticks_max


def test_a_push_that_cannot_cover_the_distance_is_flagged():
    """An infeasible commit_speed/range combination looks from the outside like a
    controller that just misses, so it says so instead."""
    ex = Expert(ExpertConfig(commit_speed_mps=0.10))
    ex._commit_ticks_for(1.2)
    assert ex.commit_clamped
    ex2 = Expert()
    ex2._commit_ticks_for(0.5)
    assert not ex2.commit_clamped


def test_it_refuses_to_drive_forward_while_badly_misaligned():
    ex = Expert()
    a, _ = ex(_ring(cx=0.6, range_m=1.2), {})
    assert a[spec.ACT_FB] == 0.0
    assert abs(a[spec.ACT_YAW]) > 0, "but it does turn towards the ring"


def test_it_refuses_to_drive_forward_while_far_off_the_rings_axis():
    ex = Expert()
    a, _ = ex(_ring(cx=0.0, axis_ratio=0.2, range_m=1.2), {})
    assert a[spec.ACT_FB] == 0.0


def test_vertical_correction_has_the_right_sign():
    """cy is positive DOWN in image coordinates, so a ring above centre means climb."""
    ex = Expert()
    above, _ = ex(_ring(cy=-0.4), {})
    ex.reset()
    below, _ = ex(_ring(cy=+0.4), {})
    assert above[spec.ACT_UD] > 0
    assert below[spec.ACT_UD] < 0


def test_lateral_seeking_reverses_when_the_ring_does_not_get_rounder():
    """The ellipse gives no sign, so the sidestep direction is found by trying one."""
    ex = Expert(ExpertConfig(seek_period=2))
    dirs = []
    for _ in range(12):
        a, _ = ex(_ring(axis_ratio=0.3, range_m=1.0), {})     # never improves
        dirs.append(np.sign(a[spec.ACT_LR]))
    assert len(set(d for d in dirs if d != 0)) == 2, "it should have tried both ways"


def test_seeking_stops_once_the_ring_looks_circular():
    ex = Expert()
    a, _ = ex(_ring(axis_ratio=0.99, range_m=1.0), {})
    assert a[spec.ACT_LR] == 0.0


# --------------------------------------------------------------- staged bring-up

def test_no_forward_never_commands_forward_or_commits():
    """Stage 2: align only."""
    cfg = ExpertConfig(allow_forward=False)
    sim, ex, _ = fly(cfg=cfg, max_ticks=200)
    assert ex.phase != COMMIT
    assert not sim.outcome.crossed

    ex2 = Expert(cfg)
    for cx in (0.0, 0.3, -0.3):
        a, _ = ex2(_ring(cx=cx, range_m=0.4), {})
        assert a[spec.ACT_FB] == 0.0


def test_no_commit_approaches_but_never_pushes_through():
    """Stage 3."""
    sim, ex, _ = fly(cfg=ExpertConfig(allow_commit=False), max_ticks=300)
    assert ex.phase != COMMIT
    assert not sim.outcome.passed


# --------------------------------------------------------------- the simulator

def test_crossing_through_the_opening_counts_as_passed():
    ring = Ring(np.array([0.0, 0.0, 1.0]), 0.0)
    sim = OnboardSim(ring=ring, profile="ideal", start_pos=[-0.5, 0.0, 1.0],
                     start_yaw_deg=0.0)
    for _ in range(200):
        sim.step([0, 1, 0, 0])
        if sim.outcome.crossed:
            break
    assert sim.outcome.passed and not sim.outcome.struck


def test_crossing_at_the_rim_counts_as_struck():
    ring = Ring(np.array([0.0, 0.0, 1.0]), 0.0, opening_m=0.5, noodle_m=0.06)
    offset = 0.5 / 2 + 0.03            # inside the noodle band
    sim = OnboardSim(ring=ring, profile="ideal", start_pos=[-0.5, offset, 1.0],
                     start_yaw_deg=0.0)
    for _ in range(200):
        sim.step([0, 1, 0, 0])
        if sim.outcome.crossed:
            break
    assert sim.outcome.struck


def test_crossing_well_outside_counts_as_missed():
    ring = Ring(np.array([0.0, 0.0, 1.0]), 0.0)
    sim = OnboardSim(ring=ring, profile="ideal", start_pos=[-0.5, 0.6, 1.0],
                     start_yaw_deg=0.0)
    for _ in range(200):
        sim.step([0, 1, 0, 0])
        if sim.outcome.crossed:
            break
    assert sim.outcome.missed


def test_the_drone_is_scored_as_an_aircraft_not_a_point():
    """Regression, and the reason the measured pass rates mean anything.

    The crossing test used to compare the drone's CENTRE against the opening radius, so a
    radial error of 0.10 m through the measured 0.30 m ring scored as a clean pass even
    though a 0.14 m aircraft centred there has its props 0.03 m inside the noodle. That
    is not a cosmetic difference: it overstated the harsh-profile pass rate by 13 points,
    and by most exactly where the clearance is thinnest and the number matters.
    """
    ring = Ring(np.array([0.0, 0.0, 1.0]), 0.0, opening_m=0.30, noodle_m=0.06)
    offset = 0.10                     # inside the 0.15 m opening radius, outside the
                                      # 0.08 m a 0.14 m-wide drone actually has

    def cross(drone_width_m):
        sim = OnboardSim(ring=ring, profile="ideal", start_pos=[-0.5, offset, 1.0],
                         start_yaw_deg=0.0, drone_width_m=drone_width_m)
        for _ in range(200):
            sim.step([0, 1, 0, 0])
            if sim.outcome.crossed:
                break
        return sim.outcome

    assert cross(spec.DRONE_WIDTH_M).struck, (
        "a 14 cm drone 10 cm off the axis of a 30 cm ring strikes the noodle")
    assert cross(0.0).passed, (
        "a zero-width drone must still reproduce the old point-mass scoring, so the "
        "pure-geometry tests keep meaning what they say")


def test_every_alignment_gate_fits_inside_the_measured_clearance():
    """The gates are what keep the aircraft off the noodle, so none may exceed the budget.

    spec.RING_CLEARANCE_M is the whole distance from the drone's centre to the noodle
    (0.08 m: a 0.15 m opening radius less a 0.07 m half-width). A commit gate wider than
    that -- which is what 0.10 m was against the measured ring -- commits to a blind push
    the aircraft cannot complete. Half the budget is spent on the gate, leaving half for
    drift during the push.
    """
    c = ExpertConfig()
    budget = spec.RING_CLEARANCE_M
    assert budget > 0, "the drone does not fit through the ring at all"

    # The commit gates are the ones that actually guarantee clearance: they are the last
    # check before the drone is blind. Half the budget each, leaving half for drift.
    for name in ("commit_lateral_m", "commit_vertical_m"):
        assert getattr(c, name) <= budget / 2.0 + 1e-9, (
            f"{name} = {getattr(c, name)} exceeds half the {budget:.3f} m clearance")

    # The align pair only labels the phase, so it is held to the weaker requirement of
    # fitting inside the clearance at all -- but it must be no TIGHTER than the commit
    # gate, or the APPROACH label would be reached only after the drone is already
    # tight enough to commit and the phase ladder would skip from ALIGN to COMMIT.
    for name in ("align_lateral_m", "align_vertical_m"):
        assert getattr(c, name) <= budget + 1e-9
    assert c.align_lateral_m >= c.commit_lateral_m
    assert c.align_vertical_m >= c.commit_vertical_m

    # Forward motion is gated more loosely still, or the drone could never close the
    # distance from which it would become aligned in the first place.
    assert c.forward_misalign_m >= c.align_lateral_m


def test_the_commit_range_keeps_the_whole_ring_in_frame():
    """The commit decision is made from the last detection, so the ring has to be whole.

    At the measured 0.42 m outer diameter and a 70 deg horizontal field of view, a 4:3
    frame's vertical dimension binds and the ring becomes tangent to it at 0.400 m.
    Committing at or below that range means deciding from a measurement that is already
    clipping. Measured, 0.42 m beats both 0.40 and 0.45.
    """
    tangent_m = spec.focal_px(4.0) * spec.RING_OUTER_M / 3.0     # any 4:3 frame
    assert tangent_m == pytest.approx(0.400, abs=0.005)
    assert ExpertConfig().commit_range_m > tangent_m, (
        "commit_range_m sits at or inside the range where the ring stops being whole")


def test_telemetry_matches_djitellopys_int_and_float_field_split():
    """Regression: survey.health tests isinstance(h, int), so a float height made every
    airborne tick abort. bool is excluded because bool is a subclass of int."""
    s = OnboardSim(seed=1).state()
    for key in ("h", "tof", "bat", "temph", "templ", "pitch", "roll", "yaw",
                "vgx", "vgy", "vgz"):
        assert isinstance(s[key], int) and not isinstance(s[key], bool), key
    for key in ("baro", "agx", "agy", "agz"):
        assert isinstance(s[key], float), key


def test_the_analytic_and_rendered_observations_agree_on_clean_geometry():
    """The analytic path is a MODEL of the detector, used for fast RL. If the two drift
    apart, a policy trained analytically would not transfer to the rendered path, let
    alone to a real camera."""
    ring = Ring(np.array([0.3, 0.0, 1.0]), 0.0)
    kw = dict(ring=ring, profile="ideal", start_pos=[-0.7, 0.0, 1.0], start_yaw_deg=0.0)
    a = OnboardSim(render=False, **kw).observe()
    r = OnboardSim(render=True, **kw).observe()
    assert a.found and r.found
    assert r.cx == pytest.approx(a.cx, abs=0.08)
    assert r.cy == pytest.approx(a.cy, abs=0.08)
    assert r.range_m == pytest.approx(a.range_m, rel=0.2)


def test_the_ring_is_not_seen_when_it_is_behind_the_drone():
    sim = OnboardSim(ring=Ring(np.array([0.0, 0.0, 1.0]), 0.0), profile="ideal",
                     start_pos=[-0.8, 0.0, 1.0], start_yaw_deg=180.0)
    assert not sim.observe().found


def test_out_of_cage_is_detected():
    sim = OnboardSim(profile="ideal", seed=0)
    assert not sim.out_of_cage()
    sim.pos = np.array([1.5, 0.0, 1.0])
    assert sim.out_of_cage()


@pytest.mark.parametrize("name", list(PROFILES))
def test_every_profile_runs(name):
    sim = OnboardSim(profile=name, seed=0)
    for _ in range(30):
        sim.step([0.1, 0.3, 0.0, 0.05])
    assert np.all(np.isfinite(sim.pos))


# --------------------------------------------------------------- the gym env

def test_env_obeys_the_gymnasium_contract():
    from onboard.gym_env import OnboardHoopEnv
    env = OnboardHoopEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (spec.OBS_DIM,)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    env.close()


def test_env_rewards_passing_more_than_missing():
    from onboard.gym_env import R_MISSED, R_PASSED, R_STRUCK
    assert R_PASSED > 0 > R_STRUCK
    assert R_PASSED > R_MISSED


def test_env_truncates_rather_than_running_for_ever():
    from onboard.gym_env import OnboardHoopEnv
    env = OnboardHoopEnv(seed=0, max_episode_sec=1.0)
    env.reset(seed=0)
    for _ in range(int(1.0 * spec.CTRL_HZ) + 5):
        _o, _r, term, trunc, _i = env.step(np.zeros(spec.ACT_DIM))
        if term or trunc:
            break
    assert term or trunc
    env.close()


def test_rollout_scores_the_expert_through_the_env():
    """Ties the env's reward and tally back to the controller a policy has to beat."""
    from onboard.expert import ExpertPolicy
    from onboard.gym_env import rollout

    res = rollout(ExpertPolicy(), episodes=8, seed=7)
    assert res["episodes"] == 8
    assert set(res) >= {"pass_rate", "passed", "struck", "missed", "mean_reward"}
    assert res["pass_rate"] >= 0.5, (
        "the expert drives the env well; a low rate here points at ring_from_obs losing "
        "information the expert needs, not at the controller")
    assert res["mean_reward"] > 0


def test_rollout_resets_a_stateful_controller_between_episodes():
    """Regression: without the reset hook the Expert stays DONE after episode one and
    every later episode scores zero."""
    from onboard.expert import ExpertPolicy
    from onboard.gym_env import rollout
    p = ExpertPolicy()
    first = rollout(p, episodes=1, seed=7)["passed"]
    many = rollout(p, episodes=4, seed=7)["passed"]
    assert first == 1
    assert many >= 2, "later episodes must not be dead"


def test_ring_from_obs_round_trips_the_fields_the_expert_uses():
    from onboard.obs import ObsBuilder, ring_from_obs
    original = _ring(cx=0.23, cy=-0.17, axis_ratio=0.71, range_m=0.9)
    state = {"h": 80, "vgx": 0, "vgy": 0, "vgz": 0, "pitch": 0, "roll": 0, "yaw": 0}
    back = ring_from_obs(ObsBuilder().build(original, state, 1.0))
    assert back.found
    assert back.cx == pytest.approx(original.cx, abs=1e-3)
    assert back.cy == pytest.approx(original.cy, abs=1e-3)
    assert back.axis_ratio == pytest.approx(original.axis_ratio, abs=1e-3)
    assert back.range_m == pytest.approx(original.range_m, abs=1e-3)


def test_the_search_sweeps_both_sides_of_the_start_heading():
    """Equal-length legs from the start heading only ever covered one side of it, so a
    ring to the other side was never found."""
    from onboard.expert import Expert
    from onboard.ring_detect import RingObs
    ex = Expert()
    heading, lo, hi = 0.0, 0.0, 0.0
    for _ in range(ex.cfg.search_max_ticks):
        a, _info = ex(RingObs())
        heading += a[spec.ACT_YAW]
        lo, hi = min(lo, heading), max(hi, heading)
    assert lo < -0.5 * hi and hi > 0, (lo, hi)
