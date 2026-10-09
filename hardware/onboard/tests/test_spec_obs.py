"""
The task interface: spec.py's layout invariants and obs.py's construction.

These are cheap tests guarding an expensive mistake. Every other module, the recorded
demonstrations and every trained policy all agree on this layout by importing spec; if a
field width changed without OBS_DIM following, old demos and new policies would silently
disagree about which number means what.
"""
import numpy as np
import pytest

from onboard import spec
from onboard.obs import MAX_YAW_RATE, ObsBuilder, wrap_deg
from onboard.ring_detect import RingObs


# --------------------------------------------------------------- spec invariants

def test_obs_dim_equals_the_sum_of_the_field_widths():
    assert spec.OBS_DIM == sum(w for _, w in spec.OBS_FIELDS)


def test_slices_tile_the_observation_exactly_without_gaps_or_overlap():
    covered = np.zeros(spec.OBS_DIM, dtype=int)
    for name, _w in spec.OBS_FIELDS:
        covered[spec.OBS_SLICES[name]] += 1
    assert covered.tolist() == [1] * spec.OBS_DIM


def test_action_buffer_occupies_the_tail():
    s = spec.OBS_SLICES["actions"]
    assert s.stop == spec.OBS_DIM
    assert s.stop - s.start == spec.ACT_BUFFER * spec.ACT_DIM


# --------------------------------------------------------------- action mapping

def test_action_to_rc_scales_and_rounds():
    assert spec.action_to_rc([1.0, -1.0, 0.0, 0.5], rc_max=20) == [20, -20, 0, 10]


def test_action_to_rc_clamps_beyond_unit_range():
    """The clamp lives in exactly one place, so no controller can exceed rc_max."""
    assert spec.action_to_rc([5.0, -5.0, 2.0, -2.0], rc_max=30) == [30, -30, 30, -30]


@pytest.mark.parametrize("bad", [
    [np.nan, 0, 0, 0], [np.inf, 0, 0, 0], [-np.inf, 0, 0, 0],
])
def test_action_to_rc_turns_non_finite_into_zero_not_a_crash(bad):
    """A policy that returns NaN must produce a hover, not an exception in the air."""
    assert spec.action_to_rc(bad)[0] == 0


def test_action_to_rc_pads_a_short_action():
    assert len(spec.action_to_rc([0.5, 0.5])) <= spec.ACT_DIM


# --------------------------------------------------------------- range geometry

def test_focal_length_grows_as_the_field_of_view_narrows():
    assert spec.focal_px(480, 50) > spec.focal_px(480, 90)


def test_outer_diameter_is_the_opening_plus_two_noodles():
    assert spec.RING_OUTER_M == pytest.approx(
        spec.RING_OPENING_M + 2 * spec.RING_NOODLE_M)
    assert spec.RING_DIAMETER_M == spec.RING_OUTER_M, (
        "ranging must use the outer edge, which is what the camera sees")


# --------------------------------------------------------------- the observation

def _ring(**kw):
    base = dict(found=True, fresh=True, cx=0.2, cy=-0.1, major_px=120.0, minor_px=100.0,
                axis_ratio=0.83, scale=0.25, range_m=1.1, age=0)
    base.update(kw)
    return RingObs(**base)


STATE = {"h": 80, "vgx": 10, "vgy": -5, "vgz": 2, "pitch": 3, "roll": -2, "yaw": 15,
         "bat": 70, "temph": 60, "tof": 80}


def test_observation_has_the_declared_shape_and_dtype():
    o = ObsBuilder().build(_ring(), STATE, 1.0)
    assert o.shape == (spec.OBS_DIM,)
    assert o.dtype == np.float32


def test_fields_land_in_their_declared_slices():
    o = ObsBuilder().build(_ring(cx=0.2, cy=-0.1, axis_ratio=0.83), STATE, 1.0)
    s = spec.OBS_SLICES
    assert o[s["found"]][0] == 1.0
    assert o[s["centre"]].tolist() == pytest.approx([0.2, -0.1], abs=1e-6)
    assert o[s["axis_ratio"]][0] == pytest.approx(0.83, abs=1e-6)
    assert o[s["height"]][0] == pytest.approx(80 / spec.HEIGHT_NORM)


def test_an_unmeasurable_range_reads_as_far_not_as_infinity():
    """inf in an observation poisons every weight it reaches."""
    o = ObsBuilder().build(_ring(found=False, range_m=float("inf")), STATE, 1.0)
    assert np.all(np.isfinite(o))
    assert o[spec.OBS_SLICES["range"]][0] == 1.0


def test_action_buffer_is_oldest_first_and_starts_at_zero():
    b = ObsBuilder()
    o = b.build(_ring(), STATE, 1.0)
    assert np.all(o[spec.OBS_SLICES["actions"]] == 0), "zeros at takeoff"

    for i in range(spec.ACT_BUFFER):
        b.push_action([0.1 * (i + 1)] * spec.ACT_DIM)
    acts = b.build(_ring(), STATE, 2.0)[spec.OBS_SLICES["actions"]]
    first, last = acts[: spec.ACT_DIM], acts[-spec.ACT_DIM:]
    assert first[0] == pytest.approx(0.1, abs=1e-6), "oldest first"
    assert last[0] == pytest.approx(0.1 * spec.ACT_BUFFER, abs=1e-6)


def test_buffer_evicts_the_oldest_once_full():
    b = ObsBuilder()
    pushed = [i / 10.0 for i in range(spec.ACT_BUFFER + 2)]   # all within [-1, 1]
    for v in pushed:
        b.push_action([v] * spec.ACT_DIM)
    acts = b.build(_ring(), STATE, 1.0)[spec.OBS_SLICES["actions"]]
    kept = pushed[-spec.ACT_BUFFER:]
    assert acts[0] == pytest.approx(kept[0], abs=1e-6), "the two earliest were evicted"
    assert acts[-spec.ACT_DIM] == pytest.approx(kept[-1], abs=1e-6)


def test_reset_clears_the_buffer():
    b = ObsBuilder()
    b.push_action([1.0] * spec.ACT_DIM)
    b.reset()
    assert np.all(b.build(_ring(), STATE, 1.0)[spec.OBS_SLICES["actions"]] == 0)


def test_pushed_actions_are_clamped_and_sanitised():
    b = ObsBuilder()
    b.push_action([5.0, np.nan, -9.0, 0.5])
    acts = b.build(_ring(), STATE, 1.0)[spec.OBS_SLICES["actions"]]
    assert acts[-spec.ACT_DIM:].tolist() == pytest.approx([1.0, 0.0, -1.0, 0.5], abs=1e-6)


# --------------------------------------------------------------- yaw rate guards

def test_wrap_deg_takes_the_short_way_round():
    assert wrap_deg(350 - 10) == pytest.approx(-20)
    assert wrap_deg(-350 + 10) == pytest.approx(20)


def test_yaw_rate_ignores_an_implausibly_short_interval():
    """hardware/README 3.7: two samples a float epsilon apart once divided a yaw
    difference by ~0, the rate reached ~1e10 and saturated the controller."""
    b = ObsBuilder()
    b.yaw_rate({"yaw": 0}, 100.0)
    rate = b.yaw_rate({"yaw": 40}, 100.0 + 1e-12)
    assert rate == 0.0, "an interval below MIN_DT must not be divided by"


def test_yaw_rate_is_clamped():
    b = ObsBuilder()
    b.yaw_rate({"yaw": 0}, 0.0)
    assert abs(b.yaw_rate({"yaw": 179}, 0.03)) <= MAX_YAW_RATE


def test_yaw_rate_sign_and_magnitude_are_right_for_a_normal_interval():
    b = ObsBuilder()
    b.yaw_rate({"yaw": 0}, 0.0)
    assert b.yaw_rate({"yaw": 10}, 0.1) == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize("value", [None, True, "80", float("nan"), [1, 2]])
def test_malformed_telemetry_values_fall_back_to_the_default(value):
    """djitellopy can hand back a str for a malformed packet, and bool is an int."""
    o = ObsBuilder().build(_ring(), {**STATE, "h": value}, 1.0)
    assert np.all(np.isfinite(o))
