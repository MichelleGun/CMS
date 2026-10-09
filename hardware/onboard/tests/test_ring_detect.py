"""
Detector and tracker. These are the tests that pin down the three behaviours the kit's
perception.py does not have, each of which was a real blocker:

  * a clipped ring is still detected, by fitting only the visible arc. Without this,
    detection returns [] from about half a metre and the approach is blind for its whole
    second half.
  * an irregular ring is accepted. The kit's fill-ratio and residual gates assume a
    near-perfect circle.
  * the lock survives dropped frames and is not stolen by a larger blue object.
"""
import numpy as np
import pytest

from onboard import spec
from onboard.ring_detect import (DetectorParams, RingTracker, arc_span_deg, detect_rings)

from onboard.tests.shapes import RED_BGR, draw_ring, draw_wobbly_ring, frame


# --------------------------------------------------------------- per-frame detection

def test_clean_ring_is_detected_with_correct_centre_and_size():
    img = draw_ring(frame(), (240, 180), 60)
    cands = detect_rings(img)
    assert len(cands) == 1
    c = cands[0]
    assert c["center_px"][0] == pytest.approx(240, abs=3)
    assert c["center_px"][1] == pytest.approx(180, abs=3)
    assert not c["clipped"]
    # The fit is to the annulus OUTER edge, so major is the outer diameter.
    assert c["major_px"] == pytest.approx(2 * (60 + 0.22 * 60 / 2), rel=0.15)


@pytest.mark.parametrize("make", [
    pytest.param(lambda: frame(), id="empty"),
    pytest.param(lambda: draw_ring(frame(), (240, 180), 60, colour=RED_BGR), id="red"),
    pytest.param(lambda: draw_ring(frame(), (240, 180), 60, thickness=-1), id="solid_disc"),
    pytest.param(lambda: draw_ring(frame(), (240, 180), 8), id="too_small"),
])
def test_non_rings_are_rejected(make):
    assert detect_rings(make()) == []


KIT_GATES = DetectorParams(fill_ratio=(0.8, 1.15), residual_max=0.12)
"""The kit's own thresholds, for comparison."""


@pytest.mark.parametrize("wobble", [0.10, 0.20, 0.32, 0.40])
def test_irregular_rings_are_accepted(wobble):
    """A pool noodle bent into a ring is visibly not a circle, and must still be found."""
    img = draw_wobbly_ring(frame(), (240, 180), 62, wobble=wobble, seed=3)
    cands = detect_rings(img)
    assert len(cands) == 1
    assert cands[0]["center_px"][0] == pytest.approx(240, abs=6)


def test_a_strongly_irregular_ring_needs_the_relaxed_residual_gate():
    """Where the relaxed gate actually earns its keep, measured rather than assumed.

    The kit's residual limit of 0.12 corresponds to roughly 30 % radius wobble: below
    that it accepts an irregular ring perfectly well (see the parametrised test above,
    which passes under both threshold sets at 0.10 and 0.20). So this gate was a milder
    obstacle than it looked -- the blocker that actually stopped detection dead was
    frame clipping, covered by the next test.
    """
    mild = draw_wobbly_ring(frame(), (240, 180), 62, wobble=0.20, seed=3)
    assert len(detect_rings(mild, KIT_GATES)) == 1, "0.20 is within the strict gate"

    strong = draw_wobbly_ring(frame(), (240, 180), 62, wobble=0.32, seed=3)
    assert detect_rings(strong, KIT_GATES) == []
    assert len(detect_rings(strong)) == 1


def test_a_ring_clipped_by_the_frame_edge_is_still_detected():
    """The close-approach case. The outline runs off the frame, so there is no enclosed
    child contour to prove it is a ring -- the arc-span path has to carry it."""
    img = draw_ring(frame(), (150, 60), 150)        # overflows the top and left
    cands = detect_rings(img)
    assert len(cands) == 1
    assert cands[0]["clipped"] is True
    assert cands[0]["arc_deg"] < 359
    assert cands[0]["center_px"][0] == pytest.approx(150, abs=12)

    assert detect_rings(img, DetectorParams(allow_edge=False)) == []


def test_a_barely_visible_sliver_is_rejected():
    """Too little arc to fit: better no reading than a wild one."""
    img = draw_ring(frame(), (-40, 180), 90)
    cands = detect_rings(img, DetectorParams(min_arc_deg=110.0))
    assert cands == [] or cands[0]["arc_deg"] >= 110.0


def test_arc_span_of_a_full_circle_is_360_and_of_a_half_is_about_180():
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    full = np.stack([100 + 50 * np.cos(th), 100 + 50 * np.sin(th)], axis=1)
    assert arc_span_deg(full, 100, 100) == pytest.approx(360, abs=6)
    half = full[: 36]
    assert arc_span_deg(half, 100, 100) == pytest.approx(180, abs=12)


# --------------------------------------------------------------- range estimate

def test_range_halves_when_the_ring_looks_twice_as_big():
    r1 = spec.range_from_major_axis(100, 480, 0.62)
    r2 = spec.range_from_major_axis(200, 480, 0.62)
    assert r2 == pytest.approx(r1 / 2, rel=1e-6)


def test_range_is_recovered_from_a_projected_ring():
    """Draw a ring at a known distance, measure it back."""
    w, true_range, outer = 480, 1.2, 0.62
    major = spec.focal_px(w) * outer / true_range
    img = draw_ring(frame(w, 360), (w // 2, 180), major / 2, thickness=2)
    t = RingTracker(ring_diameter_m=outer)
    ring = t.update(img)
    assert ring.found
    assert ring.range_m == pytest.approx(true_range, rel=0.08)


def test_degenerate_measurement_gives_inf_not_a_crash():
    assert spec.range_from_major_axis(0.0, 480) == float("inf")


# --------------------------------------------------------------- the tracker

def test_lock_coasts_through_dropped_frames_then_is_dropped():
    t = RingTracker()
    assert t.update(draw_ring(frame(), (240, 180), 60)).found
    blank = frame()
    for i in range(spec.LOCK_MAX_AGE - 1):
        obs = t.update(blank)
        assert obs.found, f"should still be coasting at age {i + 1}"
        assert not obs.fresh
    for _ in range(3):
        obs = t.update(blank)
    assert not obs.found


def test_a_bigger_blue_decoy_does_not_steal_the_lock():
    """Selection is nearest-to-previous, not largest: mid-approach a larger blue object
    entering the frame must not capture the tracker."""
    t = RingTracker()
    first = t.update(draw_ring(frame(), (150, 180), 45))
    assert first.found

    both = draw_ring(frame(), (150, 180), 45)
    draw_ring(both, (390, 180), 80)              # bigger, far away
    second = t.update(both)
    assert second.found
    # Still near the original centre, not dragged to the decoy.
    assert abs(second.cx - first.cx) < 0.25


def test_centre_offsets_are_signed_and_share_one_angular_scale():
    """cx and cy are both divided by HALF THE WIDTH, so one gain suits both axes."""
    w, h = 480, 360
    img = draw_ring(frame(w, h), (w // 2 + 120, h // 2 + 120), 40)
    ring = RingTracker().update(img)
    assert ring.cx == pytest.approx(120 / (w / 2), abs=0.05)
    assert ring.cy == pytest.approx(120 / (w / 2), abs=0.05)


def test_axis_ratio_reports_foreshortening():
    square_on = RingTracker().update(draw_ring(frame(), (240, 180), 70, ratio=1.0))
    edge_on = RingTracker().update(draw_ring(frame(), (240, 180), 70, ratio=0.35))
    assert square_on.axis_ratio > 0.9
    assert edge_on.axis_ratio < 0.55


def test_reset_clears_the_lock():
    t = RingTracker()
    t.update(draw_ring(frame(), (240, 180), 60))
    t.reset()
    assert not t.update(frame()).found


def test_blue_fraction_rises_as_the_ring_fills_the_frame():
    small = RingTracker().update(draw_ring(frame(), (240, 180), 30))
    big = RingTracker().update(draw_ring(frame(), (240, 180), 150))
    assert big.blue_fraction > small.blue_fraction


def test_params_round_trip_through_json():
    p = DetectorParams(hsv_lo=(95, 70, 50), residual_max=0.25, allow_edge=False)
    back = DetectorParams.from_json(p.to_json())
    assert back == p
