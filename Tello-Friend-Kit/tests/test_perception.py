import cv2
import numpy as np
import pytest

from tello_lab.perception import blue_ring_candidates


@pytest.mark.parametrize("axes,angle", [((100, 100), 0), ((100, 45), 30)])
def test_detects_blue_open_ring(axes, angle):
    image = np.full((480, 640, 3), 120, np.uint8)
    cv2.ellipse(image, (320, 240), axes, angle, 0, 360, (255, 0, 0), 12)
    candidates = blue_ring_candidates(image)
    assert len(candidates) == 1
    assert candidates[0]["center_px"] == pytest.approx([320, 240], abs=2)


@pytest.mark.parametrize("shape", ["solid", "red", "cut", "rectangle", "noise"])
def test_rejects_non_ring_and_incomplete_targets(shape):
    image = np.full((480, 640, 3), 120, np.uint8)
    if shape == "solid":
        cv2.circle(image, (320, 240), 100, (255, 0, 0), -1)
    elif shape == "red":
        cv2.circle(image, (320, 240), 100, (0, 0, 255), 12)
    elif shape == "cut":
        cv2.circle(image, (20, 240), 100, (255, 0, 0), 12)
    elif shape == "rectangle":
        cv2.rectangle(image, (220, 140), (420, 340), (255, 0, 0), 12)
    else:
        cv2.circle(image, (320, 240), 10, (255, 0, 0), 2)
    assert blue_ring_candidates(image) == []


def test_preserves_multiple_candidates_for_review():
    image = np.full((480, 640, 3), 120, np.uint8)
    for center in ((150, 240), (470, 240)):
        cv2.circle(image, center, 80, (255, 0, 0), 10)
    assert len(blue_ring_candidates(image)) == 2
