"""Image-space target candidates; these observations do not establish flight clearance."""

import math

import cv2
import numpy as np


def blue_ring_candidates(bgr):
    """Find complete blue elliptical outlines with a sizeable opening.

    A candidate is not proof of a vertical hoop, its physical size, or a safe path.
    HSV thresholds need validation against the actual cage lighting.
    """
    height, width = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 80, 40]), np.array([135, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    candidates = []
    for index, outer in enumerate(contours):
        child = int(hierarchy[0, index, 2])
        if hierarchy[0, index, 3] != -1 or child < 0 or len(outer) < 5:
            continue
        area = cv2.contourArea(outer)
        x, y, box_w, box_h = cv2.boundingRect(outer)
        if area < 600 or min(box_w, box_h) < 25:
            continue
        if x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1:
            continue
        opening = max(
            (cv2.contourArea(c) for i, c in enumerate(contours) if hierarchy[0, i, 3] == index),
            default=0,
        )
        if not 0.25 <= opening / area <= 0.95:
            continue
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(outer)
        if min(axis_a, axis_b) / max(axis_a, axis_b) < 0.2:
            continue
        ellipse_area = math.pi * axis_a * axis_b / 4
        if not 0.8 <= area / ellipse_area <= 1.15:
            continue
        radians = math.radians(angle)
        delta = outer[:, 0, :].astype(float) - [cx, cy]
        u = delta[:, 0] * math.cos(radians) + delta[:, 1] * math.sin(radians)
        v = -delta[:, 0] * math.sin(radians) + delta[:, 1] * math.cos(radians)
        residual = float(
            np.mean(np.abs(np.sqrt((2 * u / axis_a) ** 2 + (2 * v / axis_b) ** 2) - 1))
        )
        if residual > 0.12:
            continue
        candidates.append(
            {
                "center_px": [round(cx, 2), round(cy, 2)],
                "center_normalized": [round(cx / width, 4), round(cy / height, 4)],
                "axes_px": [round(axis_a, 2), round(axis_b, 2)],
                "angle_deg": round(angle, 2),
                "opening_area_ratio": round(opening / area, 3),
                "ellipse_residual": round(residual, 4),
                "bbox_px": [x, y, box_w, box_h],
                "outer_area_px": round(area, 1),
            }
        )
    return sorted(candidates, key=lambda c: c["outer_area_px"], reverse=True)


def annotate(bgr, candidates, state):
    result = bgr.copy()
    for index, candidate in enumerate(candidates, 1):
        center = tuple(round(v) for v in candidate["center_px"])
        axes = tuple(round(v / 2) for v in candidate["axes_px"])
        cv2.ellipse(result, center, axes, candidate["angle_deg"], 0, 360, (0, 255, 255), 2)
        cv2.putText(
            result,
            f"Blue ring candidate {index}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
    text = (
        f"Candidates: {len(candidates)} | pad: {state.get('mid', 'unknown')} | "
        f"battery: {state.get('bat')}% | board: {state.get('temph')} C"
    )
    cv2.putText(result, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return result
