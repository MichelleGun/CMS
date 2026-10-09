"""
Blue ring detection and tracking from the Tello's front camera.

Adapted from Tello-Friend-Kit/src/tello_lab/perception.py (HSV threshold -> contour
hierarchy -> ellipse fit, with the ring's opening found as a child contour via
RETR_TREE). That function is correct for what it was written for, but three of its gates
make it unusable here, so every threshold became a parameter:

  * perception.py:31 rejects any contour touching the frame edge. During a fly-through
    the ring necessarily grows until it clips the border, so that gate guarantees
    detection returns [] exactly in the last metre. Here it becomes `allow_edge`, and a
    clipped detection is reported with clipped=True instead of being discarded.
  * perception.py:43 requires area/ellipse_area in [0.8, 1.15] and perception.py:52
    requires an ellipse residual under 0.12. Both assume a near-perfect circle, and a
    pool noodle bent into a ring is visibly irregular. Measured (see
    tests/test_ring_detect.py): the residual gate starts rejecting at about 30 % radius
    wobble, so it is a smaller obstacle than it looks -- a mildly bent ring passes it.
    It is parameterised for margin, not because it was the blocker.
  * The HSV window is a parameter because it has never been validated against real
    footage in any lighting. tune_detector.py sets it from the actual cage.

Added on top of the kit's per-frame geometry, which a control loop needs and it lacks:
  * temporal tracking: nearest-to-previous selection instead of "largest wins", so a
    second blue object cannot steal the lock mid-approach;
  * lock persistence: a dropped frame coasts on the last estimate for LOCK_MAX_AGE
    ticks rather than reporting "no ring";
  * EMA smoothing, so a one-frame wobble does not become a control input;
  * a range estimate (spec.range_from_major_axis);
  * blue_fraction, a frame-level fallback: when the drone is at the ring the annulus
    leaves the frame entirely and no candidate can be found, but the fraction of blue
    pixels spikes. That is the only signal left at that distance.
"""
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import cv2
import numpy as np

from . import spec


@dataclass
class DetectorParams:
    """Every threshold. Defaults are deliberately looser than the kit's; tune_detector.py
    writes measured values to detector.json and fly_hoop.py loads them."""
    hsv_lo: tuple = (90, 60, 40)        # OpenCV hue 0-179, so 90-135 is blue..violet
    hsv_hi: tuple = (135, 255, 255)
    min_area_px: float = 400.0          # kit: 600
    min_side_px: int = 20               # kit: 25
    opening_ratio: tuple = (0.15, 0.97)  # kit: (0.25, 0.95)
    fill_ratio: tuple = (0.55, 1.45)    # kit: (0.8, 1.15) -- a bent noodle is irregular
    residual_max: float = 0.30          # kit: 0.12
    min_axis_ratio: float = 0.15        # kit: 0.2
    allow_edge: bool = True             # kit: False (hardcoded)
    close_kernel: int = 5               # kit: 3
    border_tol: int = 2                 # px; contour points this close to the frame
                                        # edge are artefacts of clipping, not the ring
    min_arc_deg: float = 110.0          # a clipped ring must still show this much arc
    min_major_px: float = 24.0           # smallest ring worth fitting

    def to_json(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, d: dict) -> "DetectorParams":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        for key in ("hsv_lo", "hsv_hi", "opening_ratio", "fill_ratio"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)


@dataclass
class RingObs:
    """What the control loop sees. `found` means "usable this tick", which includes
    coasting on a recent lock; `fresh` means a real detection happened this tick."""
    found: bool = False
    fresh: bool = False
    cx: float = 0.0              # centre offset from image centre, in half-frames
    cy: float = 0.0
    major_px: float = 0.0
    minor_px: float = 0.0
    axis_ratio: float = 0.0
    scale: float = 0.0           # major_px / frame_width
    range_m: float = float("inf")
    age: int = spec.LOCK_MAX_AGE
    clipped: bool = False
    blue_fraction: float = 0.0
    n_candidates: int = 0
    raw: Optional[dict] = field(default=None, repr=False)


def blue_mask(bgr, params: DetectorParams):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(params.hsv_lo, dtype=np.uint8),
                       np.array(params.hsv_hi, dtype=np.uint8))
    k = max(1, int(params.close_kernel))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))


def _on_border(pts: np.ndarray, width: int, height: int, tol: int) -> np.ndarray:
    """Contour points lying on the frame edge. These are an artefact of the ring being
    cut off by the field of view, not part of its outline, so a fit must exclude them."""
    x, y = pts[:, 0], pts[:, 1]
    return (x <= tol) | (y <= tol) | (x >= width - 1 - tol) | (y >= height - 1 - tol)


def arc_span_deg(pts: np.ndarray, cx: float, cy: float) -> float:
    """How much of a full turn the points cover, as seen from (cx, cy).

    Found as 360 minus the largest angular gap. A fitEllipse on a short arc is
    numerically unreliable, so this is what decides whether a clipped ring is still
    measurable.
    """
    if len(pts) < 3:
        return 0.0
    ang = np.sort(np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
    return float(360.0 - gaps.max())


def detect_rings(bgr, params: Optional[DetectorParams] = None) -> List[dict]:
    """Ring candidates in one frame, largest first.

    Keys match the kit's blue_ring_candidates (center_px, center_normalized, axes_px,
    angle_deg, opening_area_ratio, ellipse_residual, bbox_px, outer_area_px) so the kit's
    annotate() can draw these directly, plus `clipped`, `major_px`, `minor_px` and
    `arc_deg`. Note axes_px are full axis LENGTHS (diameters) from fitEllipse, not radii
    -- the same trap as in the kit, whose annotate() halves them before drawing.

    Two validation paths, because a clipped ring cannot be checked the same way:
      * fully visible -- every one of the kit's gates applies (enclosed hole, opening
        ratio, fill ratio, residual). The hole is what proves it is a ring and not a disc.
      * clipped -- the outer boundary runs off the frame, so there may be no enclosed
        child contour at all and both the area and the hole are only partly visible.
        Those three gates are therefore skipped and replaced by an arc-span requirement,
        with the ellipse residual (computed over the visible arc only) doing the work of
        proving it is still a ring. This path is what keeps a lock during the approach;
        without it detection stops dead at about half a metre.
    """
    params = params or DetectorParams()
    height, width = bgr.shape[:2]
    mask = blue_mask(bgr, params)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    out = []
    for index, outer in enumerate(contours):
        if hierarchy[0, index, 3] != -1:          # outermost contours only
            continue
        child = int(hierarchy[0, index, 2])
        pts_all = outer[:, 0, :].astype(np.float64)
        x, y, box_w, box_h = cv2.boundingRect(outer)
        clipped = bool(x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1)
        if clipped and not params.allow_edge:
            continue
        if not clipped and child < 0:
            continue                              # a disc, not a ring

        pts = pts_all[~_on_border(pts_all, width, height, params.border_tol)] \
            if clipped else pts_all
        if len(pts) < 5:
            continue

        area = cv2.contourArea(outer)
        if min(box_w, box_h) < params.min_side_px:
            continue
        if not clipped and area < params.min_area_px:
            continue

        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(
            pts.astype(np.float32).reshape(-1, 1, 2))
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
        if major < params.min_major_px or minor / major < params.min_axis_ratio:
            continue
        ellipse_area = math.pi * axis_a * axis_b / 4.0
        if ellipse_area <= 1e-6:
            continue

        arc = arc_span_deg(pts, cx, cy)
        ratio = 0.0
        fill = 0.0
        if clipped:
            if arc < params.min_arc_deg:
                continue
        else:
            opening = max((cv2.contourArea(c) for i, c in enumerate(contours)
                           if hierarchy[0, i, 3] == index), default=0.0)
            ratio = opening / area if area > 0 else 0.0
            if not params.opening_ratio[0] <= ratio <= params.opening_ratio[1]:
                continue
            fill = area / ellipse_area
            if not params.fill_ratio[0] <= fill <= params.fill_ratio[1]:
                continue

        # Mean normalised radial deviation in the fitted ellipse's own frame: how far
        # the outline strays from the ellipse. Over the visible arc only when clipped.
        radians = math.radians(angle)
        delta = pts - [cx, cy]
        u = delta[:, 0] * math.cos(radians) + delta[:, 1] * math.sin(radians)
        v = -delta[:, 0] * math.sin(radians) + delta[:, 1] * math.cos(radians)
        residual = float(np.mean(np.abs(
            np.sqrt((2 * u / axis_a) ** 2 + (2 * v / axis_b) ** 2) - 1)))
        if residual > params.residual_max:
            continue

        out.append({
            "center_px": [round(cx, 2), round(cy, 2)],
            "center_normalized": [round(cx / width, 4), round(cy / height, 4)],
            "axes_px": [round(axis_a, 2), round(axis_b, 2)],
            "major_px": round(major, 2),
            "minor_px": round(minor, 2),
            "angle_deg": round(angle, 2),
            "opening_area_ratio": round(ratio, 3),
            "ellipse_residual": round(residual, 4),
            "fill_ratio": round(fill, 3),
            "arc_deg": round(arc, 1),
            "bbox_px": [int(x), int(y), int(box_w), int(box_h)],
            "outer_area_px": round(area, 1),
            "clipped": clipped,
        })
    # Sort by fitted ellipse size, not contour area: a clipped ring's contour area is
    # only the visible part, so area would rank a near ring below a far one.
    return sorted(out, key=lambda c: c["major_px"], reverse=True)


class RingTracker:
    """Per-frame detections -> a smoothed, persistent RingObs.

    Holding a lock across dropped frames is what lets the expert keep flying through the
    gaps the detector leaves, instead of aborting on a single bad frame.
    """

    def __init__(self, params: Optional[DetectorParams] = None,
                 ring_diameter_m: float = spec.RING_DIAMETER_M,
                 hfov_deg: float = spec.CAMERA_HFOV_DEG,
                 lock_max_age: int = spec.LOCK_MAX_AGE,
                 ema: float = 0.5,
                 gate_frac: float = 0.35):
        self.params = params or DetectorParams()
        self.ring_diameter_m = float(ring_diameter_m)
        self.hfov_deg = float(hfov_deg)
        self.lock_max_age = int(lock_max_age)
        self.ema = float(ema)
        self.gate_frac = float(gate_frac)      # reject a jump farther than this x width
        self.reset()

    def reset(self):
        self._cx = self._cy = 0.0
        self._major = self._minor = 0.0
        self._have = False
        self._age = self.lock_max_age

    def _select(self, candidates, width) -> dict:
        """Nearest to the previous centre if we hold a lock, else the largest.

        "Largest wins" is wrong mid-approach: a bigger blue object entering the frame
        would steal the lock. Gating on distance from the last centre keeps it.
        """
        if not self._have:
            return candidates[0]
        gate = self.gate_frac * width
        prev = np.array([self._cx, self._cy])
        best, best_d = None, None
        for c in candidates:
            d = float(np.linalg.norm(np.array(c["center_px"]) - prev))
            if d <= gate and (best_d is None or d < best_d):
                best, best_d = c, d
        return best if best is not None else candidates[0]

    def update(self, bgr) -> RingObs:
        height, width = bgr.shape[:2]
        mask = blue_mask(bgr, self.params)
        blue_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        candidates = detect_rings(bgr, self.params)

        if not candidates:
            self._age = min(self._age + 1, self.lock_max_age)
            coasting = self._have and self._age < self.lock_max_age
            if not coasting:
                self._have = False
            return self._emit(width, height, fresh=False, found=coasting, clipped=False,
                              blue_fraction=blue_fraction, n=0, raw=None)

        pick = self._select(candidates, width)
        cx, cy = pick["center_px"]
        major, minor = pick["major_px"], pick["minor_px"]
        a = self.ema if self._have else 1.0          # first lock snaps, then smooths
        self._cx += a * (cx - self._cx)
        self._cy += a * (cy - self._cy)
        self._major += a * (major - self._major)
        self._minor += a * (minor - self._minor)
        self._have = True
        self._age = 0
        return self._emit(width, height, fresh=True, found=True,
                          clipped=bool(pick["clipped"]), blue_fraction=blue_fraction,
                          n=len(candidates), raw=pick)

    def _emit(self, width, height, *, fresh, found, clipped, blue_fraction, n, raw) -> RingObs:
        if not self._have:
            return RingObs(found=False, fresh=False, age=self.lock_max_age,
                           blue_fraction=blue_fraction, n_candidates=n, raw=raw)
        half_w = width / 2.0
        # Both offsets are divided by HALF THE WIDTH, including the vertical one. That
        # puts them in the same angular units (radians per unit, via the same focal
        # length), so one proportional gain is correct for both axes. Dividing cy by
        # half the height instead would silently make the vertical gain 4/3 too large.
        return RingObs(
            found=found, fresh=fresh,
            cx=float((self._cx - half_w) / half_w),
            cy=float((self._cy - height / 2.0) / half_w),
            major_px=float(self._major), minor_px=float(self._minor),
            axis_ratio=float(self._minor / self._major) if self._major > 1e-6 else 0.0,
            scale=float(self._major / width) if width else 0.0,
            range_m=spec.range_from_major_axis(self._major, width, self.ring_diameter_m,
                                               self.hfov_deg),
            age=int(self._age), clipped=clipped, blue_fraction=blue_fraction,
            n_candidates=n, raw=raw,
        )
