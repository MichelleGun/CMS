"""
A lightweight simulator of the onboard-camera hoop task.

Why this exists: RL cannot be trained on a real drone (thousands of episodes, crashes),
and the project's PyBullet simulator is the wrong task -- it has no ring, no ring plane
and no front camera, and it rewards stopping at a point. This gives the traversal task a
trainable environment, and gives the whole package something to be tested against with
no hardware.

What it models: a drone whose commanded rc becomes velocity through a first-order lag,
with command delay, a dead-band, wind-like drift and quantised telemetry; a ring with a
position, a plane normal and a radius; and a pinhole front camera. The drone has a WIDTH,
so a traversal is scored on whether its edge clears the noodle -- at the measured 30 cm
opening that is the whole question, and a point-mass drone answers it wrongly by 13
points on the harsh profile.

Two observation modes, deliberately:
  * render=False -- the ring's projection is computed analytically straight into a
    RingObs. Fast enough for RL (no OpenCV per step).
  * render=True  -- a frame is drawn and the REAL ring_detect.RingTracker runs on it, so
    a dry run exercises the actual detector, thresholds and tracker that will fly. This
    is what fly_hoop.py --dry-run uses.
The analytic path is a model of the detector, not the detector. Only render=True proves
anything about detection; tests assert the two agree on clean geometry.

What it does NOT model: propeller aerodynamics, the ring moving when struck, motion blur,
rolling shutter, lighting, or anything about the real ring's colour. A policy that works
here has not been shown to work in the cage -- see hardware/README.md 3.5 for the same
caveat about the project's other fake drone.
"""
import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from . import spec
from .ring_detect import DetectorParams, RingObs, RingTracker

# --- hardware-derived scale factors ---------------------------------------------------
# Both UNVERIFIED on this drone, the same status as hardware/config.yaml's conventions.
# The tutorial this project started from measured rc 15 -> ~12 cm/s, i.e. ~125 rc per
# m/s, which is what calibrate_speed.py exists to replace with a measurement.
RC_PER_MPS = 125.0
YAW_DPS_PER_RC = 1.0

MAX_SPEED = spec.RC_MAX / RC_PER_MPS          # m/s at |action| = 1  (~0.16)
MAX_YAW_DPS = spec.RC_MAX * YAW_DPS_PER_RC    # deg/s at |action| = 1  (~20)

CAGE_XY = 1.0          # half-width; |x|,|y| <= 1 m
CAGE_Z = (0.0, 2.0)
TAKEOFF_Z = 0.80       # the Tello settles near here by itself


@dataclass
class Profile:
    """How badly the pretend drone behaves. Mirrors fake_tello.py's three presets."""
    name: str = "typical"
    tau_accel: float = 0.35        # s, first-order lag spinning up
    tau_decel: float = 0.55        # s, slower to stop than to start
    delay_s: float = 0.10          # command transport delay
    deadband: float = 0.08         # |action| below this does nothing
    drift_mps: float = 0.015       # wind-like random walk
    yaw_drift_dps: float = 1.5
    vel_noise_mps: float = 0.02
    height_quant_cm: float = 1.0
    drop_frame_p: float = 0.02
    detect_miss_p: float = 0.05    # a frame where the detector would fail anyway


PROFILES = {
    "ideal": Profile("ideal", 0.15, 0.20, 0.03, 0.02, 0.0, 0.0, 0.004, 1.0, 0.0, 0.0),
    "typical": Profile("typical"),
    "harsh": Profile("harsh", 0.55, 0.85, 0.18, 0.14, 0.035, 3.5, 0.045, 2.0, 0.08, 0.15),
}


@dataclass
class Ring:
    centre: np.ndarray            # world x, y, z (m)
    normal_deg: float             # heading of the ring's axis, deg; 0 = axis along +x
    opening_m: float = spec.RING_OPENING_M      # the clear gap: decides pass vs strike
    noodle_m: float = spec.RING_NOODLE_M

    @property
    def radius(self) -> float:
        return self.opening_m / 2.0

    @property
    def outer_m(self) -> float:
        """What the camera measures, and so what the range estimate is based on."""
        return self.opening_m + 2 * self.noodle_m

    @property
    def normal(self) -> np.ndarray:
        r = math.radians(self.normal_deg)
        return np.array([math.cos(r), math.sin(r), 0.0])


@dataclass
class Outcome:
    """Set once, when the drone crosses the ring's plane."""
    crossed: bool = False
    passed: bool = False          # through the opening
    struck: bool = False          # hit the noodle itself
    missed: bool = False          # crossed the plane outside the ring
    radial_m: float = float("nan")  # distance from the ring axis at the crossing


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


class OnboardSim:
    """One episode. step(action) advances by spec.CTRL_DT."""

    FRAME_W, FRAME_H = 480, 360      # smaller than the Tello's 960x720: the detector is
                                     # scale-free and this keeps rendering cheap

    def __init__(self, ring: Optional[Ring] = None, profile="typical", seed=0,
                 render=False, detector: Optional[DetectorParams] = None,
                 start_pos=None, start_yaw_deg: Optional[float] = None,
                 ring_opening_m: float = spec.RING_OPENING_M,
                 ring_noodle_m: float = spec.RING_NOODLE_M,
                 drone_width_m: float = spec.DRONE_WIDTH_M):
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.rng = np.random.default_rng(seed)
        self.render = bool(render)
        self.ring_opening_m = float(ring_opening_m)
        self.ring_noodle_m = float(ring_noodle_m)
        # Half-width, because every comparison below is radial. 0.0 reproduces the old
        # point-mass behaviour, which is what the pure-geometry tests want.
        self.drone_radius_m = float(drone_width_m) / 2.0
        self.detector = detector or DetectorParams()
        # The tracker is given the OUTER diameter, because that is the edge it measures.
        self._tracker = RingTracker(
            self.detector,
            ring_diameter_m=self.ring_opening_m + 2 * self.ring_noodle_m,
        ) if self.render else None
        self._fixed_ring = ring
        self._fixed_start = (start_pos, start_yaw_deg)
        self.reset()

    # -- episode ----------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ring = self._fixed_ring or self._sample_ring()
        pos, yaw = self._fixed_start
        self.pos = np.array(pos, dtype=float) if pos is not None else self._sample_start()
        self.yaw_deg = float(yaw) if yaw is not None else self._sample_yaw()
        self.vel = np.zeros(3)
        self.t = 0.0
        self.outcome = Outcome()
        self._drift = np.zeros(3)
        self._pending = []                       # (apply_time, action) command delay
        self._applied = np.zeros(spec.ACT_DIM)
        self._prev_side = self._plane_side()
        self.battery = 85.0
        if self._tracker is not None:
            self._tracker.reset()
        return self.observe()

    def _sample_ring(self) -> Ring:
        # MEASURED from the real cage, so a pass rate here is a statement about THIS cage
        # rather than a generic one: centre 1.20 m up, hung 40 cm off the centreline
        # (60 cm from the near side wall), axis roughly along +-x as the real one is.
        # Biased to +x so the drone has somewhere to stand off from on the -x side.
        c = np.array([self.rng.uniform(0.05, 0.25),
                      0.40 + self.rng.uniform(-0.10, 0.10),
                      1.20 + self.rng.uniform(-0.03, 0.03)])
        return Ring(c, float(self.rng.uniform(-20, 20)), self.ring_opening_m,
                    self.ring_noodle_m)

    def _sample_start(self) -> np.ndarray:
        # On the far -x side of the ring, facing it, at a Tello's own takeoff height.
        # Standoff reflects the MEASURED 110 cm runway on this side (the other side has
        # only 90 cm). There is simply not much room, which is the core difficulty of this
        # task and must not be sampled away.
        #
        # Note the consequence of a 1.20 m ring and an 0.80 m takeoff: the drone starts
        # 40 cm BELOW the ring and has to climb into it. With a fixed forward camera that
        # puts the ring centre near cy = -0.52 at this standoff, with its top edge just
        # outside the frame, so the first detection of every episode is a clipped one and
        # ring_detect's arc-span path is load-bearing at acquisition, not just on approach.
        x = max(-0.95, self.ring.centre[0] - self.rng.uniform(1.00, 1.10))
        return np.array([x,
                         self.ring.centre[1] + self.rng.uniform(-0.20, 0.20),
                         TAKEOFF_Z + self.rng.uniform(-0.05, 0.05)])

    def _sample_yaw(self) -> float:
        d = self.ring.centre - self.pos
        return math.degrees(math.atan2(d[1], d[0])) + float(self.rng.uniform(-25, 25))

    # -- geometry ---------------------------------------------------------------------
    @property
    def forward(self) -> np.ndarray:
        r = math.radians(self.yaw_deg)
        return np.array([math.cos(r), math.sin(r), 0.0])

    @property
    def right(self) -> np.ndarray:
        # forward x up, with z up: facing +x, right is -y.
        r = math.radians(self.yaw_deg)
        return np.array([math.sin(r), -math.cos(r), 0.0])

    def _plane_side(self) -> float:
        return float(np.dot(self.pos - self.ring.centre, self.ring.normal))

    def _check_crossing(self, prev_pos):
        """Did this step cross the ring's plane, and if so where relative to the opening."""
        side = self._plane_side()
        if self.outcome.crossed or side == 0.0 or np.sign(side) == np.sign(self._prev_side):
            self._prev_side = side
            return
        # Linear interpolation to the crossing point.
        denom = side - self._prev_side
        f = 0.0 if abs(denom) < 1e-12 else -self._prev_side / denom
        hit = prev_pos + f * (self.pos - prev_pos)
        radial = float(np.linalg.norm(
            (hit - self.ring.centre) - np.dot(hit - self.ring.centre, self.ring.normal)
            * self.ring.normal))
        # The drone is an aircraft, not a point, and that distinction is the whole
        # question at a 30 cm opening: its EDGE has to clear the noodle, so the pass band
        # shrinks by the drone's half-width and the strike band grows by it on both sides.
        # Scored as a point instead, a 14 cm radial error reads as a clean pass through a
        # 30 cm hole -- measured, that overstated the harsh-profile pass rate by 13 points,
        # and it overstates by most exactly where the margin is thinnest.
        r = self.drone_radius_m
        inner = max(self.ring.radius - r, 0.0)
        outer = self.ring.radius + self.ring.noodle_m + r
        self.outcome = Outcome(crossed=True, radial_m=radial,
                               passed=radial <= inner,
                               struck=inner < radial <= outer,
                               missed=radial > outer)
        self._prev_side = side

    # -- dynamics ---------------------------------------------------------------------
    def step(self, action):
        p = self.profile
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=float).reshape(-1)[:4]), -1, 1)
        a = np.where(np.abs(a) < p.deadband, 0.0, a)

        self._pending.append((self.t + p.delay_s, a.copy()))
        while self._pending and self._pending[0][0] <= self.t:
            self._applied = self._pending.pop(0)[1]
        cmd = self._applied

        want = (cmd[spec.ACT_FB] * MAX_SPEED * self.forward
                + cmd[spec.ACT_LR] * MAX_SPEED * self.right
                + np.array([0.0, 0.0, cmd[spec.ACT_UD] * MAX_SPEED]))
        # Braking is slower than accelerating, which is the single biggest reason a real
        # Tello overshoots a visual target.
        speeding = np.linalg.norm(want) >= np.linalg.norm(self.vel)
        tau = p.tau_accel if speeding else p.tau_decel
        alpha = 1.0 - math.exp(-spec.CTRL_DT / max(tau, 1e-3))
        self.vel += alpha * (want - self.vel)

        if p.drift_mps > 0:
            self._drift += self.rng.normal(0, p.drift_mps * spec.CTRL_DT, 3)
            self._drift *= 0.98
        prev_pos = self.pos.copy()
        self.pos = self.pos + (self.vel + self._drift) * spec.CTRL_DT
        self.yaw_deg += (-cmd[spec.ACT_YAW] * MAX_YAW_DPS
                         + self.rng.normal(0, p.yaw_drift_dps)) * spec.CTRL_DT
        self.yaw_deg = (self.yaw_deg + 180.0) % 360.0 - 180.0
        self.pos[2] = max(self.pos[2], 0.0)
        self.t += spec.CTRL_DT
        self.battery = max(0.0, self.battery - 0.02)

        self._check_crossing(prev_pos)
        return self.observe()

    def out_of_cage(self) -> bool:
        return bool(abs(self.pos[0]) > CAGE_XY or abs(self.pos[1]) > CAGE_XY
                    or not CAGE_Z[0] <= self.pos[2] <= CAGE_Z[1])

    # -- sensing ----------------------------------------------------------------------
    def state(self) -> dict:
        """A Tello state packet, so obs.ObsBuilder runs unchanged.

        The INT/FLOAT split matches djitellopy's own state parser, and that matters: the
        kit's safety checks are written as `isinstance(value, int)` (with an explicit
        `isinstance(value, bool)` guard, since bool is an int in Python). Returning a
        float h here would make survey.health reject every airborne tick -- which it
        did, until this was fixed. djitellopy parses mid/x/y/z/pitch/roll/yaw/vg*/
        templ/temph/tof/h/bat/time as int and baro/agx/agy/agz as float.
        """
        p = self.profile
        q = max(p.height_quant_cm, 1e-6)
        v = self.vel + self.rng.normal(0, p.vel_noise_mps, 3)
        return {
            "h": int(round(self.pos[2] * 100.0 / q) * q),
            "tof": int(round(max(self.pos[2], 0.01) * 100.0)),
            # Tello velocities are body-frame; the unit is UNVERIFIED (config.yaml
            # assumes dm/s via vel_to_mps 0.1). Reported here in cm/s.
            "vgx": int(round(np.dot(v, self.forward) * 100.0)),
            "vgy": int(round(np.dot(v, self.right) * 100.0)),
            "vgz": int(round(v[2] * 100.0)),
            "pitch": int(round(-self._applied[spec.ACT_FB] * 8.0)),
            "roll": int(round(self._applied[spec.ACT_LR] * 8.0)),
            "yaw": int(round(self.yaw_deg)),
            "bat": int(self.battery),
            "temph": 62,
            "templ": 58,
            "baro": float(round(self.pos[2] * 100.0, 2)),
            "agx": 0.0, "agy": 0.0, "agz": -999.0,
            "time": int(self.t),
        }

    def _project(self):
        """(cx, cy, major_px, axis_ratio, zf) of the ring, or None if behind the camera."""
        d = self.ring.centre - self.pos
        zf = float(np.dot(d, self.forward))
        if zf <= 0.05:
            return None
        half_tan = math.tan(math.radians(spec.CAMERA_HFOV_DEG) / 2.0)
        cx = (float(np.dot(d, self.right)) / zf) / half_tan
        cy = (-float(d[2]) / zf) / half_tan
        major = spec.focal_px(self.FRAME_W) * self.ring.outer_m / zf
        # Foreshortening: a circle seen off its axis projects to an ellipse whose minor
        # axis shrinks by the cosine between the ring's normal and the line of sight.
        ratio = abs(float(np.dot(self.ring.normal, _unit(d))))
        return cx, cy, major, max(ratio, 0.05), zf

    def frame(self):
        """A BGR frame with the ring drawn, for the render=True path."""
        import cv2
        img = np.full((self.FRAME_H, self.FRAME_W, 3), 70, np.uint8)
        img[: self.FRAME_H // 3] = (90, 80, 75)        # a crude horizon, so the mask has
                                                       # something other than the ring
        proj = self._project()
        if proj is not None and self.rng.random() > self.profile.drop_frame_p:
            cx, cy, major, ratio, _zf = proj
            half_w = self.FRAME_W / 2.0
            px = int(round(half_w + cx * half_w))
            py = int(round(self.FRAME_H / 2.0 + cy * half_w))
            ax = int(round(major / 2.0))
            ay = max(1, int(round(major * ratio / 2.0)))
            if ax >= 2:
                thick = max(2, int(round(major * self.ring.noodle_m / self.ring.outer_m)))
                cv2.ellipse(img, (px, py), (ax, ay), self.ring.normal_deg * 0.0,
                            0, 360, (235, 120, 40), thick)   # BGR: saturated blue
        return img

    def observe(self) -> RingObs:
        if self.render:
            return self._tracker.update(self.frame())
        return self._analytic_obs()

    def _visible_arc_deg(self, cx, cy, scale, ratio):
        """How much of the ring's outline falls inside the frame, in degrees.

        Mirrors ring_detect.arc_span_deg, so the analytic path fails where the real
        detector fails. Working in half-width units: the frame is cx in [-1, 1] and cy in
        [-H/W, H/W], and the ellipse's semi-axes are `scale` and `scale * ratio`.
        """
        lim_y = self.FRAME_H / self.FRAME_W
        th = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        px = cx + scale * np.cos(th)
        py = cy + scale * ratio * np.sin(th)
        inside = (np.abs(px) <= 1.0) & (np.abs(py) <= lim_y)
        if not inside.any():
            return 0.0
        if inside.all():
            return 360.0
        deg = np.degrees(th[inside])
        deg.sort()
        gaps = np.diff(np.concatenate([deg, [deg[0] + 360.0]]))
        return float(360.0 - gaps.max())

    def _analytic_obs(self) -> RingObs:
        lost = RingObs(found=False, fresh=False, age=spec.LOCK_MAX_AGE,
                       range_m=float("inf"))
        proj = self._project()
        if proj is None or self.rng.random() < self.profile.detect_miss_p:
            return lost
        cx, cy, major, ratio, zf = proj
        scale = major / self.FRAME_W
        # The real detector's three failure modes, in the same order it applies them:
        # too small to fit, and -- once the ring clips the frame -- too little arc left
        # to fit reliably.
        if major < self.detector.min_major_px:
            return lost
        arc = self._visible_arc_deg(cx, cy, scale, ratio)
        clipped = arc < 359.0
        if clipped and arc < self.detector.min_arc_deg:
            return lost
        return RingObs(found=True, fresh=True, cx=cx, cy=cy, major_px=major,
                       minor_px=major * ratio, axis_ratio=ratio, scale=scale,
                       range_m=zf, age=0, clipped=clipped,
                       blue_fraction=min(0.9, scale ** 2 * 1.2), n_candidates=1)
