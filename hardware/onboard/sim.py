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

Robustness (domain randomisation), off unless asked for: `dr` in [0, 1] widens everything
that is uncertain about the real setup -- the drone's dynamics (each episode a profile drawn
between "ideal" and "harsh", and the unverified rc->speed / yaw-rate calibration off by up to
25 / 20 %), where the ring hangs, a ring that swings on its rope in a per-episode wind (about
the knot at its top, like scripts/hoop_net_env.py's), and detector noise and latency.
`wash_deg` models what the real noodle does (observed in the cage, 2026-10-11): it hardly swings
on its own, but the drone's prop wash pushes it away as the drone comes close, tilting it up about
the rope at its top (the bottom moves away from the drone, the face tilts up). The tilt follows
the push through a damped spring and settles back once the drone has gone.
`hard_start_p` is the chance an episode starts somewhere other than the measured front
runway: behind, beside or below the ring, or anywhere in the cage, often facing away, so the
ring has to be searched for. `ramp` in [0, 1] scales both (the training curriculum). At
dr = hard_start_p = 0 the random stream and every result are exactly as before.

What it does NOT model: propeller aerodynamics, the ring moving when struck, motion blur,
rolling shutter, lighting, or anything about the real ring's colour. A policy that works
here has not been shown to work in the cage -- see hardware/README.md 3.5 for the same
caveat about the project's other fake drone.
"""
import math
from dataclasses import dataclass, fields, replace
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
WALL_MARGIN = 0.05     # starts are kept this far inside the net

# domain randomisation at full strength (dr = 1)
SPEED_SCALE_DR = 0.25  # rc -> m/s calibration off by up to +-25 % (RC_PER_MPS is UNVERIFIED)
YAW_SCALE_DR = 0.20
SWING_TWIST_DEG = 12.0  # a hanging ring also twists on its rope
SWING_PERIOD_S = (1.5, 3.0)
SWING_GUST_MIN = 0.5   # every windy episode really swings
KNOT_DRIFT_M = 0.04    # the rope's knot moves too, so the whole ring translates
DEFAULT_SWAY_DEG = 25.0
WASH_RANGE_M = 0.45    # Gaussian reach of the prop-wash push from the ring centre
WASH_STRENGTH = (0.5, 1.5)   # per-episode multiplier of wash_deg
WASH_PERIOD_S = 2.2
WASH_DAMPING = 0.6
WASH_SIDE_SOFT_M = 0.10   # the push direction fades through the ring's plane over about this
LATENCY_P = 0.15       # chance per episode that the camera lags one tick behind
NOISE_CXY = 0.02       # detector noise (half-frames), size (fraction), axis ratio
NOISE_MAJOR = 0.05
NOISE_RATIO = 0.04
START_TYPES = ("front", "behind", "beside", "below", "anywhere")
# open_plane: crossing the plane clear of the noodle (under it, around it) is no outcome --
# in the real cage that is just flying past the ring, not a failure. Only a pass or a strike
# ends the episode then.


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


def sample_profile(rng, dr: float) -> Profile:
    """A drone somewhere between "ideal" and "harsh", `dr` of the way out from "typical"."""
    typ, lo_p, hi_p = PROFILES["typical"], PROFILES["ideal"], PROFILES["harsh"]
    vals = {}
    for f in fields(Profile):
        if f.name == "name":
            continue
        lo, hi = sorted((getattr(lo_p, f.name), getattr(hi_p, f.name)))
        t = getattr(typ, f.name)
        vals[f.name] = t + dr * (float(rng.uniform(lo, hi)) - t)
    return Profile(name="randomised", **vals)


@dataclass
class Ring:
    centre: np.ndarray            # world x, y, z (m)
    normal_deg: float             # heading of the ring's axis, deg; 0 = axis along +x
    opening_m: float = spec.RING_OPENING_M      # the clear gap: decides pass vs strike
    noodle_m: float = spec.RING_NOODLE_M
    tilt_deg: float = 0.0         # face tilted up (+) / down (-), e.g. pushed by prop wash

    @property
    def radius(self) -> float:
        return self.opening_m / 2.0

    @property
    def outer_m(self) -> float:
        """What the camera measures, and so what the range estimate is based on."""
        return self.opening_m + 2 * self.noodle_m

    @property
    def normal(self) -> np.ndarray:
        r, t = math.radians(self.normal_deg), math.radians(self.tilt_deg)
        return np.array([math.cos(t) * math.cos(r), math.cos(t) * math.sin(r), math.sin(t)])


@dataclass
class Outcome:
    """Set once, when the drone crosses the ring's plane."""
    crossed: bool = False
    passed: bool = False          # through the opening
    struck: bool = False          # hit the noodle itself
    missed: bool = False          # crossed the plane outside the ring
    contact: bool = False         # struck by touching the noodle, not at a plane crossing
    radial_m: float = float("nan")  # distance from the ring axis at the crossing
    direction: int = 0            # +1: crossed from the -normal side to the +normal side

    @property
    def done(self) -> bool:
        return self.crossed or self.struck


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
                 drone_width_m: float = spec.DRONE_WIDTH_M,
                 dr: float = 0.0, sway_deg: float = DEFAULT_SWAY_DEG, hard_start_p: float = 0.0,
                 ramp: float = 1.0, open_plane: bool = False, wash_deg: float = 0.0):
        """dr / sway_deg / hard_start_p / ramp: see the module docstring. open_plane: a crossing
        of the ring's plane clear of the noodle (under it, around it) is not an outcome and the
        episode goes on (so there is no "missed"); needed once starts can be beside or below."""
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.base_profile = self.profile
        self.dr, self.sway_deg, self.hard_start_p = float(dr), float(sway_deg), float(hard_start_p)
        self.ramp = float(ramp)
        self.open_plane = bool(open_plane)
        self.wash_deg = float(wash_deg)
        self._wash = None
        self.speed_scale = self.yaw_scale = 1.0
        self.start_type = "front"
        self._swing = None
        self._latency = False
        self._prev_pose = None
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
    # -- robustness knobs (callable through VecEnv.env_method via the gym env) ----------
    @property
    def dr_eff(self) -> float:
        return self.dr * self.ramp

    @property
    def hard_p_eff(self) -> float:
        return self.hard_start_p * self.ramp

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        dr = self.dr_eff
        self.ring = self._fixed_ring or self._sample_ring()
        self.ring0 = self.ring                  # this episode's resting pose (it may swing)
        self._swing = None
        if dr > 0:
            self._swing = dict(
                dir=float(self.rng.uniform(0, 2 * np.pi)),
                gust=float(self.rng.uniform(SWING_GUST_MIN, 1)),
                w=2 * np.pi / float(self.rng.uniform(*SWING_PERIOD_S)),
                ph=float(self.rng.uniform(0, 2 * np.pi)), ph2=float(self.rng.uniform(0, 2 * np.pi)))
            self.profile = sample_profile(self.rng, dr)
            self.speed_scale = 1.0 + float(self.rng.uniform(-1, 1)) * SPEED_SCALE_DR * dr
            self.yaw_scale = 1.0 + float(self.rng.uniform(-1, 1)) * YAW_SCALE_DR * dr
            self._latency = bool(self.rng.random() < LATENCY_P * dr)
        else:
            self.profile = self.base_profile
            self.speed_scale = self.yaw_scale = 1.0
            self._latency = False
        pos, yaw = self._fixed_start
        self.start_type = "front"
        if pos is None and self.hard_p_eff > 0 and self.rng.random() < self.hard_p_eff:
            self.start_type = START_TYPES[1 + int(self.rng.integers(len(START_TYPES) - 1))]
        self.pos = np.array(pos, dtype=float) if pos is not None else self._sample_start()
        self.yaw_deg = float(yaw) if yaw is not None else self._sample_yaw()
        self._prev_pose = (self.pos.copy(), self.yaw_deg)
        self.vel = np.zeros(3)
        self.t = 0.0
        self.outcome = Outcome()
        self._drift = np.zeros(3)
        self._pending = []                       # (apply_time, action) command delay
        self._applied = np.zeros(spec.ACT_DIM)
        # prop wash: drawn last and only when on, so runs without it keep their random streams
        self._wash = None
        if self.wash_deg > 0 and self._fixed_ring is None:
            self._wash = dict(k=float(self.rng.uniform(*WASH_STRENGTH)), ang=0.0, rate=0.0)
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
        # With dr > 0 each range widens (it may be re-hung); written as uniform(-w, w) so that
        # at dr = 0 the draws are exactly the ones they always were.
        dr = self.dr_eff
        wy, wz, wyaw = 0.10 + 0.05 * dr, 0.03 + 0.07 * dr, 20.0 + 10.0 * dr
        c = np.array([self.rng.uniform(0.05, 0.25),
                      0.40 + self.rng.uniform(-wy, wy),
                      1.20 + self.rng.uniform(-wz, wz)])
        return Ring(c, float(self.rng.uniform(-wyaw, wyaw)), self.ring_opening_m,
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
        if self.start_type != "front":
            return self._sample_hard_start()
        dr = self.dr_eff
        x = max(-0.95, self.ring.centre[0] - self.rng.uniform(1.00 - 0.10 * dr, 1.10 + 0.05 * dr))
        wy, wz = 0.20 + 0.05 * dr, 0.05 + 0.05 * dr
        return np.array([x,
                         self.ring.centre[1] + self.rng.uniform(-wy, wy),
                         TAKEOFF_Z + self.rng.uniform(-wz, wz)])

    def _sample_hard_start(self) -> np.ndarray:
        """Behind, beside or below the ring, or anywhere: relative to the ring's axis n and
        the horizontal in-plane direction u, at about takeoff height, kept inside the net."""
        c, n = self.ring.centre, self.ring.normal
        u = np.array([-n[1], n[0], 0.0])
        U = self.rng.uniform
        z = TAKEOFF_Z + U(-0.10, 0.10)
        if self.start_type == "behind":          # the far side (only ~0.75 m of cage there)
            p = c + n * U(0.45, 0.75) + u * U(-0.20, 0.20)
        elif self.start_type == "beside":        # in or near the ring's plane, off to a side
            side = 1.0 if self.rng.random() < 0.5 else -1.0
            p = c + u * side * U(0.35, 0.70) + n * U(-0.20, 0.20)
        elif self.start_type == "below":         # under it: back off and climb into it
            p = c + n * U(-0.25, 0.25) + u * U(-0.20, 0.20)
            z = U(0.70, 0.90)
        else:                                    # anywhere, clear of the ring
            lim = CAGE_XY - WALL_MARGIN
            for _ in range(20):
                p = np.array([U(-lim, lim), U(-lim, lim), 0.0])
                if np.hypot(*(p - c)[:2]) > 0.25:
                    break
        lim = CAGE_XY - WALL_MARGIN
        return np.array([float(np.clip(p[0], -lim, lim)), float(np.clip(p[1], -lim, lim)),
                         float(np.clip(z, 0.30, CAGE_Z[1] - 0.10))])

    def _sample_yaw(self) -> float:
        d = self.ring.centre - self.pos
        w = 180.0 if self.start_type != "front" else 25.0 + 10.0 * self.dr_eff
        return math.degrees(math.atan2(d[1], d[0])) + float(self.rng.uniform(-w, w))

    def _update_ring(self):
        """The ring swinging on its rope (dr > 0): a pendulum about the knot at its top, along
        this episode's wind direction, plus a little twist."""
        sw, wa = self._swing, self._wash
        if (sw is None and wa is None) or self._fixed_ring is not None:
            return
        arm = self.ring0.opening_m / 2.0 + self.ring0.noodle_m      # knot -> centre
        c, ndeg, tilt = self.ring0.centre.copy(), self.ring0.normal_deg, 0.0
        if sw is not None:
            k = self.dr_eff * sw["gust"]
            ang = math.radians(k * self.sway_deg * math.sin(sw["w"] * self.t + sw["ph"]))
            twist = k * SWING_TWIST_DEG * math.sin(0.77 * sw["w"] * self.t + sw["ph2"])
            s = np.array([math.cos(sw["dir"]), math.sin(sw["dir"]), 0.0])
            knot = k * KNOT_DRIFT_M * math.sin(sw["w"] * self.t + sw["ph"] + math.pi / 2)
            c = (c + (arm * math.sin(ang) + knot) * s
                 + np.array([0.0, 0.0, arm * (1.0 - math.cos(ang))]))
            ndeg = ndeg + twist
        if wa is not None:
            # positive angle = bottom pushed towards +normal (away from a drone on the -normal side)
            n0 = self.ring0.normal
            d = self.pos - self.ring.centre
            side = float(np.dot(d, n0))
            # tanh, not sign: inside the opening the push has no side, so the ring does not snap
            # back into a drone that is halfway through
            target = (-math.tanh(side / WASH_SIDE_SOFT_M) * self.wash_deg * self.ramp * wa["k"]
                      * math.exp(-(float(np.linalg.norm(d)) / WASH_RANGE_M) ** 2))
            w = 2 * math.pi / WASH_PERIOD_S
            wa["rate"] += (w * w * (target - wa["ang"]) - 2 * WASH_DAMPING * w * wa["rate"]) * spec.CTRL_DT
            wa["ang"] += wa["rate"] * spec.CTRL_DT
            b = math.radians(wa["ang"])
            c = c + arm * math.sin(b) * n0 + np.array([0.0, 0.0, arm * (1.0 - math.cos(b))])
            tilt = wa["ang"]
        self.ring = replace(self.ring0, centre=c, normal_deg=ndeg, tilt_deg=tilt)

    @property
    def wash_tilt_deg(self) -> float:
        return 0.0 if self._wash is None else float(self._wash["ang"])

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

    def rim_distance(self) -> float:
        """Gap from the drone's edge to the noodle's surface (m); <= 0 is contact. The noodle
        is a torus: centre circle radius + noodle/2, tube radius noodle/2."""
        d = self.pos - self.ring.centre
        n = self.ring.normal
        axial = float(np.dot(d, n))
        radial = float(np.linalg.norm(d - axial * n))
        tube = self.ring.noodle_m / 2.0
        to_tube = math.hypot(radial - (self.ring.radius + tube), axial)
        return to_tube - tube - self.drone_radius_m

    def _check_contact(self):
        """Touching the noodle anywhere -- before the plane, beside it, or the swinging ring
        running into the drone -- is a strike, not only a crossing inside the strike band."""
        if self.outcome.done or self.rim_distance() > 0.0:
            return
        d = self.pos - self.ring.centre
        radial = float(np.linalg.norm(d - np.dot(d, self.ring.normal) * self.ring.normal))
        self.outcome = Outcome(struck=True, contact=True, radial_m=radial)

    def _check_crossing(self, prev_pos):
        """Did this step cross the ring's plane, and if so where relative to the opening.

        Both ends are measured against the ring's CURRENT pose: otherwise a swinging ring sweeping
        its plane past a still drone reads as a pass (a ring swinging into the drone is a contact)."""
        side = self._plane_side()
        self._prev_side = float(np.dot(prev_pos - self.ring.centre, self.ring.normal))
        if self.outcome.done or side == 0.0 or np.sign(side) == np.sign(self._prev_side):
            self._prev_side = side
            return
        # Linear interpolation to the crossing point.
        denom = side - self._prev_side
        f = 0.0 if abs(denom) < 1e-12 else -self._prev_side / denom
        hit = prev_pos + f * (self.pos - prev_pos)
        radial = float(np.linalg.norm(
            (hit - self.ring.centre) - np.dot(hit - self.ring.centre, self.ring.normal)
            * self.ring.normal))
        if self.open_plane and radial > (self.ring.radius + self.ring.noodle_m
                                         + self.drone_radius_m):
            self._prev_side = side           # cleared the ring (under / around it): fly on
            return
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
                               missed=radial > outer,
                               direction=1 if side > 0 else -1)
        self._prev_side = side

    def clear_outcome(self):
        """Score the next crossing too (lap mode: a pass does not end the episode)."""
        self.outcome = Outcome()

    # -- dynamics ---------------------------------------------------------------------
    def step(self, action):
        p = self.profile
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=float).reshape(-1)[:4]), -1, 1)
        a = np.where(np.abs(a) < p.deadband, 0.0, a)

        self._pending.append((self.t + p.delay_s, a.copy()))
        while self._pending and self._pending[0][0] <= self.t:
            self._applied = self._pending.pop(0)[1]
        cmd = self._applied

        vmax = MAX_SPEED * self.speed_scale
        want = (cmd[spec.ACT_FB] * vmax * self.forward
                + cmd[spec.ACT_LR] * vmax * self.right
                + np.array([0.0, 0.0, cmd[spec.ACT_UD] * vmax]))
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
        self._prev_pose = (prev_pos, self.yaw_deg)
        self.pos = self.pos + (self.vel + self._drift) * spec.CTRL_DT
        self.yaw_deg += (-cmd[spec.ACT_YAW] * MAX_YAW_DPS * self.yaw_scale
                         + self.rng.normal(0, p.yaw_drift_dps)) * spec.CTRL_DT
        self.yaw_deg = (self.yaw_deg + 180.0) % 360.0 - 180.0
        self.pos[2] = max(self.pos[2], 0.0)
        self.t += spec.CTRL_DT
        self.battery = max(0.0, self.battery - 0.02)
        self._update_ring()

        self._check_crossing(prev_pos)
        self._check_contact()
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

    def _project(self, pos=None, yaw_deg=None):
        """(cx, cy, major_px, axis_ratio, zf) of the ring, or None if behind the camera.
        pos / yaw_deg: project from another pose (the one-tick camera latency)."""
        pos = self.pos if pos is None else pos
        r = math.radians(self.yaw_deg if yaw_deg is None else yaw_deg)
        forward = np.array([math.cos(r), math.sin(r), 0.0])
        right = np.array([math.sin(r), -math.cos(r), 0.0])
        d = self.ring.centre - pos
        zf = float(np.dot(d, forward))
        if zf <= 0.05:
            return None
        half_tan = math.tan(math.radians(spec.CAMERA_HFOV_DEG) / 2.0)
        cx = (float(np.dot(d, right)) / zf) / half_tan
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
        proj = self._project(*self._prev_pose) if self._latency and self._prev_pose else self._project()
        if proj is None or self.rng.random() < self.profile.detect_miss_p:
            return lost
        cx, cy, major, ratio, zf = proj
        dr = self.dr_eff
        if dr > 0:                         # the real detector is not exact
            cx += float(self.rng.normal(0, NOISE_CXY * dr))
            cy += float(self.rng.normal(0, NOISE_CXY * dr))
            noisy = max(1e-3, 1.0 + float(self.rng.normal(0, NOISE_MAJOR * dr)))
            major *= noisy
            zf /= noisy                    # the range estimate comes from the major axis
            ratio = float(np.clip(ratio + self.rng.normal(0, NOISE_RATIO * dr), 0.05, 1.0))
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
