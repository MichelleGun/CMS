"""
Builds the spec.OBS_DIM observation from a RingObs plus one Tello state packet.

The same builder runs on the real drone and in sim.py, so a policy cannot tell the two
apart from its input alone. That is the whole point: it is what lets a policy trained in
the sim be flown for real, and a demonstration recorded for real be trained on offline.

Yaw rate is differentiated from the 10 Hz yaw reading rather than measured. The
hardware/ stack hit a real bug doing this (README 3.7: two telemetry samples a
floating-point epsilon apart divided a yaw difference by ~0, the rate reached ~1e10 and
saturated the controller), so the same two guards are applied here: ignore intervals
under MIN_DT, and clamp the result.
"""
from collections import deque
from typing import Optional

import numpy as np

from . import spec
from .ring_detect import RingObs

MIN_DT = 0.02           # s; below this, reuse the previous yaw rate instead of dividing
MAX_YAW_RATE = 360.0    # deg/s, clamp
MAX_PUSH_DT = 0.5       # s; a telemetry gap longer than this is not integrated as distance
VGX_TO_MPS = 0.01       # Tello vgx -> m/s. UNVERIFIED unit: the sim reports cm/s, the same
                        # assumption the `vel` field's VEL_NORM makes (config.yaml has dm/s);
                        # calibrate_speed.py's result must be applied to both together.


def push_step(pushed_m: float, vgx: float, dt: float) -> float:
    """One step of the forward odometer: forward speed x dt, never below zero (backing up
    takes distance off, but the drone cannot be "behind" where it last saw the ring)."""
    return max(0.0, pushed_m + float(vgx) * VGX_TO_MPS * dt)


def upgrade_obs(obs, episode=None, t=None) -> np.ndarray:
    """33-wide observations (recorded before the `pushed` field) -> today's spec.OBS_DIM.

    Replays ObsBuilder's odometer over each episode from fields the old observation already
    has: a fresh detection is age == 0, the forward speed is the `vel` field's vgx, and the
    time step comes from `t` (or spec.CTRL_DT). Wider-by-zero input passes through unchanged."""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] == spec.OBS_DIM:
        return obs
    if obs.shape[1] != spec.LEGACY_OBS_DIM:
        raise ValueError(f"cannot upgrade observations of width {obs.shape[1]}")
    s = spec.OBS_SLICES
    old = lambda name: s[name].start if s[name].start < s["pushed"].start else s[name].start - 1
    i_age, i_vgx = old("age"), old("vel")
    ep = np.zeros(len(obs), np.int64) if episode is None else np.asarray(episode)
    tt = np.arange(len(obs)) * spec.CTRL_DT if t is None else np.asarray(t, dtype=float)
    pushed = np.zeros(len(obs), np.float32)
    run = 0.0
    for i in range(len(obs)):
        new_ep = i == 0 or ep[i] != ep[i - 1]
        dt = 0.0 if new_ep else float(np.clip(tt[i] - tt[i - 1], 0.0, MAX_PUSH_DT))
        if new_ep or obs[i, i_age] <= 1e-6:
            run = 0.0
        else:
            run = push_step(run, obs[i, i_vgx] * spec.VEL_NORM, dt)
        pushed[i] = min(run / spec.RANGE_NORM, 1.0)
    return np.insert(obs, s["pushed"].start, pushed, axis=1)


def _num(state: dict, key: str, default: float = 0.0) -> float:
    """Tello state values, defensively. A regular Tello leaves EDU-only keys absent, and
    djitellopy can hand back a str for a malformed packet."""
    v = state.get(key, default)
    if isinstance(v, bool) or v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


def wrap_deg(d: float) -> float:
    """Shortest signed angle, in (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


class ObsBuilder:
    def __init__(self, act_buffer: int = spec.ACT_BUFFER):
        self.act_buffer = int(act_buffer)
        self._acts = deque(maxlen=self.act_buffer)
        self._prev_yaw: Optional[float] = None
        self._prev_t: Optional[float] = None
        self._yaw_rate = 0.0
        self.reset()

    def reset(self):
        """Clear the action history to zeros and drop the yaw-rate state.

        Called at takeoff. The simulator's equivalent buffer is famously NOT cleared
        between episodes (hardware/README.md 3.3), which is a quirk a real flight must
        not inherit: a real flight genuinely starts from rest.
        """
        self._acts.clear()
        for _ in range(self.act_buffer):
            self._acts.append(np.zeros(spec.ACT_DIM, dtype=np.float32))
        self._prev_yaw = None
        self._prev_t = None
        self._yaw_rate = 0.0
        self._last_seen_range = None       # at the last FRESH detection
        self._last_seen_cx = 0.0
        self._last_seen_cy = 0.0
        self._ticks_since_seen = spec.AGE_NORM
        self._pushed_m = 0.0               # forward distance since the last fresh detection
        self._prev_build_t = None

    def push_action(self, action):
        """Record what was actually executed (post-clamp), oldest-first in the buffer."""
        a = np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(-1)[:spec.ACT_DIM])
        if a.shape[0] < spec.ACT_DIM:
            a = np.pad(a, (0, spec.ACT_DIM - a.shape[0]))
        self._acts.append(np.clip(a, -1.0, 1.0))

    def yaw_rate(self, state: dict, t: float) -> float:
        yaw = _num(state, "yaw")
        if self._prev_yaw is not None and self._prev_t is not None:
            dt = t - self._prev_t
            if dt >= MIN_DT:
                rate = wrap_deg(yaw - self._prev_yaw) / dt
                self._yaw_rate = float(np.clip(rate, -MAX_YAW_RATE, MAX_YAW_RATE))
                self._prev_yaw, self._prev_t = yaw, t
        else:
            self._prev_yaw, self._prev_t = yaw, t
        return self._yaw_rate

    def build(self, ring: RingObs, state: dict, t: float) -> np.ndarray:
        o = spec.empty_obs()
        s = spec.OBS_SLICES

        # Remember the last real sighting. This is what keeps the final blind push
        # distinguishable from an initial search; see spec.OBS_FIELDS.
        dt = 0.0 if self._prev_build_t is None else float(np.clip(t - self._prev_build_t, 0.0, MAX_PUSH_DT))
        self._prev_build_t = t
        if ring.fresh:
            self._ticks_since_seen = 0
            self._last_seen_cx, self._last_seen_cy = ring.cx, ring.cy
            if np.isfinite(ring.range_m):
                self._last_seen_range = ring.range_m
            self._pushed_m = 0.0
        else:
            self._ticks_since_seen = min(self._ticks_since_seen + 1, spec.AGE_NORM)
            self._pushed_m = push_step(self._pushed_m, _num(state, "vgx"), dt)

        o[s["found"]] = 1.0 if ring.found else 0.0
        o[s["centre"]] = (ring.cx, ring.cy)
        o[s["scale"]] = ring.scale * spec.SCALE_NORM
        o[s["axis_ratio"]] = ring.axis_ratio
        # An unmeasurable range (no lock, or a degenerate fit) reads as 1.0 = "far", not
        # inf, because inf would poison every weight it touches.
        rng = ring.range_m
        o[s["range"]] = min(rng / spec.RANGE_NORM, 1.0) if np.isfinite(rng) else 1.0
        o[s["age"]] = min(self._ticks_since_seen / float(spec.AGE_NORM), 1.0)
        o[s["last_range"]] = (min(self._last_seen_range / spec.RANGE_NORM, 1.0)
                              if self._last_seen_range is not None else 1.0)
        o[s["last_centre"]] = (self._last_seen_cx, self._last_seen_cy)
        o[s["pushed"]] = min(self._pushed_m / spec.RANGE_NORM, 1.0)

        o[s["height"]] = _num(state, "h") / spec.HEIGHT_NORM
        o[s["vel"]] = (_num(state, "vgx") / spec.VEL_NORM,
                       _num(state, "vgy") / spec.VEL_NORM,
                       _num(state, "vgz") / spec.VEL_NORM)
        o[s["att"]] = (_num(state, "pitch") / spec.ATT_NORM,
                       _num(state, "roll") / spec.ATT_NORM,
                       self.yaw_rate(state, t) / spec.YAWRATE_NORM)

        o[s["actions"]] = np.concatenate(list(self._acts))
        return np.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0)


def ring_from_obs(obs) -> RingObs:
    """Recover the ring geometry from an observation: the inverse of build()'s ring part.

    This is what lets the scripted Expert be driven from an observation, so it is
    interchangeable with a learned policy wherever one is expected (gym_env.rollout,
    demonstration generation, and comparing the two on identical episodes).

    Lossy in one way worth knowing: build() stores range as min(range / RANGE_NORM, 1.0),
    so any true range beyond RANGE_NORM comes back as exactly RANGE_NORM. That is
    harmless here because the cage is 2 m across and the ring is never that far, but it
    means this is not a general-purpose inverse.
    """
    o = np.asarray(obs, dtype=float).reshape(-1)
    s = spec.OBS_SLICES
    found = bool(o[s["found"]][0] > 0.5)
    scale = float(o[s["scale"]][0]) / spec.SCALE_NORM
    rng = float(o[s["range"]][0]) * spec.RANGE_NORM
    return RingObs(
        found=found, fresh=found,
        cx=float(o[s["centre"]][0]), cy=float(o[s["centre"]][1]),
        axis_ratio=float(o[s["axis_ratio"]][0]), scale=scale,
        range_m=rng if found else float("inf"),
        age=int(round(float(o[s["age"]][0]) * spec.AGE_NORM)),
    )
