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
VGX_TO_MPS = 0.01       # Tello vgx -> m/s (kept for upgrade_obs on very old demos only)
# Since 2026-10-13 the policy gets NO speed from telemetry, in the sim or on the drone: the real
# Tello's vgx/vgy/vgz in the cage are +-1 jitter with no direction (28 real flights,
# hardware/privileged/reports/sim_fidelity.md), while the sim reported clean speeds, and the
# policies leaned on them (v4: 93 % passes in sim, 5 % with them blanked). So the `vel` field is
# always 0, and the forward odometer `pushed` is dead-reckoned from the forward command actually
# executed, at the nominal full-stick speed. Both are computed the same way in both worlds.
BLANK_VELOCITY = True
NOMINAL_FWD_MPS = spec.RC_MAX / 125.0      # sim.RC_PER_MPS; the real figure is not measured yet


def push_step(pushed_m: float, vgx: float, dt: float) -> float:
    """One step of the forward odometer from a vgx reading (cm/s). Only upgrade_obs uses it, for
    demos recorded before the command odometer."""
    return max(0.0, pushed_m + float(vgx) * VGX_TO_MPS * dt)


def push_step_cmd(pushed_m: float, fb_action: float, dt: float) -> float:
    """One step of the forward odometer from the executed forward command (-1..1): command x
    nominal speed x dt, never below zero (backing up takes distance off, but the drone cannot be
    "behind" where it last saw the ring)."""
    return max(0.0, pushed_m + float(fb_action) * NOMINAL_FWD_MPS * dt)


PASS_DETECT_MAX_RANGE_M = 0.8   # a pass is only declared after a close sighting ...
PASS_DETECT_MARGIN_M = 0.15     # ... and a blind push this far beyond where the ring was


class PassDetector:
    """Declares a pass from telemetry alone, the same way in the sim and on the real drone:
    after a fresh sighting closer than PASS_DETECT_MAX_RANGE_M, the forward odometer (`pushed`)
    went PASS_DETECT_MARGIN_M beyond that range. Remembers the heading at that moment."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.passes = 0
        self.yaw = None                # heading (deg) at the last detected pass
        self._marked = False           # this blind segment already counted

    def step(self, fresh: bool, last_range, pushed_m: float, yaw: float):
        if fresh:
            self._marked = False
            return
        if (not self._marked and last_range is not None and last_range < PASS_DETECT_MAX_RANGE_M
                and pushed_m > last_range + PASS_DETECT_MARGIN_M):
            self._marked = True
            self.passes += 1
            self.yaw = float(yaw)

    def fields(self, yaw: float):
        """-> ((cos, sin) of yaw minus the heading at the last pass, streak)."""
        if self.yaw is None:
            return (0.0, 0.0), 0.0
        a = np.radians(wrap_deg(float(yaw) - self.yaw))
        return (float(np.cos(a)), float(np.sin(a))), min(self.passes, 4) / 4.0


def upgrade_obs(obs, episode=None, t=None) -> np.ndarray:
    """Older observations (33 wide: before `pushed`; 34: before the heading memory) -> today's
    spec.OBS_DIM.

    Replays ObsBuilder's odometer and pass detector over each episode from fields the old
    observation already has: a fresh detection is age == 0, the forward speed is the `vel`
    field's vgx, the heading is the integrated yaw rate (only differences matter), and the time
    step comes from `t` (or spec.CTRL_DT). Input that is already spec.OBS_DIM wide passes
    through unchanged."""
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] == spec.OBS_DIM:
        return obs
    width = obs.shape[1]
    if width not in spec.LAYOUT_FIELDS:
        raise ValueError(f"cannot upgrade observations of width {width}")
    old = spec.layout_slices(width)
    s = spec.OBS_SLICES
    out = np.zeros((len(obs), spec.OBS_DIM), np.float32)
    for name, sl in old.items():
        out[:, s[name]] = obs[:, sl]
    i_age, i_vgx = old["age"].start, old["vel"].start
    i_last, i_rate = old["last_range"].start, old["att"].start + 2
    ep = np.zeros(len(obs), np.int64) if episode is None else np.asarray(episode)
    tt = np.arange(len(obs)) * spec.CTRL_DT if t is None else np.asarray(t, dtype=float)
    det = PassDetector()
    run = yaw = 0.0
    last_range = None
    for i in range(len(obs)):
        new_ep = i == 0 or ep[i] != ep[i - 1]
        dt_raw = 0.0 if new_ep else float(tt[i] - tt[i - 1])
        dt = float(np.clip(dt_raw, 0.0, MAX_PUSH_DT))
        fresh = obs[i, i_age] <= 1e-6
        if new_ep:
            det.reset()
            run = yaw = 0.0
            last_range = None
        else:
            yaw += float(obs[i, i_rate]) * spec.YAWRATE_NORM * dt_raw
        if "pushed" in old:
            run = float(obs[i, old["pushed"].start]) * spec.RANGE_NORM
        elif new_ep or fresh:
            run = 0.0
        else:
            run = push_step(run, obs[i, i_vgx] * spec.VEL_NORM, dt)
        if fresh:
            last_range = float(obs[i, i_last]) * spec.RANGE_NORM
        det.step(bool(fresh), last_range, run, yaw)
        (hc, hs), streak = det.fields(yaw)
        out[i, s["pushed"]] = min(run / spec.RANGE_NORM, 1.0)
        out[i, s["pass_heading"]] = (hc, hs)
        out[i, s["streak"]] = streak
    return out


def refresh_obs(obs, episode=None, t=None) -> np.ndarray:
    """Recorded observations (any width, any date) -> exactly what today's ObsBuilder would have
    built: today's layout (upgrade_obs), the `vel` field blanked, and `pushed`, the pass heading
    and the streak replayed from the command odometer (the newest executed forward command is in
    the action history of the same row). Idempotent."""
    obs = upgrade_obs(np.asarray(obs, dtype=np.float32), episode, t)
    if obs.ndim != 2:
        return obs
    out = obs.copy()
    s = spec.OBS_SLICES
    if BLANK_VELOCITY:
        out[:, s["vel"]] = 0.0
    i_age, i_last, i_rate = s["age"].start, s["last_range"].start, s["att"].start + 2
    i_fb = s["actions"].stop - spec.ACT_DIM + spec.ACT_FB
    ep = np.zeros(len(obs), np.int64) if episode is None else np.asarray(episode)
    tt = np.arange(len(obs)) * spec.CTRL_DT if t is None else np.asarray(t, dtype=float)
    det = PassDetector()
    run = yaw = 0.0
    last_range = None
    for i in range(len(obs)):
        new_ep = i == 0 or ep[i] != ep[i - 1]
        dt_raw = 0.0 if new_ep else float(tt[i] - tt[i - 1])
        dt = float(np.clip(dt_raw, 0.0, MAX_PUSH_DT))
        fresh = obs[i, i_age] <= 1e-6
        if new_ep:
            det.reset()
            run = yaw = 0.0
            last_range = None
        else:
            yaw += float(obs[i, i_rate]) * spec.YAWRATE_NORM * dt_raw
        if new_ep or fresh:
            run = 0.0
        else:
            run = push_step_cmd(run, obs[i, i_fb], dt)
        if fresh:
            last_range = float(obs[i, i_last]) * spec.RANGE_NORM
        det.step(bool(fresh), last_range, run, yaw)
        (hc, hs), streak = det.fields(yaw)
        out[i, s["pushed"]] = min(run / spec.RANGE_NORM, 1.0)
        out[i, s["pass_heading"]] = (hc, hs)
        out[i, s["streak"]] = streak
    return out


def add_tracker(obs, episode=None, t=None) -> np.ndarray:
    """37-wide observations (as refresh_obs returns them) -> 41 wide, the RingTracker fields
    replayed from columns the observation already has: a fresh detection is age == 0 with the
    centre and range of that row, the heading is the integrated yaw rate, the height is the
    `height` field, and the executed command is the newest one in the action history. A 41-wide
    input gets its tracker columns recomputed the same way."""
    obs = np.asarray(obs, dtype=np.float32)
    base = obs[:, :spec.OBS_DIM]
    s = spec.OBS_SLICES
    ep = np.zeros(len(obs), np.int64) if episode is None else np.asarray(episode)
    tt = np.arange(len(obs)) * spec.CTRL_DT if t is None else np.asarray(t, dtype=float)
    a0 = s["actions"].stop - spec.ACT_DIM
    tr = RingTracker()
    out = np.zeros((len(obs), spec.TRACKER_OBS_DIM - spec.OBS_DIM), np.float32)
    yaw = 0.0
    for i in range(len(obs)):
        new_ep = i == 0 or ep[i] != ep[i - 1]
        dt_raw = 0.0 if new_ep else float(tt[i] - tt[i - 1])
        if new_ep:
            tr.reset()
            yaw = 0.0
        else:
            yaw += float(base[i, s["att"].start + 2]) * spec.YAWRATE_NORM * dt_raw
        fresh = base[i, s["age"].start] <= 1e-6
        # the range column is clipped at RANGE_NORM; the apparent size is not, and the detector's
        # range is exactly ring diameter / (2 tan(HFOV/2) x major/width)
        scale = float(base[i, s["scale"].start]) / spec.SCALE_NORM
        rng = (spec.RING_DIAMETER_M / (2.0 * RingTracker.HALF_TAN * scale) if scale > 1e-6
               else float("inf"))
        cx, cy = (float(v) for v in base[i, s["centre"]])
        tr.step(bool(fresh), cx, cy, rng, yaw,
                float(base[i, s["height"].start]) * spec.HEIGHT_NORM / 100.0,
                float(base[i, a0 + spec.ACT_FB]), float(base[i, a0 + spec.ACT_LR]),
                float(np.clip(dt_raw, 0.0, MAX_PUSH_DT)))
        out[i] = tr.fields()
    return np.concatenate([base, out], axis=1)


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


class RingTracker:
    """Where the ring is when the camera cannot see it: a dead-reckoned belief state.

    Position of the ring centre in the drone's body frame, metres: (forward, right, up).
    * Fresh detection: measured. The detector's range is the forward depth (pinhole: the major
      axis shrinks with depth), and cx, cy are offsets in half-frame-WIDTHS on both axes, so
      right = cx * tan(HFOV/2) * range and up = -cy * tan(HFOV/2) * range (image y points down).
    * Otherwise, each tick: the drone's turn rotates the ring the other way round it (telemetry
      yaw, clockwise-positive), and the drone's own motion is subtracted: forward and sideways
      from the executed command x the nominal full-stick speed (the same odometer as `pushed`),
      vertical from the change in height.
    It drifts while blind (the speed figure is nominal, the drone lags its commands); the `age`
    field tells the policy how long it has been blind, and each fresh sighting resets it.
    """

    HALF_TAN = float(np.tan(np.radians(spec.CAMERA_HFOV_DEG) / 2.0))

    def __init__(self):
        self.reset()

    def reset(self):
        self.p = np.zeros(3)
        self.valid = False
        self._yaw = None
        self._h = None

    def step(self, fresh: bool, cx: float, cy: float, range_m: float, yaw_deg: float,
             height_m: float, fb: float, lr: float, dt: float):
        if self.valid and not fresh:
            if self._yaw is not None:
                d = np.radians(wrap_deg(yaw_deg - self._yaw))
                c, s_ = np.cos(d), np.sin(d)
                f, r = self.p[0], self.p[1]
                self.p[0], self.p[1] = f * c + r * s_, r * c - f * s_
            self.p[0] -= float(fb) * NOMINAL_FWD_MPS * dt
            self.p[1] -= float(lr) * NOMINAL_FWD_MPS * dt
            if self._h is not None:
                self.p[2] -= height_m - self._h
        if fresh and np.isfinite(range_m) and range_m > 0:
            self.p = np.array([range_m, cx * self.HALF_TAN * range_m, -cy * self.HALF_TAN * range_m])
            self.valid = True
        self._yaw, self._h = yaw_deg, height_m

    def fields(self) -> np.ndarray:
        """The spec.TRACKER_FIELDS values: ring_est (3) then est_valid (1)."""
        if not self.valid:
            return np.zeros(4, np.float32)
        est = np.clip(self.p / spec.RANGE_NORM, -spec.TRACKER_CLIP, spec.TRACKER_CLIP)
        return np.append(est, 1.0).astype(np.float32)


class ObsBuilder:
    def __init__(self, act_buffer: int = spec.ACT_BUFFER, width: int = spec.OBS_DIM):
        """width: spec.OBS_DIM (37, the default) or spec.TRACKER_OBS_DIM (41: + RingTracker)."""
        if width not in (spec.OBS_DIM, spec.TRACKER_OBS_DIM):
            raise ValueError(f"ObsBuilder width must be {spec.OBS_DIM} or {spec.TRACKER_OBS_DIM}")
        self.width = int(width)
        self.tracker = RingTracker()
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
        self.passes = PassDetector()       # heading memory for same-side laps
        if hasattr(self, "tracker"):
            self.tracker.reset()

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
            # the newest action in the buffer is the one executed over the interval just ended
            self._pushed_m = push_step_cmd(self._pushed_m, float(self._acts[-1][spec.ACT_FB]), dt)

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
        yaw = _num(state, "yaw")
        self.passes.step(bool(ring.fresh), self._last_seen_range, self._pushed_m, yaw)
        o[s["pass_heading"]], o[s["streak"]] = self.passes.fields(yaw)

        o[s["height"]] = _num(state, "h") / spec.HEIGHT_NORM
        if not BLANK_VELOCITY:
            o[s["vel"]] = (_num(state, "vgx") / spec.VEL_NORM,
                           _num(state, "vgy") / spec.VEL_NORM,
                           _num(state, "vgz") / spec.VEL_NORM)
        o[s["att"]] = (_num(state, "pitch") / spec.ATT_NORM,
                       _num(state, "roll") / spec.ATT_NORM,
                       self.yaw_rate(state, t) / spec.YAWRATE_NORM)

        o[s["actions"]] = np.concatenate(list(self._acts))
        last = self._acts[-1]
        self.tracker.step(bool(ring.fresh), ring.cx, ring.cy, ring.range_m, yaw,
                          _num(state, "h") / 100.0, last[spec.ACT_FB], last[spec.ACT_LR], dt)
        if self.width == spec.TRACKER_OBS_DIM:
            o = np.concatenate([o, self.tracker.fields()])
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
