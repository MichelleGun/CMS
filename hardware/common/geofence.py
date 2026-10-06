"""
Geofence that works on the *action*, not the velocity.

The policy was trained with no walls (leaving the cage just truncated the
episode), so a post-hoc velocity clamp would fight it invisibly. Instead we
edit the sim-format action [dx, dy, dz, speed_fraction] before it is mapped to
rc, and the edited action is what goes into the observation's action buffer.
So the next observation reflects what actually got commanded.

Inside a wall's soft zone the outward velocity component is tapered linearly
to zero `zero_offset` short of the wall (the drone needs room to coast: command
latency plus slow braking is ~0.15 m at 0.25 m/s); the result is re-expressed as direction + speed fraction.
`hard` is set when the drone is beyond a wall by more than hard_margin, which
the safety monitor treats as a land trigger.
"""
from dataclasses import dataclass

import numpy as np

from .constants import SPEED_LIMIT


@dataclass
class FenceResult:
    action: np.ndarray    # (4,) edited action
    clamped: bool         # action was changed by the fence
    hard: bool            # beyond the wall by more than hard_margin
    zone: str             # 'ok' | 'soft' | 'hard'


def action_to_velocity(action, speed_limit=SPEED_LIMIT):
    """Sim VEL semantics (BaseRLAviary): direction from dims 0-2, speed fraction from |dim 3|."""
    a = np.asarray(action, dtype=float).reshape(-1)[:4]
    d = a[:3]
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.zeros(3)
    return speed_limit * abs(a[3]) * d / n


def velocity_to_action(v, speed_limit=SPEED_LIMIT):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(4)
    return np.concatenate([v / n, [min(1.0, n / speed_limit)]])


class Geofence:
    def __init__(self, fence_cfg: dict, speed_limit=SPEED_LIMIT):
        self.lo = np.array([fence_cfg["x"][0], fence_cfg["y"][0], fence_cfg["z"][0]], dtype=float)
        self.hi = np.array([fence_cfg["x"][1], fence_cfg["y"][1], fence_cfg["z"][1]], dtype=float)
        self.soft = float(fence_cfg["soft_margin"])
        self.zero = float(fence_cfg.get("zero_offset", 0.0))
        self.hard = float(fence_cfg["hard_margin"])
        self.push_gain = float(fence_cfg.get("push_gain", 0.0))
        self.push_max = float(fence_cfg.get("push_max", 0.0))
        self.speed_limit = speed_limit
        self._in_soft = np.zeros(3, dtype=bool)   # hysteresis state per axis

    def apply(self, action, pos) -> FenceResult:
        pos = np.asarray(pos, dtype=float)
        v = action_to_velocity(action, self.speed_limit)
        v_new = v.copy()
        hard = False
        soft_any = False
        for ax in range(3):
            d_hi = self.hi[ax] - pos[ax]        # >0 inside
            d_lo = pos[ax] - self.lo[ax]
            if d_hi < -self.hard or d_lo < -self.hard:
                hard = True
            for dist, outward, inward in ((d_hi, v[ax] > 0, -1.0), (d_lo, v[ax] < 0, 1.0)):
                # hysteresis: enter the soft zone at `soft`, leave it at 1.2 * soft
                thresh = self.soft * (1.2 if self._in_soft[ax] else 1.0)
                if dist < thresh:
                    soft_any = True
                    self._in_soft[ax] = True
                    if outward:
                        scale = max(0.0, min(1.0, (dist - self.zero) / max(self.soft - self.zero, 1e-6)))
                        v_new[ax] = v[ax] * scale
                    if dist < self.zero and self.push_gain > 0:
                        # past the zero-speed line: gently push back in, so slow drift cannot carry us through the wall
                        push = min(self.push_max, self.push_gain * (self.zero - dist))
                        v_new[ax] = v_new[ax] if inward * v_new[ax] >= push else inward * push
                elif dist >= thresh and not (d_hi < thresh or d_lo < thresh):
                    self._in_soft[ax] = False
        out = velocity_to_action(v_new, self.speed_limit)
        # the action only counts as clamped if the commanded motion really changed
        clamped = bool(np.linalg.norm(v_new - v) > 1e-6)
        if not clamped:
            out = np.asarray(action, dtype=float).reshape(-1)[:4].copy()
        zone = "hard" if hard else ("soft" if soft_any else "ok")
        return FenceResult(out, clamped, hard, zone)
