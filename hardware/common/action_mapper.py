"""
Sim VEL action -> Tello rc channels.

Sim semantics (BaseRLAviary): action = [dx, dy, dz, s]; the world-frame target
velocity is SPEED_LIMIT * |s| * unit(d) (zero vector when d == 0); yaw is held.

Hardware: rotate that world velocity into the body frame using the cage-frame
yaw estimate, divide by the calibrated rc_per_mps, clamp, and add a damped
P-loop on yaw to hold the cage heading. Body frame is FLU (x forward, y left,
z up); `conventions.rc_signs` maps it to Tello's rc channels (lr positive =
right, fb forward, ud up, yaw clockwise). Those signs are UNVERIFIED defaults
until calibration/calibrate_axes.py has been run.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import SPEED_LIMIT
from .geofence import action_to_velocity
from .smoothing import wrap_deg

# DISCRETE_MOVES lives next to the sim code; importing it needs only numpy + gymnasium.
import sys
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def discrete_to_action(idx: int) -> np.ndarray:
    from discrete_action_wrapper import DISCRETE_MOVES
    return DISCRETE_MOVES[int(idx)].astype(float).copy()


@dataclass
class RcCommand:
    lr: int
    fb: int
    ud: int
    yaw: int

    def as_tuple(self):
        return (self.lr, self.fb, self.ud, self.yaw)

    @staticmethod
    def zero():
        return RcCommand(0, 0, 0, 0)


class ActionMapper:
    def __init__(self, cfg: dict):
        c = cfg["control"]
        self.rc_per_mps = float(c["rc_per_mps"])
        self.rc_max = int(c["rc_max"])
        self.yaw_rc_max = int(c["yaw_rc_max"])
        self.yaw_ref = math.radians(float(c["yaw_ref_deg"]))
        self.yaw_kp = float(c["yaw_kp"])
        self.yaw_kd = float(c["yaw_kd"])
        self.yaw_db = float(c["yaw_deadband_deg"])
        self.speed_limit = float(c["speed_limit"])
        s = cfg["conventions"]["rc_signs"]
        self.signs = (s["lr"], s["fb"], s["ud"], s["yaw"])

    def world_velocity(self, action):
        return action_to_velocity(action, self.speed_limit)

    def yaw_command(self, yaw_rad, yaw_rate_rad_s) -> float:
        """Body-frame CCW rate command (rc units, pre-sign): P on error with damping and deadband."""
        err_deg = wrap_deg(math.degrees(self.yaw_ref - yaw_rad))
        if abs(err_deg) < self.yaw_db:
            err_deg = 0.0
        return self.yaw_kp * err_deg - self.yaw_kd * math.degrees(yaw_rate_rad_s)

    def to_rc(self, action, yaw_rad, yaw_rate_rad_s=0.0, speed_scale=1.0) -> RcCommand:
        """action: (4,) sim action (already geofence-clamped). yaw_rad: cage-frame yaw, CCW positive."""
        v_world = self.world_velocity(action) * float(speed_scale)
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        # world -> body: rotate by -yaw
        vx_b = c * v_world[0] + s * v_world[1]
        vy_b = -s * v_world[0] + c * v_world[1]
        vz_b = v_world[2]
        raw = np.array([vy_b, vx_b, vz_b]) * self.rc_per_mps      # left, forward, up in rc units
        yaw_raw = self.yaw_command(yaw_rad, yaw_rate_rad_s)
        vals = [raw[0] * self.signs[0], raw[1] * self.signs[1], raw[2] * self.signs[2],
                yaw_raw * self.signs[3]]
        lim = [self.rc_max, self.rc_max, self.rc_max, self.yaw_rc_max]
        out = [int(round(max(-l, min(l, v)))) if math.isfinite(v) else 0 for v, l in zip(vals, lim)]
        return RcCommand(*out)

    def from_rc(self, rc: RcCommand, yaw_rad) -> np.ndarray:
        """Inverse (translation part only): the world velocity an rc command asks for. Used in tests."""
        lr, fb, ud, _ = rc.as_tuple()
        vy_b = lr * self.signs[0] / self.rc_per_mps
        vx_b = fb * self.signs[1] / self.rc_per_mps
        vz_b = ud * self.signs[2] / self.rc_per_mps
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        return np.array([c * vx_b - s * vy_b, s * vx_b + c * vy_b, vz_b])
