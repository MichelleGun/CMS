"""
Kalman filter fusing overhead-camera position with Tello dead-reckoning.

State x = [px, py, pz, vx, vy, vz] in the cage frame, constant-velocity model
with white-noise acceleration (q_acc). Yaw is treated as a *known input* when
Tello's body velocity is rotated into the world frame, which keeps every
measurement model linear. With that choice the filter is exactly the EKF one
would write for this problem (the Jacobians are constant), so there is no
linearisation error to tune around. Plain numpy rather than FilterPy: FilterPy
is unmaintained and fragile with numpy 2.x, and Q/R here map 1:1 onto its API.

Measurements
  velocity  (Tello, ~10 Hz)   H = [0 I]      R = sigma_vel^2 I
  height    (Tello ToF/baro)  H = pz         R = sigma_z^2
  camera xy (sporadic)        H = [I2 0]     R = (sigma_cam / quality)^2 + (speed * sigma_latency)^2
Camera measurements are Mahalanobis-gated; if the gate rejects several in a row
the filter assumes it, not the camera, is wrong and re-acquires.

Out-of-sequence handling: every posterior is snapshotted. A measurement stamped
earlier than the filter time restores the snapshot before it and replays all later
measurements in time order, so late camera frames and bursty telemetry both land
at the right moment.

YawEstimator keeps `yaw_offset = camera_yaw - tello_yaw`, updated slowly and rate-limited,
because Tello yaw is relative to its takeoff heading and drifts, while ArUco yaw is
absolute but noisy.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from common.smoothing import wrap_pi
from .est_types import OK, DEGRADED, LOST


@dataclass
class _Meas:
    t: float
    kind: str               # 'vel' | 'z' | 'cam'
    z: np.ndarray
    R: np.ndarray


_H = {
    "vel": np.hstack([np.zeros((3, 3)), np.eye(3)]),
    "z": np.array([[0, 0, 1, 0, 0, 0]], dtype=float),
    "cam": np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=float),
}


class FusionKF:
    def __init__(self, est_cfg: dict, t0: float = 0.0):
        self.q_acc = float(est_cfg["q_acc"])
        self.sigma_vel = float(est_cfg["sigma_vel"])
        self.sigma_z = float(est_cfg["sigma_z"])
        self.sigma_cam = float(est_cfg["sigma_cam_xy"])
        self.gate = float(est_cfg["cam_gate_chi2"])
        self.history_s = float(est_cfg["history_s"])
        self.x = np.zeros(6)
        self.P = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]) ** 2
        self.t = t0
        self.xy_initialized = False
        self.z_initialized = False
        self._snaps = deque()
        self._log = deque()
        self._cam_rejects = 0
        self.cam_accepted = 0
        self.cam_rejected = 0

    # -- model ------------------------------------------------------------
    def _F_Q(self, dt):
        F = np.eye(6)
        F[0:3, 3:6] = dt * np.eye(3)
        q2 = self.q_acc ** 2
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = q2 * dt ** 4 / 4 * np.eye(3)
        Q[0:3, 3:6] = q2 * dt ** 3 / 2 * np.eye(3)
        Q[3:6, 0:3] = q2 * dt ** 3 / 2 * np.eye(3)
        Q[3:6, 3:6] = q2 * dt ** 2 * np.eye(3)
        return F, Q

    def _predict_to(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def peek(self, t):
        """Predicted (x, P) at time t without mutating the filter."""
        dt = max(0.0, t - self.t)
        F, Q = self._F_Q(dt)
        return F @ self.x, F @ self.P @ F.T + Q

    # -- update -----------------------------------------------------------
    def _apply(self, m: _Meas) -> bool:
        H = _H[m.kind]
        y = m.z - H @ self.x
        S = H @ self.P @ H.T + m.R
        if m.kind == "cam":
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 > self.gate:
                self._cam_rejects += 1
                if self._cam_rejects < 5:
                    self.cam_rejected += 1
                    return False
                # five rejections in a row: trust the camera again, widen our own uncertainty
                self.P[0:2, 0:2] += (0.5 ** 2) * np.eye(2)
                S = H @ self.P @ H.T + m.R
            self._cam_rejects = 0
            self.cam_accepted += 1
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ m.R @ K.T       # Joseph form
        return True

    def _snapshot(self):
        self._snaps.append((self.t, self.x.copy(), self.P.copy()))
        while self._snaps and self.t - self._snaps[0][0] > self.history_s:
            self._snaps.popleft()
        while self._log and self.t - self._log[0].t > self.history_s:
            self._log.popleft()

    def _insert(self, m: _Meas) -> bool:
        if m.t >= self.t - 1e-9:
            self._predict_to(m.t)
            ok = self._apply(m)
            self._log.append(m)
            self._snapshot()
            return ok
        # out of sequence: restore the last posterior at or before m.t and replay
        base = None
        for snap in reversed(self._snaps):
            if snap[0] <= m.t:
                base = snap
                break
        if base is None:
            return False                              # older than the history window
        t_end = self.t
        t_b, x_b, P_b = base
        self.t, self.x, self.P = t_b, x_b.copy(), P_b.copy()
        while self._snaps and self._snaps[-1][0] > t_b:
            self._snaps.pop()
        later = [q for q in self._log if q.t > t_b]
        self._log = deque(q for q in self._log if q.t <= t_b)
        pending = sorted(later + [m], key=lambda q: q.t)
        ok = True
        for q in pending:
            self._predict_to(q.t)
            r = self._apply(q)
            if q is m:
                ok = r
            self._log.append(q)
            self._snapshot()
        self._predict_to(t_end)
        self._snapshot()
        return ok

    # -- public measurement API ------------------------------------------
    def update_velocity(self, t, v_world):
        R = (self.sigma_vel ** 2) * np.eye(3)
        return self._insert(_Meas(t, "vel", np.asarray(v_world, dtype=float), R))

    def update_height(self, t, z):
        if not self.z_initialized:
            self.x[2] = z
            self.P[2, 2] = self.sigma_z ** 2
            self.z_initialized = True
            self.t = max(self.t, t)
        return self._insert(_Meas(t, "z", np.array([z], dtype=float), np.array([[self.sigma_z ** 2]])))

    def update_camera(self, t, xy, quality=1.0, sigma_latency=0.0):
        xy = np.asarray(xy, dtype=float)
        if not self.xy_initialized:
            self.x[0:2] = xy
            self.P[0:2, 0:2] = (self.sigma_cam ** 2) * np.eye(2)
            self.xy_initialized = True
            self.t = max(self.t, t)
            self._snapshot()
            return True
        speed = float(np.linalg.norm(self.x[3:5]))
        r2 = (self.sigma_cam / max(quality, 0.1)) ** 2 + (speed * sigma_latency) ** 2
        return self._insert(_Meas(t, "cam", xy, r2 * np.eye(2)))

    # -- readouts ----------------------------------------------------------
    def pos_std(self, t=None):
        _, P = self.peek(self.t if t is None else t)
        return float(math.sqrt(max(np.linalg.eigvalsh(P[0:2, 0:2]))))


class YawEstimator:
    """Cage-frame yaw = Tello yaw + slowly estimated offset, with health monitoring."""

    def __init__(self, est_cfg: dict):
        self.gain = float(est_cfg["yaw_offset_gain"])
        self.max_rate = math.radians(float(est_cfg["yaw_offset_max_rate_dps"]))
        self.deg_thr = math.radians(float(est_cfg["yaw_degraded_deg"]))
        self.deg_s = float(est_cfg["yaw_degraded_s"])
        self.lost_thr = math.radians(float(est_cfg["yaw_lost_deg"]))
        self.jump_thr = math.radians(float(est_cfg["yaw_jump_deg"]))
        self.offset = 0.0
        self.initialized = False
        self.health = OK
        self.tello_yaw = 0.0
        self._hist = deque(maxlen=64)          # (t, tello_yaw) for camera-time lookup
        self._last_cam_t = None
        self._last_cam_yaw = None
        self._bad_since = None
        self._good_since = None
        self.last_error = 0.0

    @property
    def yaw(self):
        return wrap_pi(self.tello_yaw + self.offset)

    def update_tello(self, t, yaw_rad):
        self.tello_yaw = yaw_rad
        self._hist.append((t, yaw_rad))

    def _tello_yaw_at(self, t):
        if not self._hist:
            return self.tello_yaw
        ts = np.array([h[0] for h in self._hist])
        ys = np.unwrap([h[1] for h in self._hist])
        return float(np.interp(t, ts, ys))

    def update_camera(self, t, yaw_cam):
        if yaw_cam is None or not math.isfinite(yaw_cam):
            return
        ty = self._tello_yaw_at(t)
        if not self.initialized:
            self.offset = wrap_pi(yaw_cam - ty)
            self.initialized = True
            self._last_cam_t, self._last_cam_yaw = t, yaw_cam
            return
        err = wrap_pi(yaw_cam - (ty + self.offset))
        self.last_error = err
        dt = min(0.2, max(1e-3, t - self._last_cam_t)) if self._last_cam_t is not None else 0.05
        jump = (self._last_cam_yaw is not None
                and abs(wrap_pi(yaw_cam - self._last_cam_yaw)) > self.jump_thr and dt < 0.2)
        self._last_cam_t, self._last_cam_yaw = t, yaw_cam

        if abs(err) > self.lost_thr or jump:
            self.health = LOST
            self._bad_since = self._bad_since or t
            self._good_since = None
            return                                           # never learn from a bad reading
        if abs(err) > self.deg_thr:
            self._bad_since = self._bad_since or t
            self._good_since = None
            if t - self._bad_since >= self.deg_s:
                self.health = DEGRADED                       # freeze the offset
            return
        # within tolerance
        self._bad_since = None
        self._good_since = self._good_since or t
        if self.health != OK and t - self._good_since >= self.deg_s:
            self.health = OK
        if self.health == OK:
            step = max(-self.max_rate * dt, min(self.max_rate * dt, self.gain * err * dt))
            self.offset = wrap_pi(self.offset + step)
