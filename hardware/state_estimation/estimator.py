"""
StateEstimator: telemetry + camera in, unified state estimate out.

Pure and clock-agnostic: every method takes the time it needs, so it runs the
same way against a real Tello (time.monotonic()) and against FakeTello (a
simulated clock). Threads live in the callers (tello_link, aruco_tracker), not here.

Output is a StateEstimate whose `vector()` is [x, y, z, vx, vy, vz, yaw, pitch, roll].
"""
import math
from typing import List

import numpy as np

from common.smoothing import EMA, HoldTimer, SmoothedSignal, wrap_pi
from .est_types import CameraDetection, StateEstimate, OK, DEGRADED, LOST
from .fusion_kf import FusionKF, YawEstimator
from .latency import LatencyTracker
from .time_sync import TelemetryGate, TelemetrySample


class AxisMonitor:
    """Detects flipped / swapped axes at runtime.

    Compares the world velocity we commanded (smoothed, because the drone lags the
    command) with the velocity the *camera* sees. Sustained negative cosine while
    both are clearly non-zero means the rc / frame conventions are wrong.
    """

    def __init__(self, hold_s=1.0, min_cmd=0.08, min_cam=0.05):
        self.timer = HoldTimer(hold_s)
        self.min_cmd, self.min_cam = min_cmd, min_cam
        self._cmd = np.zeros(2)

    def update(self, now, v_cmd_world, cam_vel_xy) -> bool:
        self._cmd += 0.15 * (np.asarray(v_cmd_world[:2], dtype=float) - self._cmd)
        if cam_vel_xy is None:
            return self.timer.update(False, now)
        cam = np.asarray(cam_vel_xy[:2], dtype=float)
        nc, nv = np.linalg.norm(self._cmd), np.linalg.norm(cam)
        if nc < self.min_cmd or nv < self.min_cam:
            return self.timer.update(False, now)
        cos = float(self._cmd @ cam / (nc * nv))
        return self.timer.update(cos < 0.0, now)


class StateEstimator:
    def __init__(self, cfg: dict, t0: float = 0.0):
        self.cfg = cfg
        e = cfg["estimator"]
        conv = cfg["conventions"]
        self.vel_to_mps = float(conv["vel_to_mps"])
        self.vel_signs = np.array(conv["vel_signs"], dtype=float)
        self.att = conv["att_signs"]
        self.gate = TelemetryGate(delay_s=float(conv["telemetry_delay_s"]), vel_to_mps=self.vel_to_mps)
        self.kf = FusionKF(e, t0)
        self.yaw = YawEstimator(e)
        self.latency = LatencyTracker(e)
        self.height = SmoothedSignal(n=5, alpha=0.3)
        self.axis = AxisMonitor(float(cfg["safety"]["axis_mismatch_hold_s"]))
        self.cam_degraded_s = float(e["cam_degraded_s"])
        self.lost_std = float(e["pos_lost_std"])
        self._roll = self._pitch = 0.0
        self._yaw_rate = EMA(0.3, 0.0)
        self._prev_yaw_t = None
        self._prev_yaw = None
        self._cam_recv = None
        self._fields = {}
        self.t_last_sample = None

    # -- inputs ----------------------------------------------------------------
    def on_raw_telemetry(self, raws: list, t_recv: float) -> List[TelemetrySample]:
        samples = self.gate.process_batch(raws, t_recv)
        self.on_telemetry(samples)
        return samples

    def on_telemetry(self, samples: List[TelemetrySample]):
        for s in samples:
            f = s.f
            self._fields = dict(f)
            self._roll = math.radians(f["roll"]) * self.att["roll"]
            self._pitch = math.radians(f["pitch"]) * self.att["pitch"]
            yaw_t = math.radians(f["yaw"]) * self.att["yaw"]
            self.yaw.update_tello(s.t, yaw_t)
            if self._prev_yaw_t is not None and s.t - self._prev_yaw_t >= 0.02:
                rate = wrap_pi(yaw_t - self._prev_yaw) / (s.t - self._prev_yaw_t)
                self._yaw_rate.update(max(-4.0, min(4.0, rate)))      # rad/s, a Tello cannot yaw faster
                self._prev_yaw_t, self._prev_yaw = s.t, yaw_t
            elif self._prev_yaw_t is None:
                self._prev_yaw_t, self._prev_yaw = s.t, yaw_t

            v_body = self.vel_signs * np.array([f["vgx"], f["vgy"], f["vgz"]]) * self.vel_to_mps   # FLU
            c, sn = math.cos(self.yaw.yaw), math.sin(self.yaw.yaw)
            v_world = np.array([c * v_body[0] - sn * v_body[1], sn * v_body[0] + c * v_body[1], v_body[2]])
            self.kf.update_velocity(s.t, v_world)
            z = f["h"] / 100.0
            self.kf.update_height(s.t, z)
            self.height.update(z)
            self.latency.add_tello_velocity(s.t, v_world[0], v_world[1])
            self.t_last_sample = s.t_recv

    def on_camera(self, det: CameraDetection, now: float):
        lat = self.latency.latency
        t_meas = det.t - lat
        self.latency.add_camera(det.t, det.x, det.y)
        self.kf.update_camera(t_meas, (det.x, det.y), det.quality, self.latency.sigma)
        if det.yaw is not None:
            self.yaw.update_camera(t_meas, det.yaw)
        self._cam_recv = now
        self.latency.maybe_update(now)

    # -- output ------------------------------------------------------------------
    def height_hint(self):
        """Current height estimate, for the camera tracker's perspective correction."""
        return float(self.kf.x[2]) if self.kf.z_initialized else None

    def estimate(self, now: float) -> StateEstimate:
        x, P = self.kf.peek(now)
        std = float(math.sqrt(max(np.linalg.eigvalsh(P[0:2, 0:2]))))
        tele_age = self.gate.age(now)
        cam_age = float("inf") if self._cam_recv is None else now - self._cam_recv
        if not self.kf.xy_initialized or std > self.lost_std:
            pos_h = LOST
        elif cam_age > self.cam_degraded_s:
            pos_h = DEGRADED
        else:
            pos_h = OK
        ready = (self.kf.xy_initialized and self.kf.z_initialized and self.yaw.initialized
                 and tele_age < 0.5)
        return StateEstimate(
            t=now, pos=x[0:3].copy(), vel=x[3:6].copy(),
            roll=self._roll, pitch=self._pitch, yaw=self.yaw.yaw, yaw_rate=float(self._yaw_rate.value or 0.0),
            pos_std=std, pos_health=pos_h, yaw_health=self.yaw.health,
            tele_age=tele_age, cam_age=cam_age, ready=ready,
            z_smooth=self.height.value, fields=dict(self._fields),
        )
