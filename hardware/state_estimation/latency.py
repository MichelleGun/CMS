"""
Rolling estimate of overhead-camera latency.

Camera frames are stamped at grab(), but the image shows the world slightly
earlier (exposure, USB transfer, decode). A fixed calibrated value is the
baseline; this tracker slowly corrects it online by cross-correlating the
camera's velocity with Tello's own velocity over a window of recent motion,
and reports a jitter figure (std of recent raw estimates) so the filter can
inflate the camera noise at speed:  R_cam += (speed * sigma_latency)^2.

It only updates while the drone is actually moving and the correlation peak is
clear; otherwise it keeps its last value. The result is EMA-smoothed and bounded
to [latency_min, latency_max]; if raw estimates are unstable it falls back to the
calibrated baseline.
"""
from collections import deque

import numpy as np

BASELINE_DT = 0.3        # camera velocity is a difference over this baseline (noise vs lag trade-off)


class LatencyTracker:
    def __init__(self, est_cfg: dict):
        self.baseline = float(est_cfg["cam_latency_s"])
        self.lo = float(est_cfg["latency_min_s"])
        self.hi = float(est_cfg["latency_max_s"])
        self.window = float(est_cfg["latency_window_s"])
        self.ema = float(est_cfg["latency_ema"])
        self.latency = self.baseline
        self._cam_xy = deque()          # (t, x, y)
        self._cam_v = deque()           # (t_center, vx, vy)
        self._tello_v = deque()         # (t, vx, vy)
        self._raw = deque(maxlen=12)
        self._last_update = None

    def add_camera(self, t, x, y):
        self._cam_xy.append((t, x, y))
        while self._cam_xy and t - self._cam_xy[0][0] > self.window + BASELINE_DT:
            self._cam_xy.popleft()
        # velocity over BASELINE_DT, stamped at the centre of the baseline
        for i in range(len(self._cam_xy) - 1, -1, -1):
            t0, x0, y0 = self._cam_xy[i]
            if t - t0 >= BASELINE_DT:
                dt = t - t0
                self._cam_v.append((0.5 * (t + t0), (x - x0) / dt, (y - y0) / dt))
                break
        while self._cam_v and t - self._cam_v[0][0] > self.window:
            self._cam_v.popleft()

    def add_tello_velocity(self, t, vx, vy):
        self._tello_v.append((t, vx, vy))
        while self._tello_v and t - self._tello_v[0][0] > self.window + 1.0:
            self._tello_v.popleft()

    @property
    def cam_velocity(self):
        """Latest camera-derived (vx, vy), or None."""
        return None if not self._cam_v else (self._cam_v[-1][1], self._cam_v[-1][2], self._cam_v[-1][0])

    @property
    def sigma(self) -> float:
        """Jitter of the latency estimate (s). Floor reflects frame-timing uncertainty."""
        if len(self._raw) < 4:
            return 0.02
        return float(max(0.005, np.std(self._raw)))

    def maybe_update(self, now) -> bool:
        if self._last_update is not None and now - self._last_update < 1.0:
            return False
        if len(self._cam_v) < 20 or len(self._tello_v) < 20:
            return False
        cv = np.array(self._cam_v)
        tv = np.array(self._tello_v)
        if np.std(cv[:, 1]) + np.std(cv[:, 2]) < 0.04:        # not moving enough to say anything
            return False
        self._last_update = now
        lags = np.arange(self.lo, self.hi + 1e-9, 0.01)
        # Normalised squared error between camera velocity and Tello velocity shifted by `lag`.
        # (Correlation is nearly flat across lags for smooth motion; the error has a clear minimum.)
        var = float(np.var(cv[:, 1]) + np.var(cv[:, 2]))
        errs = np.full(len(lags), np.inf)
        for i, lag in enumerate(lags):
            ts = cv[:, 0] - lag                                # camera shows the world `lag` earlier
            if ts[0] < tv[0, 0] or ts[-1] > tv[-1, 0]:
                continue
            vx = np.interp(ts, tv[:, 0], tv[:, 1])
            vy = np.interp(ts, tv[:, 0], tv[:, 2])
            errs[i] = (np.mean((cv[:, 1] - vx) ** 2) + np.mean((cv[:, 2] - vy) ** 2)) / var
        if not np.isfinite(errs).any():
            return False
        k = int(np.argmin(errs))
        finite = errs[np.isfinite(errs)]
        if errs[k] > 0.3 or np.median(finite) < 1.3 * errs[k]:
            return False                                        # poor match or no clear minimum: keep last value
        lag_hat = float(lags[k])
        if 0 < k < len(lags) - 1 and np.isfinite(errs[k - 1]) and np.isfinite(errs[k + 1]):
            a, b, c = errs[k - 1], errs[k], errs[k + 1]
            denom = a - 2 * b + c
            if denom > 1e-12:
                lag_hat += float(np.clip(0.5 * (a - c) / denom, -1, 1)) * 0.01   # parabolic refinement
        self._raw.append(lag_hat)
        target = float(np.median(self._raw))
        self.latency = float(np.clip(self.latency + self.ema * (target - self.latency), self.lo, self.hi))
        if self.sigma > 0.06:                                   # unstable -> fall back to calibration
            self.latency = self.baseline
        return True
