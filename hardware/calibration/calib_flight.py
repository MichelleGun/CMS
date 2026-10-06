"""Shared plumbing for the flying calibration scripts: a supervised session plus trial recording."""
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
for p in (HW, HW.parent / "scripts", HW / "control"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibration.analysis import Trial                       # noqa: E402
from common.flight_session import FlightSession             # noqa: E402
from common.logger import FlightLogger                        # noqa: E402
from common.safety import GlobalKillSwitch, Level            # noqa: E402


class RecordingCamera:
    """Wraps a camera and remembers every raw detection, so trials use camera data only."""

    def __init__(self, cam):
        self.cam = cam
        self.dets = []

    def set_height_hint(self, z):
        if hasattr(self.cam, "set_height_hint"):
            self.cam.set_height_hint(z)

    def poll(self, now):
        out = self.cam.poll(now)
        self.dets.extend(out)
        return out

    def close(self):
        if hasattr(self.cam, "close"):
            self.cam.close()


class CalibFlight:
    """Takes off under supervision and runs raw-rc trials. Any safety fault ends the calibration."""

    def __init__(self, cfg, backend, camera, log=print, logger=None, hotkeys=True):
        self.log = log
        self.rec_cam = RecordingCamera(camera)
        # the calibration flights are what establishes calibrated axes, so they must be allowed to take off without it
        self.session = FlightSession(cfg, backend, self.rec_cam, hoop=None, logger=logger or FlightLogger(enabled=False),
                                     require_calibrated_axes=False)
        self.kill = GlobalKillSwitch(self.session.monitor)
        if hotkeys:
            self.kill.start()
        self.aborted = False

    # -- helpers -------------------------------------------------------------------
    def tick(self, rc=(0, 0, 0, 0)):
        info = self.session.tick(raw_rc=rc)
        if info.decision.level >= Level.LAND or self.session.phase in ("landing", "landed", "killed"):
            self.aborted = True
        return info

    def wait_ready(self, secs=10.0):
        n = int(secs * 30)
        for _ in range(n):
            info = self.session.tick()
            if info.est.ready and len(self.session._bat_samples) >= 5:
                return True
        return False

    def takeoff(self, timeout=25.0) -> bool:
        probs = self.session.request_takeoff()
        if probs:
            self.log("[calib] takeoff refused: " + "; ".join(probs))
            return False
        for _ in range(int(timeout * 30)):
            info = self.tick()
            if self.session.phase == "flying":
                return True
            if self.aborted:
                return False
        return False

    def run(self, rc, seconds, name="trial", settle_before=1.2, settle_after=1.5) -> Trial:
        """Hover, apply rc for `seconds`, hover again; record camera-only data and raw telemetry fields."""
        t_buf, f_buf = [], []
        c0 = len(self.rec_cam.dets)

        def step(r):
            info = self.tick(r)
            t_buf.append(info.est.t)
            f_buf.append(dict(info.est.fields))
            return info

        for _ in range(int(settle_before * 30)):
            step((0, 0, 0, 0))
            if self.aborted:
                break
        t_cmd = t_buf[-1] if t_buf else 0.0
        heading = self._camera_heading(c0)
        for _ in range(int(seconds * 30)):
            step(rc)
            if self.aborted:
                break
        t_end = t_buf[-1] if t_buf else t_cmd
        for _ in range(int(settle_after * 30)):
            step((0, 0, 0, 0))
            if self.aborted:
                break
        dets = self.rec_cam.dets[c0:]
        cam = np.array([[d.t, d.x, d.y, d.yaw if d.yaw is not None else np.nan] for d in dets]) if dets else np.zeros((0, 4))
        return Trial(name=name, rc=tuple(rc), t=np.array(t_buf), cam=cam, fields=f_buf,
                     t_cmd=t_cmd, t_end=t_end, heading_cage=heading)

    def _camera_heading(self, c0):
        ys = [d.yaw for d in self.rec_cam.dets[-15:] if d.yaw is not None]
        return float(np.angle(np.mean(np.exp(1j * np.array(ys))))) if ys else 0.0

    def land(self):
        self.session.request_land("calibration_done")
        for _ in range(30 * 20):
            info = self.session.tick()
            if self.session.phase in ("landed", "killed"):
                break

    def close(self):
        self.kill.stop()
        self.session.close()
