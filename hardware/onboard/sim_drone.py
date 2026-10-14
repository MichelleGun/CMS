"""
A pretend Tello presenting the djitellopy surface that fly_hoop.py uses.

The point is that --dry-run runs the REAL flight loop -- the same tracker, the same
observation builder, the same safety checks, the same landing-in-finally -- against this
instead of hardware. A separate simulation path would prove nothing about the code that
actually flies; this way a dry run exercises it.

It also deliberately reproduces the two Tello behaviours the kit's safety layer depends
on, because getting either wrong would make the dry run pass where a real flight fails:
  * get_current_state() returns a NEW dict each call. TelemetryRecorder detects staleness
    by dict IDENTITY, not content (see its docstring), so returning the same object would
    look like a dead link.
  * height reads 0 on the ground. survey.health(airborne=False) refuses to start unless
    it does, and takeoff then settles near 80 cm by itself rather than where asked.

Frames are returned as RGB, like djitellopy's BackgroundFrameRead, so fly_hoop's
RGB->BGR conversion is exercised rather than bypassed.
"""
import time
from typing import Optional

import numpy as np

from .ring_detect import DetectorParams
from .sim import TAKEOFF_Z, OnboardSim, Ring


class SimFrameRead:
    """Stands in for djitellopy's BackgroundFrameRead."""

    def __init__(self, drone):
        self._drone = drone
        self.stopped = False

    @property
    def frame(self):
        if self.stopped or not self._drone._streaming:
            return None
        return self._drone._rgb_frame()

    def stop(self):
        self.stopped = True


class SimDrone:
    """Duck-typed Tello. Every call is appended to .calls, as the kit's own test fakes do,
    so tests can assert the exact command sequence (notably that land always happens)."""

    is_real = False

    def __init__(self, ring: Optional[Ring] = None, profile="typical", seed=0,
                 detector: Optional[DetectorParams] = None,
                 ring_opening_m: Optional[float] = None,
                 ring_noodle_m: Optional[float] = None,
                 drone_width_m: Optional[float] = None,
                 fail_at: Optional[str] = None,
                 sdk: str = "20"):
        kw = {}
        if ring_opening_m is not None:
            kw["ring_opening_m"] = ring_opening_m
        if ring_noodle_m is not None:
            kw["ring_noodle_m"] = ring_noodle_m
        if drone_width_m is not None:
            kw["drone_width_m"] = drone_width_m
        self.sim = OnboardSim(ring=ring, profile=profile, seed=seed, render=True,
                              detector=detector, **kw)
        self.calls = []
        # Inject a failure: 'takeoff', 'land', 'rc', or 'takeoff_refused' -- the drone
        # answering takeoff with 'error' on the ground, as a real one does.
        self.fail_at = fail_at
        # What `sdk?` answers. 20 is the Tello EDU this flies; it was "30", a RoboMaster
        # TT, which hid that setbitrate (SDK 3.0 only) is refused on the real drone.
        self.sdk = sdk
        self._streaming = False
        self._airborne = False
        self._ground_pos = None
        self._last_frame = None

    # -- lifecycle --------------------------------------------------------------------
    def _maybe_fail(self, what):
        if self.fail_at == what:
            raise RuntimeError(f"simulated {what} failure")
        if (self.fail_at == "takeoff_refused" and what in ("takeoff", "land")
                and not self._airborne):
            # djitellopy's wording; a grounded Tello refuses land as well as takeoff.
            raise RuntimeError(f"Command '{what}' was unsuccessful for 2 tries. "
                               "Latest response: 'error'")

    def connect(self, wait_for_state: bool = True):
        self.calls.append("connect")
        # Park the drone on the floor; takeoff puts it back at the sim's start pose.
        self._ground_pos = self.sim.pos.copy()
        self.sim.pos[2] = 0.0

    def end(self):
        self.calls.append("end")

    def set_speed(self, cm_s):
        self.calls.append(("speed", int(cm_s)))

    def set_video_bitrate(self, mbps):
        self.calls.append(("bitrate", int(mbps)))
        if not self.sdk.startswith("3"):
            raise RuntimeError("Command 'setbitrate 1' was unsuccessful for 2 tries. "
                               "Latest response: 'unknown command: setbitrate'")

    def streamon(self):
        self.calls.append("streamon")
        self._streaming = True

    def streamoff(self):
        self.calls.append("streamoff")
        self._streaming = False

    def get_frame_read(self, **kwargs):
        self.calls.append("get_frame_read")
        return SimFrameRead(self)

    # -- flight -----------------------------------------------------------------------
    def takeoff(self):
        self.calls.append("takeoff")
        self._maybe_fail("takeoff")
        # A real Tello climbs to about 0.8 m on its own, wherever it was told to go.
        if self._ground_pos is not None:
            self.sim.pos = self._ground_pos.copy()
        self.sim.pos[2] = TAKEOFF_Z
        self.sim._prev_side = self.sim._plane_side()
        self._airborne = True

    def land(self):
        self.calls.append("land")
        self._maybe_fail("land")
        self._airborne = False
        self.sim.pos[2] = 0.0

    def emergency(self):
        self.calls.append("emergency")
        self._airborne = False

    def send_command_without_return(self, cmd):
        self.calls.append(("cmd", cmd))
        if cmd == "emergency":
            self._airborne = False

    def send_rc_control(self, lr, fb, ud, yaw):
        self._maybe_fail("rc")
        self.calls.append(("rc", (int(lr), int(fb), int(ud), int(yaw))))
        if not self._airborne:
            return
        from . import spec
        self.sim.step([lr / spec.RC_MAX, fb / spec.RC_MAX,
                       ud / spec.RC_MAX, yaw / spec.RC_MAX])

    # -- telemetry / video ------------------------------------------------------------
    def get_current_state(self):
        s = self.sim.state()
        if not self._airborne:
            s["h"] = 0
            s["tof"] = 10          # a grounded Tello still reports a small range
            s["vgx"] = s["vgy"] = s["vgz"] = 0
        s["_t"] = time.monotonic()     # forces a distinct dict every call
        return s                        # a NEW dict each time, as djitellopy does

    def query_sdk_version(self):
        self.calls.append("sdk?")
        return self.sdk

    def get_battery(self):
        return int(self.sim.battery)

    def _rgb_frame(self):
        import cv2
        self._last_frame = cv2.cvtColor(self.sim.frame(), cv2.COLOR_BGR2RGB)
        return self._last_frame

    # -- for assertions ---------------------------------------------------------------
    @property
    def outcome(self):
        return self.sim.outcome

    def rc_calls(self):
        return [c[1] for c in self.calls if isinstance(c, tuple) and c[0] == "rc"]
