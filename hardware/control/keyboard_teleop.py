"""
Fly the Tello (or the FakeTello with --dry-run) from the keyboard, optionally recording demonstrations
for behaviour cloning.

    python hardware/control/keyboard_teleop.py --dry-run                       # try everything with no drone
    python hardware/control/keyboard_teleop.py --dry-run --record --hoop 0.4 -0.3 1.0
    python hardware/control/keyboard_teleop.py --record --hoop 0.4 -0.3 1.0    # real drone, recording a demo

Keys (the window must have focus; Ctrl+Shift+X/L/H work from anywhere if pynput is installed):
  T takeoff   L safe land   X or BACKSPACE EMERGENCY motor cut   SPACE hold/hover   ESC land + quit
  Arrow keys: x/y in the cage frame   W/S: up/down   1/2/3: 40/70/100 % speed   R: toggle recording
Yaw is held automatically, exactly as in the simulator, so there are no yaw keys.

Everything goes through FlightSession: geofence, safety monitor, warm-up and landing escalation
cannot be bypassed from here.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_setup import add_common_args, build, confirm_hoop, resolve_hoop   # noqa: E402

from common.flight_session import FlightSession            # noqa: E402
from common.logger import FlightLogger                      # noqa: E402
from common.safety import GlobalKillSwitch                   # noqa: E402
from control.inputs import NullInput, PygameInput            # noqa: E402
from control.loop import drive                               # noqa: E402
from imitation.record_demo import DemoRecorder               # noqa: E402


def main(argv=None, inputs=None, realtime=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--record", action="store_true", help="record (obs, action) demonstrations for imitation learning")
    ap.add_argument("--headless", action="store_true", help="no window (tests/CI); use with --max-seconds")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--yes", action="store_true", help="skip the hoop confirmation prompt")
    a = ap.parse_args(argv)
    if a.speed_scale == 0.5 and "--speed-scale" not in (argv or sys.argv):
        a.speed_scale = 1.0          # teleop: the human sets the speed with keys 1/2/3

    cfg, backend, camera = build(a)
    hoop = resolve_hoop(a, cfg, required=a.record)
    if hoop is not None:
        confirm_hoop(hoop, a)
    logger = FlightLogger("teleop", enabled=not a.no_log)
    session = FlightSession(cfg, backend, camera, hoop=hoop, logger=logger, speed_scale=a.speed_scale)
    rec = DemoRecorder({"source": "keyboard", "dry_run": a.dry_run, "profile": a.profile if a.dry_run else "real",
                        "hoop": None if hoop is None else hoop.tolist(), "speed_scale": a.speed_scale}) if a.record else None

    window = None
    if inputs is None:
        inputs = NullInput() if a.headless else PygameInput("Tello keyboard control")
        window = None if a.headless else inputs
    kill = GlobalKillSwitch(session.monitor)
    if not a.no_hotkeys and not a.headless:
        kill.start()
    rt = (not a.headless) if realtime is None else realtime
    summary = {}
    try:
        summary = drive(session, inputs, recorder=rec, realtime=rt, max_seconds=a.max_seconds,
                        window=window, speed=0.7 if not a.headless else 1.0)
    except KeyboardInterrupt:
        print("\n[teleop] Ctrl+C: landing")
        session.request_land("ctrl_c")
        try:
            drive(session, NullInput(), realtime=rt, max_seconds=0.0)
        except KeyboardInterrupt:
            print("[teleop] second Ctrl+C: leaving. The Tello lands by itself ~15 s after commands stop.")
    finally:
        kill.stop()
        session.close()
        if rec is not None and len(rec):
            path = rec.save()
            print(f"[teleop] saved {len(rec)} demo samples -> {path}")
        elif rec is not None:
            print("[teleop] nothing was recorded (no movement keys while flying)")
        try:
            inputs.close()
            backend.close()
            if hasattr(camera, "close"):
                camera.close()
        except Exception:
            pass
        print(f"[teleop] log: {logger.path}" if logger.path else "[teleop] done")
    return summary, rec, session


if __name__ == "__main__":
    main()
