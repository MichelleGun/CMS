"""Shared control loop for keyboard_teleop.py and run_policy.py."""
import time
from typing import Callable, Optional

import numpy as np

from common.constants import CTRL_DT
from common.safety import Level

SPEEDS = {"speed1": 0.4, "speed2": 0.7, "speed3": 1.0}
AXES = {"x+": (0, 1), "x-": (0, -1), "y+": (1, 1), "y-": (1, -1), "z+": (2, 1), "z-": (2, -1)}


def movement_action(held, speed) -> np.ndarray:
    """Held movement keys -> sim action [dx, dy, dz, speed_fraction] (world/cage frame)."""
    d = np.zeros(3)
    for k in held:
        if k in AXES:
            ax, sgn = AXES[k]
            d[ax] += sgn
    if np.linalg.norm(d) < 1e-9:
        return np.zeros(4)
    return np.array([*d, speed])


def status_lines(info, session, speed, rec_n, recording, source_label):
    e = info.est
    lvl = info.decision.level
    head = f"{info.phase.upper()}  {source_label}" + ("  *** " + ",".join(info.decision.reasons[:2]) + " ***" if lvl else "")
    return [
        head,
        f"pos  x{e.pos[0]:+.2f} y{e.pos[1]:+.2f} z{e.pos[2]:+.2f} m   +-{e.pos_std:.2f}   yaw {np.degrees(e.yaw):+.0f} deg",
        f"vel  {e.vel[0]:+.2f} {e.vel[1]:+.2f} {e.vel[2]:+.2f} m/s    rc {info.rc.as_tuple()}",
        f"health pos={e.pos_health} yaw={e.yaw_health}  cam_age={min(e.cam_age, 99):.2f}s tele_age={min(e.tele_age, 99):.2f}s",
        f"battery {e.fields.get('bat', '?')} %   temp {e.fields.get('temph', '?')} C   speed {int(speed * 100)} %",
        f"recording: {'ON' if recording else 'off'}  samples {rec_n}",
        "T takeoff  L land  X/BKSP KILL  SPACE hold  ESC land+quit",
        "arrows x/y   W/S up/down   1/2/3 speed   R record",
        "global: Ctrl+Shift+X kill, +L land, +H hold",
    ]


def drive(session, inputs, policy_fn: Optional[Callable] = None, recorder=None, hz=30.0, realtime=True,
          max_seconds: Optional[float] = None, auto_takeoff=False, window=None, speed=0.7,
          log=print) -> dict:
    """Run until landed/killed. Returns a small summary dict."""
    recording = recorder is not None
    armed = False
    t_loop0 = time.monotonic()
    sim_t0 = session.backend.now()
    quit_requested = False
    end_since = None
    n_ticks = 0
    last_takeoff_try = -1e9
    reached = False
    next_wall = time.monotonic()
    while True:
        st = inputs.poll()
        for k in st.pressed:
            if k == "takeoff":
                probs = session.request_takeoff()
                if probs:
                    log("[takeoff refused] " + "; ".join(probs))
            elif k == "land":
                session.request_land()
            elif k == "kill":
                session.request_kill()
            elif k == "hold":
                session.toggle_hold()
            elif k in SPEEDS:
                speed = SPEEDS[k]
            elif k == "record" and recorder is not None:
                recording = not recording
            elif k == "quit":
                quit_requested = True
                session.request_land("quit")
        manual = movement_action(st.held, speed) if st.held else None
        # teleop: no key means "hover" (a zero action). policy mode: no key means "policy decides";
        # any movement key is a human takeover for that tick.
        tick_manual = manual if manual is not None else (np.zeros(4) if policy_fn is None else None)
        info = session.tick(manual_action=tick_manual, policy_fn=policy_fn)
        n_ticks += 1
        reached = reached or info.reached

        if recorder is not None and recording and info.obs is not None and info.source == "manual" \
                and info.decision.level == Level.NONE:
            if tick_manual is not None and np.linalg.norm(tick_manual) > 0:
                armed = True
            if armed:
                recorder.add(info.obs, tick_manual, info.executed_action, info.est.t)

        if auto_takeoff and session.phase == "ground" and not quit_requested:
            now = session.backend.now()
            if now - last_takeoff_try > 1.0:
                last_takeoff_try = now
                if not session.request_takeoff():
                    log("[auto] takeoff")

        if window is not None:
            label = {"policy": "POLICY", "manual": "MANUAL", "hover": "HOVER", "none": ""}.get(info.source, info.source)
            window.show(status_lines(info, session, speed, len(recorder) if recorder else 0, recording, label),
                        alert=info.decision.level >= Level.LAND)

        if session.phase in ("landed", "killed"):
            end_since = end_since or time.monotonic()
            if time.monotonic() - end_since > 0.3 or not realtime:
                break
        elapsed = session.backend.now() - sim_t0
        if max_seconds is not None and elapsed > max_seconds and session.phase in ("ground", "landed", "killed"):
            break
        if max_seconds is not None and elapsed > max_seconds and not quit_requested:
            quit_requested = True
            session.request_land("max_seconds")
        if max_seconds is not None and elapsed > max_seconds + 25:
            log("[loop] giving up waiting for landing")
            break
        if realtime:
            next_wall += 1.0 / hz
            sl = next_wall - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                next_wall = time.monotonic()
    return dict(phase=session.phase, ticks=n_ticks, reached=reached, speed=speed)
