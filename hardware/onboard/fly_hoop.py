"""
Fly the Tello through the ring using only its own front camera.

    # stage 0: no drone at all
    python hardware/onboard/fly_hoop.py --dry-run
    python hardware/onboard/fly_hoop.py --dry-run --record --attempts 10

    # stage 2: in the air, align only -- never translates
    python hardware/onboard/fly_hoop.py --mode hover          # detect only, no motion
    python hardware/onboard/fly_hoop.py --no-forward

    # stage 3: short approach, no blind push
    python hardware/onboard/fly_hoop.py --no-commit --max-travel 0.4

    # stage 4: the whole thing
    python hardware/onboard/fly_hoop.py

    # stage 5: fly a trained policy instead of the scripted expert
    python hardware/onboard/fly_hoop.py --mode policy --model hardware/flights/bc/clone/bc_model.zip

--dry-run runs this exact loop against sim_drone.SimDrone, so what is tested is what flies.

SAFETY. The layers, in the order they can stop the flight, all reusing the Tello-Friend-Kit
primitives that have actually flown this drone:
  * before takeoff -- ensure_takeoff_clearance (cage height), health(airborne=False)
    (battery >= minimum_battery_percent, mainboard under 80 C, on the ground), and a
    first-frame check, because taking off with no video would leave the loop blind.
  * every tick -- health(airborne=True) re-checks battery against the lower in-air
    abort threshold, temperature against 85 C, and height inside 30-120 cm; a frame
    watchdog; an airborne-time budget; and a FORWARD-TRAVEL budget, which substitutes for
    the position feedback this task does not have (there is no overhead camera, so the
    only way to bound where the drone goes is to bound how far it has been told to go).
  * at any time -- land.flag or kill.flag appearing in the flight directory. Flag files,
    not a socket, so a dead dashboard cannot hang the drone and a live one cannot be
    spoofed into launching it. This is the same mechanism the training dashboard already
    uses for its camera toggle.
  * always -- the finally block owns zeroing the rc and landing, and takeoff_attempted is
    set BEFORE drone.takeoff() so a lost acknowledgement still lands.

A failed landing is reported as an error, never as success.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

HW_ROOT = Path(__file__).resolve().parents[1]           # CMS/hardware
if str(HW_ROOT) not in sys.path:
    sys.path.insert(0, str(HW_ROOT))

# Absolute imports throughout, so this file works identically whether it is run as a
# script (python hardware/onboard/fly_hoop.py), run as a module
# (python -m onboard.fly_hoop) or imported by the tests and the dashboard backend.
# flake8: noqa: E402
from onboard import spec
from onboard.expert import DONE, Expert, ExpertConfig
from onboard.kit import (MAX_AIR_TEMPERATURE_C, MAX_START_TEMPERATURE_C, EventLog,
                         FlightError, TelemetryRecorder, assert_ready, connected,
                         create_drone, ensure_takeoff_clearance, load_settings)
from onboard.obs import ObsBuilder
from onboard.ring_detect import DetectorParams, RingTracker

FLIGHTS_DIR = HW_ROOT / "flights"
DETECTOR_FILE = Path(__file__).resolve().parent / "detector.json"

FIRST_FRAME_TIMEOUT = 6.0
FRAME_WATCHDOG = 3.0
CLIMB_TIMEOUT = 8.0        # seconds after takeoff for the height to reach 30 cm
STATUS_HZ = 5.0
CAMERA_HZ = 2.5


# --------------------------------------------------------------------- small helpers

def _atomic_write(path: Path, data: bytes):
    """Write via a temp file and os.replace.

    Wrapped by the caller in try/except OSError: this repository lives inside a OneDrive
    folder, and both OneDrive's sync and the dashboard's own polling can hold a transient
    lock on a file being replaced. Losing a status update must never end a flight.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, obj: dict):
    try:
        _atomic_write(path, json.dumps(obj, indent=2).encode("utf-8"))
    except OSError:
        pass


def load_detector(path: Path = DETECTOR_FILE) -> DetectorParams:
    """Thresholds measured by tune_detector.py, or the (untuned) defaults."""
    if path.is_file():
        try:
            return DetectorParams.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as e:
            print(f"[detector] ignoring {path.name}: {e}")
    return DetectorParams()


class TickLog:
    """One JSON object per control tick, with a 'type' discriminator, so the dashboard can
    tail it the same way telemetry_ws.py tails the trainer's telemetry.jsonl."""

    def __init__(self, path: Path):
        self.f = path.open("w", encoding="utf-8")

    def write(self, kind: str, **fields):
        try:
            self.f.write(json.dumps({"type": kind, **_round(fields)}) + "\n")
            self.f.flush()
        except OSError:
            pass

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass


def _round(d: dict, nd=4):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, nd)
        elif isinstance(v, (list, tuple)):
            out[k] = [round(x, nd) if isinstance(x, float) else x for x in v]
        elif isinstance(v, np.ndarray):
            out[k] = [round(float(x), nd) for x in v.reshape(-1)]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = round(float(v), nd)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------- the flight

class Flight:
    def __init__(self, args, settings, flight_dir: Path):
        self.a = args
        self.settings = settings
        self.dir = flight_dir
        self.detector = load_detector()
        # Ranging uses the OUTER diameter, because the noodle's outer edge is what the
        # camera measures. Confusing it with the opening biases every range by ~12 cm.
        outer = (args.ring_opening or spec.RING_OPENING_M) + 2 * (
            args.ring_noodle if args.ring_noodle is not None else spec.RING_NOODLE_M)
        self.tracker = RingTracker(self.detector, ring_diameter_m=outer,
                                   hfov_deg=args.hfov)
        self.builder = ObsBuilder()
        self.controller, self.cfg = self._build_controller()
        self.recorder = None            # demo recorder, if --record
        self.travel_m = 0.0
        self.max_travel = args.max_travel
        self.outcome = "unknown"

    # -- setup ------------------------------------------------------------------------
    def _build_controller(self):
        """-> (callable(ring, state) -> (action, info), ExpertConfig).

        All three controllers share one signature so the flight loop does not branch on
        which is flying. A policy is driven from the observation rather than the RingObs,
        which is exactly the interface it was trained on.
        """
        a = self.a
        # approach_target_m must stay STRICTLY below commit_range_m or the forward term
        # decays to zero exactly at the commit threshold and the drone asymptotes just
        # outside it (see ExpertConfig.approach_target_m). It is therefore derived from
        # whatever commit range is in force, keeping the tuned gap, rather than pinned to
        # a constant that a --commit-range override would invalidate.
        tuned = ExpertConfig()
        gap = tuned.commit_range_m - tuned.approach_target_m
        cfg = ExpertConfig(
            allow_forward=not a.no_forward and a.mode != "hover",
            allow_commit=not a.no_commit and a.mode != "hover",
            commit_speed_mps=a.commit_speed,
            commit_range_m=a.commit_range,
            approach_target_m=max(a.commit_range - gap, 0.05),
        )
        if a.mode == "hover":
            return (lambda ring, state: (np.zeros(spec.ACT_DIM), None)), cfg
        if a.mode == "policy":
            from onboard.policy import load_policy
            fn = load_policy(a.model, algo=a.algo)
            return (lambda ring, state: (fn(self._last_obs), None)), cfg
        return Expert(cfg), cfg

    # -- control-flag files -----------------------------------------------------------
    def _flag(self, name) -> bool:
        return (self.dir / name).exists()

    # -- health -----------------------------------------------------------------------
    def _check(self, recorder, min_batt, *, airborne) -> dict:
        """Battery, telemetry freshness, temperature and height.

        Deliberately NOT tello_lab.survey.health, although it checks the same four
        things. Its airborne height window is hardcoded to 30-120 cm, which suits the
        camera survey it was written for -- that hovers over its own takeoff spot. This
        task has to fly at the RING's height, and a ring hung at 1.15 m plus normal
        vertical overshoot exceeds 120 cm, which aborted every such flight until the
        window was separated out and made configurable.

        What is still reused is the part that carries the real value: the kit's
        assert_ready, which is the proven battery-and-telemetry-staleness gate, plus its
        two temperature limits. Only the height window is ours.
        """
        assert_ready(recorder, min_batt)
        state, _age = recorder.snapshot()

        temp = state.get("temph")
        if isinstance(temp, bool) or not isinstance(temp, int):
            raise FlightError("No valid mainboard temperature telemetry")
        limit = MAX_AIR_TEMPERATURE_C if airborne else MAX_START_TEMPERATURE_C
        if temp > limit:
            raise FlightError(f"Mainboard reports {temp} C; this flight limits it to "
                              f"{limit} C")

        h = state.get("h")
        if not airborne:
            if h != 0:
                raise FlightError("The drone must start on the ground")
            return state
        lo, hi = self.a.min_height, self.a.max_height
        if isinstance(h, bool) or not isinstance(h, int) or not lo <= h <= hi:
            raise FlightError(f"Airborne height {h} cm is outside this flight's "
                              f"{lo}-{hi} cm range")
        tof = state.get("tof")
        if isinstance(tof, bool) or not isinstance(tof, int) or not 0 < tof < 6553:
            raise FlightError("Downward range telemetry is invalid; stopping the flight")
        return state

    # -- the tick ---------------------------------------------------------------------
    def _safety_overrides(self, action, info):
        """Clamp the controller's request. Anything here also gets logged, because a
        silently trimmed command is how a control bug hides."""
        reasons = []
        a = np.asarray(action, dtype=float).copy()
        if self.a.no_forward or self.a.mode == "hover":
            if abs(a[spec.ACT_FB]) > 1e-6:
                reasons.append("no_forward")
            a[spec.ACT_FB] = 0.0
        if self.max_travel is not None and self.travel_m >= self.max_travel:
            if a[spec.ACT_FB] > 0:
                reasons.append("travel_budget")
            a[spec.ACT_FB] = min(a[spec.ACT_FB], 0.0)
        return np.clip(a, -1.0, 1.0), reasons

    def run(self, drone) -> dict:
        a = self.a
        d = self.dir
        events = EventLog(d / "events.jsonl")
        ticks = TickLog(d / "ticks.jsonl")
        recorder = TelemetryRecorder(drone, d / "telemetry.csv", self.settings.sample_hz)
        reader = None
        stream_on = False
        takeoff_attempted = False
        primary_error = None
        landing_error = None
        result = {"status": "started", "outcome": "unknown", "travel_m": 0.0}

        try:
            with connected(drone):
                try:
                    recorder.start()
                    events.write("connected", host=self.settings.host, mode=a.mode,
                                 dry_run=bool(a.dry_run))
                    self._check(recorder, self.settings.minimum_battery_percent, airborne=False)

                    try:
                        drone.set_video_bitrate(1)      # survey.py's answer to in-flight
                        events.write("video_bitrate", mbps=1)   # video corruption
                    except Exception as e:
                        events.write("video_bitrate_failed", error=str(e))
                    drone.set_speed(self.settings.speed_cm_s)

                    drone.streamon()
                    stream_on = True
                    reader = drone.get_frame_read(with_queue=True, max_queue_len=2)
                    self._await_first_frame(reader, recorder, events)

                    for n in range(a.countdown, 0, -1):
                        print(f"Takeoff in {n}s; Ctrl+C cancels.", flush=True)
                        self._wait(1.0, recorder, self.settings.minimum_battery_percent)
                        if self._flag("land.flag") or self._flag("kill.flag"):
                            raise FlightError("stopped before takeoff by a control flag")

                    events.write("takeoff_start")
                    # Set BEFORE the call: a timeout can arrive after the drone has
                    # actually left the ground, and that still has to be landed.
                    takeoff_attempted = True
                    drone.takeoff()
                    started = time.monotonic()
                    self.builder.reset()
                    self.tracker.reset()
                    events.write("takeoff_done")

                    self._settle(reader, recorder, events)
                    result.update(self._loop(drone, reader, recorder, events, ticks,
                                             started))
                except BaseException as exc:          # includes KeyboardInterrupt
                    primary_error = exc
                    result["status"] = "aborted"
                    result["reason"] = f"{type(exc).__name__}: {exc}"
                    try:
                        events.write("flight_aborted", reason=result["reason"])
                    except OSError:
                        pass                           # logging must never block landing
                    raise
                finally:
                    if takeoff_attempted:
                        print("Requesting landing...", flush=True)
                        # BaseException, not Exception, on both of these. A second Ctrl+C
                        # arriving while the drone is being landed raises
                        # KeyboardInterrupt, which is a BaseException -- so `except
                        # Exception` here would let it propagate out of the zeroing call
                        # and skip drone.land() altogether. Someone mashing Ctrl+C would
                        # then prevent the landing they were asking for. A test injects
                        # exactly that.
                        try:
                            drone.send_rc_control(0, 0, 0, 0)
                        except BaseException:
                            pass
                        try:
                            drone.land()
                            events.write("land_complete")
                            result["landing_acknowledged"] = True
                        except BaseException as exc:
                            landing_error = exc
                            result["landing_acknowledged"] = False
                            print(f"LANDING NOT CONFIRMED: {exc}", flush=True)
                            try:
                                events.write("land_failed", error=str(exc))
                            except OSError:
                                pass
                    if reader is not None:
                        try:
                            reader.stop()
                        except Exception:
                            pass
                    if stream_on:
                        try:
                            drone.streamoff()
                        except Exception:
                            pass
                    try:
                        recorder.close()
                    except Exception:
                        pass
        finally:
            ticks.close()
            try:
                events.close()
            except OSError:
                pass
            result["travel_m"] = round(self.travel_m, 3)
            if self.recorder is not None and len(self.recorder):
                try:
                    p = self.recorder.save(self.dir / "demo.npz")
                    result["demo"] = str(p)
                    print(f"[demo] {len(self.recorder)} samples -> {p}")
                except (ValueError, OSError) as e:
                    print(f"[demo] not saved: {e}")
            write_json(self.dir / "summary.json", result)

        if landing_error is not None and primary_error is None:
            raise FlightError(f"Landing was not confirmed: {landing_error}") from landing_error
        return result

    # -- phases -----------------------------------------------------------------------
    def _wait(self, seconds, recorder, min_batt):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._check(recorder, min_batt, airborne=False)
            time.sleep(min(0.05, max(0.0, end - time.monotonic())))

    def _await_first_frame(self, reader, recorder, events):
        end = time.monotonic() + FIRST_FRAME_TIMEOUT
        while time.monotonic() < end:
            if reader.frame is not None:
                events.write("first_frame")
                return
            self._check(recorder, self.settings.minimum_battery_percent, airborne=False)
            time.sleep(0.05)
        raise FlightError(
            f"No camera frame within {FIRST_FRAME_TIMEOUT:.0f} s. Refusing to take off "
            "blind: the whole flight depends on the video feed.")

    def _settle(self, reader, recorder, events):
        """Wait out the climb, then hover while the video settles.

        The climb has to be waited for BEFORE the airborne health envelope is enforced.
        TelemetryRecorder samples in the background at sample_hz (5 Hz by default), so for
        up to 200 ms after takeoff its latest snapshot is still the pre-takeoff packet
        with h = 0 -- and survey.health(airborne=True) rejects any height outside
        30-120 cm, so checking immediately aborts every flight on stale data.

        Then frames are flushed rather than trusted: survey.py found on real hardware
        that frames straight after a motion command are distorted.
        """
        deadline = time.monotonic() + CLIMB_TIMEOUT
        while time.monotonic() < deadline:
            state, _ = recorder.snapshot()
            h = state.get("h")
            if isinstance(h, (int, float)) and not isinstance(h, bool) and h >= 30:
                break
            _ = reader.frame
            time.sleep(0.05)
        else:
            raise FlightError(
                f"height never reached 30 cm within {CLIMB_TIMEOUT:.0f} s of takeoff; "
                "the drone may not have left the ground")
        events.write("climbed", height_cm=recorder.snapshot()[0].get("h"))

        end = time.monotonic() + self.a.settle
        while time.monotonic() < end:
            self._check(recorder, self.settings.abort_battery_percent, airborne=True)
            _ = reader.frame
            time.sleep(0.03)
        events.write("settled", seconds=self.a.settle)

    def _loop(self, drone, reader, recorder, events, ticks, started):
        a = self.a
        res = {"status": "flown", "outcome": "no_result"}
        last_frame_t = time.monotonic()
        next_tick = time.monotonic()
        last_status = last_camera = 0.0
        tick_i = 0
        self._last_obs = spec.empty_obs()

        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick += spec.CTRL_DT
            if next_tick < now:                      # overran: do not burst to catch up
                next_tick = now + spec.CTRL_DT
            tick_i += 1

            if self._flag("kill.flag"):
                events.write("kill_flag")
                self._emergency(drone)
                res.update(status="killed", outcome="killed")
                return res
            if self._flag("land.flag"):
                events.write("land_flag")
                res.update(status="landed_on_request", outcome="land_requested")
                return res

            airborne_s = now - started
            if airborne_s > a.max_airborne:
                events.write("airborne_budget", seconds=round(airborne_s, 1))
                res.update(status="landed", outcome="airborne_budget")
                return res

            state = self._check(recorder, self.settings.abort_battery_percent, airborne=True)

            rgb = reader.frame
            if rgb is None:
                if now - last_frame_t > FRAME_WATCHDOG:
                    raise FlightError(
                        f"No camera frame for {FRAME_WATCHDOG:.0f} s while airborne")
                continue
            last_frame_t = now
            import cv2
            # djitellopy/PyAV produce RGB; the detector and OpenCV want BGR.
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            ring = self.tracker.update(bgr)
            obs = self.builder.build(ring, state, now)
            self._last_obs = obs

            action, info = self.controller(ring, state)
            action = np.asarray(action, dtype=float).reshape(-1)[:spec.ACT_DIM]
            action, trimmed = self._safety_overrides(action, info)
            rc = spec.action_to_rc(action, spec.RC_MAX)
            drone.send_rc_control(*rc)
            self.builder.push_action(action)

            # No position feedback exists, so distance flown is integrated from what was
            # COMMANDED. It over-counts while accelerating and under-counts while
            # coasting, which is why the budget is a safety bound and not a measurement.
            if action[spec.ACT_FB] > 0:
                self.travel_m += action[spec.ACT_FB] * a.commit_speed * spec.CTRL_DT

            phase = getattr(info, "phase", a.mode)
            ticks.write("tick", t=round(airborne_s, 3), phase=phase,
                        found=bool(ring.found), fresh=bool(ring.fresh),
                        cx=ring.cx, cy=ring.cy, range_m=(ring.range_m if
                                                          np.isfinite(ring.range_m) else None),
                        axis_ratio=ring.axis_ratio, scale=ring.scale,
                        clipped=bool(ring.clipped), age=ring.age,
                        action=list(action), rc=rc, travel_m=self.travel_m,
                        trimmed=trimmed, bat=state.get("bat"), h=state.get("h"),
                        temph=state.get("temph"))

            if self.recorder is not None and ring.found:
                self.recorder.add(obs, action, action, airborne_s)

            if now - last_status >= 1.0 / STATUS_HZ:
                last_status = now
                self._write_status(phase, ring, state, action, airborne_s, tick_i,
                                   getattr(info, "reason", ""))
            if self._flag("camera_enabled.flag") and now - last_camera >= 1.0 / CAMERA_HZ:
                last_camera = now
                self._write_camera(bgr, ring, state)

            if a.verbose and tick_i % 5 == 0:
                r = f"{ring.range_m:4.2f}" if np.isfinite(ring.range_m) else " inf"
                print(f"  {airborne_s:5.1f}s {phase:9s} range={r} cx={ring.cx:+.3f} "
                      f"cy={ring.cy:+.3f} ar={ring.axis_ratio:.2f} rc={rc} "
                      f"trav={self.travel_m:.2f}", flush=True)

            if getattr(self.controller, "finished", False):
                ph = getattr(self.controller, "phase", "")
                events.write("controller_finished", phase=ph,
                             reason=getattr(self.controller, "reason", ""))
                res.update(status="landed",
                           outcome="traversed" if ph == DONE else "aborted_by_controller")
                return res

    def _emergency(self, drone):
        for fn in ("emergency",):
            if hasattr(drone, fn):
                try:
                    getattr(drone, fn)()
                    return
                except Exception:
                    pass
        try:
            drone.send_command_without_return("emergency")
        except Exception:
            pass

    # -- outputs ----------------------------------------------------------------------
    def _write_status(self, phase, ring, state, action, airborne_s, tick_i, reason):
        write_json(self.dir / "status.json", {
            "phase": phase, "reason": reason, "mode": self.a.mode,
            "dry_run": bool(self.a.dry_run), "airborne_s": round(airborne_s, 2),
            "ticks": tick_i,
            "ring_found": bool(ring.found), "ring_fresh": bool(ring.fresh),
            "ring_clipped": bool(ring.clipped),
            "cx": round(ring.cx, 4), "cy": round(ring.cy, 4),
            "range_m": (round(ring.range_m, 3) if np.isfinite(ring.range_m) else None),
            "axis_ratio": round(ring.axis_ratio, 3),
            "travel_m": round(self.travel_m, 3),
            "max_travel_m": self.max_travel,
            "action": [round(float(v), 3) for v in action],
            "battery": state.get("bat"), "height_cm": state.get("h"),
            "temp_c": state.get("temph"), "tof_cm": state.get("tof"),
            "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        })

    def _write_camera(self, bgr, ring, state):
        """The annotated frame, written only while the dashboard asks for it.

        Gated on the flag file so that the JPEG encode costs nothing when nobody is
        watching -- the same reason the trainer's CameraCallback is gated.
        """
        try:
            import cv2
            img = bgr.copy()
            if ring.raw is not None:
                c = ring.raw
                centre = tuple(int(round(v)) for v in c["center_px"])
                axes = tuple(max(1, int(round(v / 2))) for v in c["axes_px"])
                colour = (0, 200, 255) if not ring.clipped else (0, 120, 255)
                cv2.ellipse(img, centre, axes, c["angle_deg"], 0, 360, colour, 2)
                cv2.drawMarker(img, centre, colour, cv2.MARKER_CROSS, 14, 2)
            h, w = img.shape[:2]
            cv2.drawMarker(img, (w // 2, h // 2), (200, 200, 200), cv2.MARKER_CROSS, 18, 1)
            rng = f"{ring.range_m:.2f}m" if np.isfinite(ring.range_m) else "--"
            cv2.putText(img, f"{'LOCK' if ring.found else 'no ring'} {rng} "
                             f"bat {state.get('bat')}%", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                _atomic_write(self.dir / "camera_latest.jpg", buf.tobytes())
        except Exception:
            pass          # a preview frame is never worth interrupting a flight for


# --------------------------------------------------------------------- cli

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="fly sim_drone.SimDrone instead of hardware; nothing spins")
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh"], default="typical",
                    help="dry-run only: how badly the pretend drone behaves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["expert", "policy", "hover"], default="expert")
    ap.add_argument("--model", type=Path, default=None, help="--mode policy: a .zip")
    ap.add_argument("--algo", default=None, choices=["ppo", "a2c", "sac", "td3"])

    ap.add_argument("--no-forward", action="store_true",
                    help="stage 2: align with yaw/up-down only, never translate")
    ap.add_argument("--no-commit", action="store_true",
                    help="stage 3: approach but never make the blind push")
    ap.add_argument("--max-travel", type=float, default=None,
                    help="metres of commanded forward travel before forward is cut. This "
                         "is the integral of the FORWARD COMMAND over the whole flight -- "
                         "the approach plus the entire blind commit push -- not the "
                         "displacement from takeoff. Measured through this loop at the "
                         "current gates: 1.24-1.37 m on the typical profile and up to "
                         "1.63 m on harsh. A budget under that truncates the blind push "
                         "-- and it can still LOOK fine, because the clamp fires near "
                         "the end and the drone coasts through anyway, which is exactly "
                         "how it would hide until a slower speed calibration left the "
                         "drone stopped in the ring plane. Use 1.8, or leave it unset")
    ap.add_argument("--max-airborne", type=float, default=40.0)
    ap.add_argument("--min-height", type=int, default=25,
                    help="abort below this height in cm while airborne")
    ap.add_argument("--max-height", type=int, default=160,
                    help="abort above this height in cm. Must clear the ring's centre "
                         "height with room for overshoot, and stay under the cage roof")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="hover seconds after takeoff before the controller engages")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--attempts", type=int, default=1,
                    help="dry-run only: repeat the flight this many times")

    ap.add_argument("--ring-opening", type=float, default=None,
                    help=f"ring opening in m (default {spec.RING_OPENING_M})")
    ap.add_argument("--ring-noodle", type=float, default=None,
                    help=f"noodle thickness in m (default {spec.RING_NOODLE_M})")
    ap.add_argument("--drone-width", type=float, default=None,
                    help=f"aircraft width in m, which decides whether it FITS (default "
                         f"{spec.DRONE_WIDTH_M}, an unguarded Tello; with prop guards "
                         f"use 0.18). Dry-run scoring only")
    ap.add_argument("--hfov", type=float, default=spec.CAMERA_HFOV_DEG)
    ap.add_argument("--commit-range", type=float, default=ExpertConfig().commit_range_m,
                    help="range in m at which the blind push starts (default "
                         f"{ExpertConfig().commit_range_m}). Do not raise this to where "
                         "the ring stops being detectable: the commit decision is made "
                         "from that last measurement, so it needs the ring still whole")
    ap.add_argument("--commit-speed", type=float, default=0.24,
                    help="assumed forward speed at full commit throttle; MEASURE at stage 3")

    ap.add_argument("--record", action="store_true", help="save a demonstration .npz")
    ap.add_argument("--config", type=Path, default=None, help="kit TOML settings file")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    if a.mode == "policy" and a.model is None:
        ap.error("--mode policy needs --model")
    if a.attempts > 1 and not a.dry_run:
        ap.error("--attempts > 1 is dry-run only; fly real attempts one at a time")
    if a.max_travel is not None and a.max_travel <= 0:
        ap.error("--max-travel must be positive")
    if not 0 < a.min_height < a.max_height:
        ap.error("--min-height must be positive and below --max-height")
    return a


def new_flight_dir() -> Path:
    FLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    d = FLIGHTS_DIR / f"tello_{datetime.now():%Y%m%d-%H%M%S}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def main(argv=None):
    a = parse(argv)
    # The kit's own TOML holds the battery thresholds, speed and cage dimensions.
    settings = load_settings(a.config) if a.config else load_settings()

    # The cage check runs before any drone object exists, as the kit does it.
    ensure_takeoff_clearance(settings)

    results = []
    for attempt in range(a.attempts):
        d = new_flight_dir()
        write_json(d / "meta.json", {
            "flight_id": d.name, "mode": a.mode, "dry_run": bool(a.dry_run),
            "profile": a.profile if a.dry_run else None, "pid": os.getpid(),
            "no_forward": bool(a.no_forward), "no_commit": bool(a.no_commit),
            "max_travel_m": a.max_travel, "max_airborne_s": a.max_airborne,
            "ring_opening_m": a.ring_opening or spec.RING_OPENING_M,
            "drone_width_m": a.drone_width or spec.DRONE_WIDTH_M,
            "commit_range_m": a.commit_range, "commit_speed_mps": a.commit_speed,
            "model": str(a.model) if a.model else None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "final_status": None,
        })
        print(f"[flight] {d.name}  mode={a.mode} dry_run={a.dry_run}"
              + (f" profile={a.profile}" if a.dry_run else ""))
        if not a.dry_run:
            print("REAL FLIGHT: props guarded, cage clear, hand on Ctrl+C.")

        flight = Flight(a, settings, d)
        if a.record:
            from imitation.record_demo import DemoRecorder
            flight.recorder = DemoRecorder(
                meta={"source": "onboard_expert" if a.mode == "expert" else a.mode,
                      "dry_run": bool(a.dry_run), "profile": a.profile,
                      "flight_id": d.name},
                obs_dim=spec.OBS_DIM, task=spec.TASK_NAME,
                obs_shape=(spec.OBS_DIM,))

        if a.dry_run:
            from onboard.sim_drone import SimDrone
            drone = SimDrone(profile=a.profile, seed=a.seed + attempt,
                             detector=flight.detector,
                             ring_opening_m=a.ring_opening,
                             ring_noodle_m=a.ring_noodle,
                             drone_width_m=a.drone_width)
        else:
            drone = create_drone(settings)

        status = "crashed"
        try:
            res = flight.run(drone)
            status = res.get("status", "unknown")
            results.append(res)
            extra = ""
            if a.dry_run and hasattr(drone, "outcome"):
                o = drone.outcome
                extra = ("  sim: " + ("PASSED" if o.passed else "struck" if o.struck
                                      else "missed" if o.missed else "no crossing")
                         + (f" radial {o.radial_m:.3f} m" if o.crossed else ""))
                res["sim_outcome"] = ("passed" if o.passed else "struck" if o.struck
                                      else "missed" if o.missed else "no_crossing")
            print(f"[flight] {status}  travel {res.get('travel_m')} m"
                  f"  outcome {res.get('outcome')}{extra}")
        except KeyboardInterrupt:
            status = "interrupted"
            print("Stopped by Ctrl+C; landing was requested.", file=sys.stderr)
            raise
        except FlightError as e:
            status = "refused"
            print(f"REFUSED: {e}", file=sys.stderr)
        except Exception as e:
            status = "crashed"
            print(f"ERROR: {e}", file=sys.stderr)
            raise
        finally:
            meta = {}
            try:
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            meta["final_status"] = status
            meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
            write_json(d / "meta.json", meta)

    if a.dry_run and a.attempts > 1:
        passed = sum(1 for r in results if r.get("sim_outcome") == "passed")
        print(f"\n[dry-run] {passed}/{a.attempts} passed through the ring")
    return results


if __name__ == "__main__":
    main()
