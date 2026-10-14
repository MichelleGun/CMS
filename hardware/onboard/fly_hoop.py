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

    # fly it yourself and record the run (W/S fwd/back, A/D strafe, arrows: up/down and
    # rotate; asked to save after landing)
    python hardware/onboard/fly_hoop.py --mode human --record --max-travel 1.8

--dry-run runs this exact loop against sim_drone.SimDrone, so what is tested is what flies.

SAFETY. The layers, in the order they can stop the flight, all reusing the Tello-Friend-Kit
primitives that have actually flown this drone:
  * before takeoff -- ensure_takeoff_clearance (cage height), health(airborne=False)
    (battery >= minimum_battery_percent, mainboard under 80 C, on the ground), and a
    first-frame check, because taking off with no video would leave the loop blind.
  * every tick -- health(airborne=True) re-checks battery against the lower in-air
    abort threshold, temperature against 85 C, and height inside --min-height..--max-height
    (10-190 cm; the minimum only once the post-takeoff hover and climb are done); a frame
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
import re
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
READER_RESTART_S = 1.0     # no frame this long and the decoder thread dead: reopen the stream
CLIMB_TIMEOUT = 8.0        # seconds after takeoff for the height to reach 30 cm
START_CLIMB_TIMEOUT = 4.0  # policy mode: seconds allowed to climb to --start-height
START_CLIMB_RC = 30        # rc up/down used for that climb (gentle: ~0.3 m/s)
FLOOR_CLIMB = 0.4          # up/down action forced below --floor-height minus 10 cm
STATUS_HZ = 5.0
CAMERA_HZ = 5.0


# --------------------------------------------------------------------- small helpers

def _yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes")


def _count(answer: str) -> Optional[int]:
    """A pass count typed at the prompt: a whole number, or y / n as 1 / 0. None if it is
    neither, so the caller can ask again."""
    text = answer.strip().lower()
    if text in ("y", "yes"):
        return 1
    if text in ("n", "no"):
        return 0
    return int(text) if text.isdigit() else None


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


def _clearml_record(flight_dir: Path, meta: dict, status: str, background: bool = True):
    """Upload a finished flight to ClearML (Tello/flights). Any failure is printed, never raised:
    by now the drone is on the ground and nothing here may turn a landed flight into an error.

    background (the default since 2026-10-14): hand the upload to a DETACHED process
    (scripts/clearml_sync.py record ...) and return at once. The synchronous upload took ~35 s
    per flight, and for all of it this process stayed alive -- holding the drone's UDP port 8889
    and keeping the dashboard's Start button greyed out. The detached process reads the flight's
    files from disk, so it needs nothing from this one; its output goes to clearml_upload.log."""
    try:
        scripts = HW_ROOT.parent / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        import clearml_sync
        if not background:
            if clearml_sync.record("flights", flight_dir, meta, status=status) == "uploaded":
                print(f"[flight] uploaded to ClearML: Tello/flights/{flight_dir.name}")
            return
        if not clearml_sync.enabled():
            return
        import subprocess
        flags = 0
        if os.name == "nt":
            flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                     | subprocess.CREATE_NO_WINDOW)
        log = open(flight_dir / "clearml_upload.log", "w", encoding="utf-8")
        subprocess.Popen([sys.executable, str(scripts / "clearml_sync.py"), "record", "flights",
                          str(flight_dir), "--status", str(status)],
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=flags, close_fds=True, cwd=str(HW_ROOT.parent))
        log.close()
        print(f"[flight] ClearML upload continues in the background "
              f"({flight_dir.name}/clearml_upload.log)")
    except Exception as e:
        print(f"[flight] ClearML upload skipped: {e}", file=sys.stderr)


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


def _height_cm(state: dict) -> Optional[int]:
    """How high the drone is, preferring the downward distance sensor to `h`.

    `h` is fused from a barometer that drifts as the drone warms over a session (the
    `baro` reading climbed from 7 to 16 during one afternoon), so in flight it read -10 cm
    with the drone hovering at 75 cm, and flights were aborted as "never left the ground".
    `tof` measures the floor directly; it is 6553 when it has no valid reading, and then
    `h` is all there is.
    """
    tof = state.get("tof")
    if isinstance(tof, int) and not isinstance(tof, bool) and 0 < tof < 6553:
        return tof
    h = state.get("h")
    return h if isinstance(h, int) and not isinstance(h, bool) else None


def _ground_fields(state: dict) -> dict:
    """The telemetry that explains a takeoff the drone refused, for the event log."""
    return _round({k: state.get(k) for k in ("bat", "temph", "templ", "h", "tof", "baro")})


# --------------------------------------------------------------------- the flight

class _PolicyController:
    """A learned policy in the controller slot. reset() (called at the handover) restarts a
    history policy's memory (onboard/history.py), so it starts from the hover's frame."""

    def __init__(self, fn, last_obs):
        self.fn, self._last_obs = fn, last_obs

    def reset(self):
        r = getattr(self.fn, "reset", None)
        if r is not None:
            r()

    def __call__(self, ring, state):
        return self.fn(self._last_obs()), None


class Flight:
    def __init__(self, args, settings, flight_dir: Path, keys=None):
        self.a = args
        self.settings = settings
        self.dir = flight_dir
        self._keys = keys               # --mode human: key source; None opens the window
        # Post-landing prompts. --save-answer file answers them from the dashboard;
        # tests replace this directly.
        self._ask = (self._file_ask() if getattr(args, "save_answer", "prompt") == "file"
                     else input)
        self._ring_seen = False         # recording starts at the first lock
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
        self._events = None             # set by run(); lets helpers log without threading it
        self._reader = None             # the frame reader, once the stream is on
        self._last_camera = 0.0
        self._camera_failures = 0
        self._last_ring = None          # human mode: keys still drive the sticks between frames
        self._last_obs = np.zeros(self.builder.width, np.float32)
        self._cam_job = None            # the newest frame waiting for the camera thread
        self._cam_cv = None
        self._cam_thread = None
        self._reader_restarts = 0

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
        if a.mode == "human":
            from onboard.human import HumanController, PygameKeys, UdpKeys
            keys = self._keys
            if keys is None:
                keys = (UdpKeys(a.keys_port) if a.keys == "udp" else PygameKeys())
            return HumanController(keys, gain=a.human_gain, yaw_gain=a.human_yaw_gain), cfg
        if a.mode == "policy":
            from onboard.policy import load_policy
            fn = load_policy(a.model, algo=a.algo)
            # a ring-tracker policy (41 inputs) needs the builder to append the tracker's fields
            self.builder = ObsBuilder(width=getattr(fn, "obs_width", spec.OBS_DIM))
            return _PolicyController(fn, lambda: self._last_obs), cfg
        return Expert(cfg), cfg

    # -- dashboard save answer ----------------------------------------------------------
    def _file_ask(self, timeout_s: float = 600.0, poll_s: float = 0.2):
        """-> an `input`-like callable that takes its answers from save_answer.json.

        The first prompt announces that a save decision is wanted (an event plus a
        status.json phase the dashboard shows as buttons) and waits for the file
        {"save": bool, "passed": bool}. Both prompts are answered from that one file.
        No answer within `timeout_s` discards the run, the same as answering no.
        Only called after landing, so waiting here holds nothing in the air.
        """
        answer = {}

        def ask(prompt: str) -> str:
            if not answer:
                answer.update(self._wait_for_save_answer(timeout_s, poll_s))
            if prompt.lower().startswith("save"):
                return "y" if answer.get("save") else "n"
            # The dashboard may send a count; its pass / fail buttons send only a bool.
            passes = answer.get("passes")
            if isinstance(passes, int) and not isinstance(passes, bool) and passes >= 0:
                return str(passes)
            return "1" if answer.get("passed") else "0"
        return ask

    def _wait_for_save_answer(self, timeout_s, poll_s) -> dict:
        n = len(self.recorder) if self.recorder is not None else 0
        path = self.dir / "save_answer.json"
        try:
            with (self.dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"utc": datetime.now().isoformat(timespec="seconds"),
                                    "event": "awaiting_save", "samples": n}) + "\n")
        except OSError:
            pass
        status = {}
        try:
            status = json.loads((self.dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        status.update(phase="awaiting_save", reason=f"{n} samples recorded",
                      samples=n, updated_at=datetime.now().isoformat(timespec="milliseconds"))
        write_json(self.dir / "status.json", status)
        print(f"[demo] waiting for the dashboard: save this run ({n} samples)?", flush=True)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if path.is_file():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass               # half-written or locked: read it next poll
            time.sleep(poll_s)
        print("[demo] no answer from the dashboard; discarding")
        return {"save": False, "passed": False}

    # -- control-flag files -----------------------------------------------------------
    def _flag(self, name) -> bool:
        return (self.dir / name).exists()

    # -- health -----------------------------------------------------------------------
    def _check(self, recorder, min_batt, *, airborne, enforce_min=True) -> dict:
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

        enforce_min=False (the post-takeoff hover and climb): only the ceiling applies. A Tello
        often stops its own takeoff low (15 cm was logged), and aborting a flight for being
        low WHILE it is being climbed to the start height was the crash this replaces.
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
        h = _height_cm(state)
        if not enforce_min:
            lo = -10 ** 6
        if h is None or not lo <= h <= hi:
            raise FlightError(f"Airborne height {h} cm is outside this flight's "
                              f"{lo}-{hi} cm range")
        tof = state.get("tof")
        if isinstance(tof, bool) or not isinstance(tof, int) or not 0 < tof < 6553:
            raise FlightError("Downward range telemetry is invalid; stopping the flight")
        return state

    # -- the tick ---------------------------------------------------------------------
    def _safety_overrides(self, action, info, height_cm=None):
        """Clamp the controller's request. Anything here also gets logged, because a
        silently trimmed command is how a control bug hides.

        Soft floor: below --floor-height the controller may not descend, and 10 cm lower it
        is made to climb. A policy that sinks after takeoff used to hit the hard
        --min-height abort and end the flight; now it is held up and keeps flying."""
        reasons = []
        a = np.asarray(action, dtype=float).copy()
        floor = getattr(self.a, "floor_height", 0) or 0
        if height_cm is not None and floor > 0 and height_cm < floor:
            if height_cm < floor - 10 and a[spec.ACT_UD] < FLOOR_CLIMB:
                a[spec.ACT_UD] = FLOOR_CLIMB
                reasons.append("floor_climb")
            elif a[spec.ACT_UD] < 0:
                a[spec.ACT_UD] = 0.0
                reasons.append("floor")
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
        self._events = events
        ticks = TickLog(d / "ticks.jsonl")
        recorder = TelemetryRecorder(drone, d / "telemetry.csv", self.settings.sample_hz)
        reader = None
        stream_on = False
        takeoff_attempted = False
        refused_on_ground = False
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

                    # Opt-in: the extra `sdk?` command is not part of the startup sequence
                    # this was flown with before, and takeoff was refused more often once it
                    # was added (not proven; 4 of 10 against 8 of 56). See --probe-sdk.
                    sdk = self._sdk_version(drone, events) if a.probe_sdk else None
                    result["sdk"] = sdk
                    if sdk is not None and sdk < 30:
                        # setbitrate is an SDK 3.0 (RoboMaster TT) command. A Tello EDU on
                        # SDK 2.0 answers "unknown command", which read as a fault in the log.
                        events.write("video_bitrate_unsupported", sdk=sdk)
                    else:
                        try:
                            drone.set_video_bitrate(1)      # survey.py's answer to in-flight
                            events.write("video_bitrate", mbps=1)   # video corruption
                        except Exception as e:
                            events.write("video_bitrate_failed", error=str(e))
                    drone.set_speed(self.settings.speed_cm_s)

                    try:
                        drone.streamoff()
                    except Exception:
                        pass
                    drone.streamon()
                    stream_on = True
                    reader = drone.get_frame_read(with_queue=True, max_queue_len=2)
                    self._reader = reader
                    self._await_first_frame(reader, recorder, events)

                    for n in range(a.countdown, 0, -1):
                        print(f"Takeoff in {n}s; Ctrl+C cancels.", flush=True)
                        self._wait(1.0, recorder, self.settings.minimum_battery_percent)
                        if (self._flag("land.flag") or self._flag("kill.flag")
                                or self._controller_stop()):
                            raise FlightError("stopped before takeoff by a control flag")

                    # The ground state goes in the log: when the drone refuses takeoff it
                    # is the only evidence of why (a sagging battery, a hot mainboard).
                    state = self._check(recorder, self.settings.minimum_battery_percent,
                                        airborne=False)
                    events.write("takeoff_start", **_ground_fields(state))
                    # Set BEFORE the call: a timeout can arrive after the drone has
                    # actually left the ground, and that still has to be landed.
                    takeoff_attempted = True
                    try:
                        drone.takeoff()
                    except BaseException as exc:
                        # An explicit 'error' reply with the drone still reading h == 0 is a
                        # refusal on the ground, not a lost acknowledgement. It is still
                        # landed below; this only stops the log blaming the landing.
                        st, _ = recorder.snapshot()
                        refused_on_ground = "'error'" in str(exc) and st.get("h") == 0
                        if refused_on_ground:
                            print(f"\nThe drone REFUSED takeoff (battery {st.get('bat')}%, "
                                  f"mainboard {st.get('temph')} C). This is the drone, not "
                                  "the camera. Power-cycle it, let it cool a few minutes, "
                                  "and try a takeoff from the Tello app.", flush=True)
                        try:
                            events.write("takeoff_failed", error=str(exc),
                                         refused_on_ground=refused_on_ground,
                                         **_ground_fields(st))
                        except OSError:
                            pass
                        raise
                    started = time.monotonic()
                    self.builder.reset()
                    self.tracker.reset()
                    events.write("takeoff_done")

                    # Ground start, in this order: (connected, camera on, first frame: above)
                    # -> takeoff -> HOVER (rc 0) -> climb to --start-height -> only then the
                    # controller (policy, keys or expert) gets the sticks. Its first action is
                    # always taken from a stable hover at the height it was trained / meant to
                    # start at; the low-height abort only applies from then on.
                    self._settle(reader, recorder, events)
                    if a.start_height > 0 and a.mode != "hover":
                        self._climb_to_start(drone, reader, recorder, events)
                    self._handover(drone, events)
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
                            # Re-read the height: only a drone that refused takeoff AND
                            # still reads as grounded has a land refusal that means nothing.
                            st, _ = recorder.snapshot()
                            if refused_on_ground and st.get("h") == 0:
                                result["landing_needed"] = False
                                print("The drone refused takeoff and is still on the ground; "
                                      "the land command was refused too, as expected.",
                                      flush=True)
                                try:
                                    events.write("land_refused_on_ground", error=str(exc))
                                except OSError:
                                    pass
                            else:
                                landing_error = exc
                                result["landing_acknowledged"] = False
                                print(f"LANDING NOT CONFIRMED: {exc}", flush=True)
                                try:
                                    events.write("land_failed", error=str(exc))
                                except OSError:
                                    pass
                    self._camera_close()
                    reader = self._reader or reader      # a restarted reader replaces the first
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
            if self._camera_failures:
                result["camera_write_failures"] = self._camera_failures
            close = getattr(self.controller, "close", None)
            if close is not None:
                try:
                    close()                    # the key window goes before any prompt
                except Exception:
                    pass
            self._save_demo(result, aborted=primary_error is not None)
            write_json(self.dir / "summary.json", result)

        if landing_error is not None and primary_error is None:
            raise FlightError(f"Landing was not confirmed: {landing_error}") from landing_error
        return result

    def _save_demo(self, result, aborted: bool):
        """Write the recorded demonstration.

        A human flight asks first, and only after landing -- the drone is on the ground
        and the motors are off, so there is no hurry. A run that did not end normally
        (error, Ctrl+C, motor cut) is never offered for saving. A pass is demo.npz and a
        miss is demo_failed.npz, so the natural training glob (tello_*/demo.npz) takes
        passes only: cloning a failed traversal teaches the failure.
        """
        rec = self.recorder
        if rec is None:
            return
        if aborted:
            print("[demo] not saved: the flight did not end normally")
            return
        if not len(rec):
            # Recording starts at the first ring lock, so a flight that never saw the
            # ring has nothing to keep. Say so rather than exit as if nothing was asked.
            print("[demo] not saved: the ring was never detected, so nothing was recorded")
            return
        name = "demo.npz"
        if self.a.mode == "human":
            if aborted or result.get("outcome") == "killed":
                print("[demo] not saved: the flight did not end normally")
                return
            try:
                keep = _yes(self._ask(f"Save this run ({len(rec)} samples)? [y/n] "))
                passes = None
                while keep and passes is None:
                    passes = _count(self._ask(
                        "How many times did it go through the ring? (0 if it missed) "))
            except (KeyboardInterrupt, EOFError):
                print("\n[demo] discarded")
                return
            if not keep:
                print("[demo] discarded")
                return
            passed = passes > 0
            rec.meta["passed"] = passed
            rec.meta["passes"] = passes
            result["passes"] = passes
            name = "demo.npz" if passed else "demo_failed.npz"
        try:
            p = rec.save(self.dir / name)
            result["demo"] = str(p)
            print(f"[demo] {len(rec)} samples -> {p}")
        except (ValueError, OSError) as e:
            print(f"[demo] not saved: {e}")

    # -- phases -----------------------------------------------------------------------
    @staticmethod
    def _sdk_version(drone, events) -> Optional[int]:
        """The SDK version the drone reports (20 for a Tello EDU, 30 for a RoboMaster TT),
        logged once, or None if it will not say. A failed query is tolerated, as
        tello_lab's `check` does."""
        try:
            raw = str(drone.query_sdk_version())
        except Exception as exc:
            events.write("sdk_version", sdk=None, error=str(exc))
            return None
        m = re.match(r"\s*(\d+)", raw)
        sdk = int(m.group(1)) if m else None
        events.write("sdk_version", sdk=sdk, raw=raw)
        return sdk

    def _pump(self):
        """Let a human controller keep its window responsive before it is engaged."""
        pump = getattr(self.controller, "pump", None)
        if pump is not None:
            pump()

    def _controller_stop(self) -> bool:
        """A human asked to land or cut the motors (always False for other controllers)."""
        c = self.controller
        return bool(getattr(c, "kill_requested", False)) or (
            self.a.mode == "human" and bool(getattr(c, "finished", False)))

    def _wait(self, seconds, recorder, min_batt):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            state = self._check(recorder, min_batt, airborne=False)
            self._pump()
            self._preview(state)
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
            h = _height_cm(state)
            if h is not None and h >= 15:
                break
            if not self._preview(state):
                _ = reader.frame
            self._pump()
            time.sleep(0.05)
        else:
            raise FlightError(
                f"neither the height reading nor the downward range sensor showed 15 cm "
                f"within {CLIMB_TIMEOUT:.0f} s of takeoff (last: h={state.get('h')}, "
                f"tof={state.get('tof')}); the drone may not have left the ground")
        events.write("climbed", height_cm=_height_cm(recorder.snapshot()[0]))

        end = time.monotonic() + self.a.settle
        self._write_phase("hover", recorder)
        while time.monotonic() < end:
            # the hover: rc 0, only the ceiling enforced (a low takeoff is climbed out of next)
            state = self._check(recorder, self.settings.abort_battery_percent, airborne=True,
                                enforce_min=False)
            if not self._preview(state):
                _ = reader.frame
            self._pump()
            time.sleep(0.03)
        events.write("settled", seconds=self.a.settle)

    def _climb_to_start(self, drone, reader, recorder, events):
        """Every mode but hover (was: policy only): start where the simulator starts, hovering
        at --start-height (the sim's takeoff height, 0.80 m). A Tello's own takeoff often stops
        lower (15 cm was logged), and a controller handed the sticks near the floor either sank
        into the abort or, for a human, started from a height the ring is far above.
        rc up/down with a timeout, never the blocking SDK `up` (it can hang the link)."""
        target = self.a.start_height
        deadline = time.monotonic() + START_CLIMB_TIMEOUT
        self._write_phase("climb", recorder)
        h = _height_cm(recorder.snapshot()[0])
        while h is not None and h < target and time.monotonic() < deadline:
            self._check(recorder, self.settings.abort_battery_percent, airborne=True,
                        enforce_min=False)
            drone.send_rc_control(0, 0, START_CLIMB_RC, 0)
            if not self._preview(recorder.snapshot()[0]):
                _ = reader.frame
            self._pump()
            time.sleep(spec.CTRL_DT)
            h = _height_cm(recorder.snapshot()[0])
        drone.send_rc_control(0, 0, 0, 0)
        events.write("start_height", height_cm=h, target_cm=target,
                     reached=bool(h is not None and h >= target))

    def _handover(self, drone, events):
        """The controller takes the sticks from here: everything that remembers the past is
        started afresh, so nothing from the takeoff and climb (rc commands the controller
        never chose, frames from the ground) leaks into its first decision."""
        drone.send_rc_control(0, 0, 0, 0)
        self.builder.reset()
        self.tracker.reset()
        reset = getattr(self.controller, "reset", None)
        if reset is not None:
            try:
                reset()
            except Exception:
                pass
        events.write("control_handover", mode=self.a.mode)

    def _write_phase(self, phase, recorder):
        """status.json during the pre-control phases, so the dashboard shows hover / climb."""
        try:
            state, _ = recorder.snapshot()
        except Exception:
            state = {}
        write_json(self.dir / "status.json", {
            "phase": phase, "reason": "", "mode": self.a.mode, "ring_found": False,
            "battery": state.get("bat"), "height_cm": state.get("h"), "tof_cm": state.get("tof"),
            "travel_m": round(self.travel_m, 3), "max_travel_m": self.max_travel,
            "updated_at": datetime.now().isoformat(timespec="milliseconds")})

    def _loop(self, drone, reader, recorder, events, ticks, started):
        a = self.a
        res = {"status": "flown", "outcome": "no_result"}
        last_frame_t = time.monotonic()
        next_tick = time.monotonic()
        last_status = 0.0
        tick_i = 0
        self._last_obs = np.zeros(self.builder.width, np.float32)

        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick += spec.CTRL_DT
            if next_tick < now:                      # overran: do not burst to catch up
                next_tick = now + spec.CTRL_DT
            tick_i += 1

            if self._flag("kill.flag") or getattr(self.controller, "kill_requested", False):
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
            fresh_frame = rgb is not None
            if not fresh_frame:
                if now - last_frame_t > FRAME_WATCHDOG:
                    raise FlightError(
                        f"No camera frame for {FRAME_WATCHDOG:.0f} s while airborne")
                if now - last_frame_t > READER_RESTART_S and self._restart_reader(drone, events):
                    reader = self._reader
                # A human flies from the keyboard, not the camera: their keys are read and sent
                # EVERY tick, frame or no frame (the camera arrives at ~10-30 Hz with jitter, and
                # waiting for it was a large part of the key lag). Everything else needs a frame.
                if a.mode == "human" and self._last_ring is not None:
                    ring, obs, bgr = self._last_ring, self._last_obs, None
                else:
                    # The controller is skipped without a frame, and it is the only thing that
                    # reads the keyboard -- so without this, land and kill pressed during a
                    # video stall of up to FRAME_WATCHDOG seconds were never seen.
                    latch = getattr(self.controller, "latch", None)
                    if latch is not None:
                        latch()
                        done = self._controller_exit(drone, events, res)
                        if done is not None:
                            return done
                    continue
            else:
                last_frame_t = now
                import cv2
                # djitellopy/PyAV produce RGB; the detector and OpenCV want BGR.
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                ring = self.tracker.update(bgr)
                obs = self.builder.build(ring, state, now)
                self._last_obs = obs
                self._last_ring = ring

            if a.mode == "human":
                self.controller.status_extra = (
                    f"travel {self.travel_m:.2f}"
                    + (f"/{self.max_travel:.1f} m" if self.max_travel else " m"))
            action, info = self.controller(ring, state)
            # The check at the top of the tick runs before this poll, so a cut latched
            # just now would otherwise wait a full tick with the motors still driven.
            if getattr(self.controller, "kill_requested", False):
                return self._controller_exit(drone, events, res)
            action = np.asarray(action, dtype=float).reshape(-1)[:spec.ACT_DIM]
            action, trimmed = self._safety_overrides(action, info, _height_cm(state))
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

            # Record from the first lock onward, not only while the ring is visible: the
            # blind push through the hoop is the ring NOT being visible, and a demo
            # without it teaches a policy everything except how to finish. The
            # observation carries where the ring was last seen, so those ticks are
            # interpretable.
            self._ring_seen = self._ring_seen or bool(ring.found)
            if self.recorder is not None and self._ring_seen and fresh_frame:
                self.recorder.add(obs, action, action, airborne_s)

            if now - last_status >= 1.0 / STATUS_HZ:
                last_status = now
                self._write_status(phase, ring, state, action, airborne_s, tick_i,
                                   getattr(info, "reason", ""))
            if bgr is not None and self._camera_due():
                self._camera_async(bgr, ring, state)

            if a.verbose and tick_i % 5 == 0:
                r = f"{ring.range_m:4.2f}" if np.isfinite(ring.range_m) else " inf"
                print(f"  {airborne_s:5.1f}s {phase:9s} range={r} cx={ring.cx:+.3f} "
                      f"cy={ring.cy:+.3f} ar={ring.axis_ratio:.2f} rc={rc} "
                      f"trav={self.travel_m:.2f}", flush=True)

            done = self._controller_exit(drone, events, res)
            if done is not None:
                return done

    def _controller_exit(self, drone, events, res):
        """End the flight if the controller asked to: a motor cut, or it finished (for a
        human, a landing request). None means keep flying."""
        c = self.controller
        if getattr(c, "kill_requested", False):
            events.write("kill_flag")
            self._emergency(drone)
            res.update(status="killed", outcome="killed")
            return res
        if getattr(c, "finished", False):
            ph = getattr(c, "phase", "")
            events.write("controller_finished", phase=ph, reason=getattr(c, "reason", ""))
            # A human decides when the run is over, and whether it went through is
            # their call at the save prompt, not something the controller knows.
            outcome = ("human_landed" if self.a.mode == "human" else
                       "traversed" if ph == DONE else "aborted_by_controller")
            res.update(status="landed", outcome=outcome)
            return res
        return None

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

    def _camera_due(self) -> bool:
        """Rate limit for camera_latest.jpg, shared by every phase that writes it.

        The slot is spent whether or not the flag exists, so the flag is stat()ed at
        CAMERA_HZ rather than on every 15 Hz tick: the flight directory is on OneDrive.
        """
        now = time.monotonic()
        if now - self._last_camera < 1.0 / CAMERA_HZ:
            return False
        self._last_camera = now
        return self._flag("camera_enabled.flag")

    def _preview(self, state) -> bool:
        """Keep the dashboard camera live outside the control loop: the countdown and
        the climb. Frames used to be written only from _loop, so a flight whose takeoff
        failed never produced one and the dashboard pane stayed blank.

        Returns True if it consumed a frame, so callers that flush frames need not.
        """
        if not self.a.preview_before_takeoff or self._reader is None or not self._camera_due():
            return False
        rgb = self._reader.frame
        if rgb is None:
            return False
        import cv2
        self._write_camera(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, state)
        return True

    # The JPEG encode and the write into the (OneDrive-synced) flight folder used to run inside
    # the 15 Hz loop, 5 times a second; a slow write there delayed the next rc command, which
    # is part of what made dashboard flying feel laggy. Now the loop only hands the newest frame
    # to a thread (a one-slot mailbox: an older frame still waiting is simply replaced).
    def _camera_async(self, bgr, ring, state):
        import threading
        if self._cam_thread is None:
            self._cam_cv = threading.Condition()
            self._cam_thread = threading.Thread(target=self._camera_worker, name="camera-writer",
                                                daemon=True)
            self._cam_stop = False
            self._cam_thread.start()
        with self._cam_cv:
            self._cam_job = (bgr, ring, dict(state))
            self._cam_cv.notify()

    def _camera_worker(self):
        while True:
            with self._cam_cv:
                while self._cam_job is None and not self._cam_stop:
                    self._cam_cv.wait(0.5)
                job, self._cam_job = self._cam_job, None
                stop = self._cam_stop
            if job is not None:
                self._write_camera(*job)
            if stop and job is None:
                return

    def _camera_close(self):
        """Write any frame still waiting and stop the thread (end of the flight)."""
        if self._cam_thread is None:
            return
        with self._cam_cv:
            self._cam_stop = True
            self._cam_cv.notify()
        self._cam_thread.join(timeout=2.0)
        self._cam_thread = None

    def _restart_reader(self, drone, events) -> bool:
        """djitellopy decodes the video in a daemon thread that only catches av's ExitError: any
        other H.264 decode error ends that thread silently, the frames stop, and FRAME_WATCHDOG
        lands the flight. If the decoder thread has died (not merely stalled), hover and start a
        new reader, once per flight. -> True if a new reader is in place."""
        reader = self._reader
        worker = getattr(reader, "worker", None)
        if self._reader_restarts >= 1 or worker is None or worker.is_alive():
            return False
        self._reader_restarts += 1
        try:
            drone.send_rc_control(0, 0, 0, 0)       # hover while the stream is reopened
            try:
                reader.stop()
            except Exception:
                pass
            if hasattr(drone, "background_frame_read"):
                drone.background_frame_read = None   # else get_frame_read hands back the dead one
            self._reader = drone.get_frame_read(with_queue=True, max_queue_len=2)
            events.write("video_reader_restarted")
            return True
        except Exception as exc:
            events.write("video_reader_restart_failed", error=f"{type(exc).__name__}: {exc}")
            return False

    def _write_camera(self, bgr, ring, state):
        """The annotated frame, written only while the dashboard asks for it.

        Gated on the flag file so that the JPEG encode costs nothing when nobody is
        watching -- the same reason the trainer's CameraCallback is gated. `ring` is None
        before takeoff, when no detection has run.

        A failure never interrupts the flight, but it is no longer silent: a bare
        `except: pass` here once hid an os.replace that failed on every frame, and the
        dashboard showed one frozen frame for whole flights.
        """
        tmp = self.dir / "camera_latest.jpg.tmp"
        try:
            import cv2
            img = bgr.copy()
            if ring is not None and ring.raw is not None:
                c = ring.raw
                centre = tuple(int(round(v)) for v in c["center_px"])
                axes = tuple(max(1, int(round(v / 2))) for v in c["axes_px"])
                colour = (0, 200, 255) if not ring.clipped else (0, 120, 255)
                cv2.ellipse(img, centre, axes, c["angle_deg"], 0, 360, colour, 2)
                cv2.drawMarker(img, centre, colour, cv2.MARKER_CROSS, 14, 2)
            h, w = img.shape[:2]
            cv2.drawMarker(img, (w // 2, h // 2), (200, 200, 200), cv2.MARKER_CROSS, 18, 1)
            if ring is None:
                label = "on the ground"
            else:
                rng = f"{ring.range_m:.2f}m" if np.isfinite(ring.range_m) else "--"
                label = f"{'LOCK' if ring.found else 'no ring'} {rng}"
            cv2.putText(img, f"{label} bat {state.get('bat')}%", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                final = self.dir / "camera_latest.jpg"
                try:
                    _atomic_write(final, buf.tobytes())
                except PermissionError:
                    # Windows refuses to replace a file another process has open (the
                    # dashboard, or OneDrive syncing it). A plain write is still allowed
                    # to such a file, so use it; a reader may see one torn frame, and the
                    # dashboard keeps the last good one when a frame fails to load.
                    final.write_bytes(buf.tobytes())
                    tmp.unlink(missing_ok=True)
        except Exception as exc:
            self._camera_failures += 1
            if self._camera_failures == 1 and self._events is not None:
                try:
                    self._events.write("camera_write_failing",
                                       error=f"{type(exc).__name__}: {exc}")
                except OSError:
                    pass
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


# --------------------------------------------------------------------- cli

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="fly sim_drone.SimDrone instead of hardware; nothing spins")
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh"], default="typical",
                    help="dry-run only: how badly the pretend drone behaves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["expert", "policy", "hover", "human"],
                    default="expert",
                    help="human: fly it yourself from the keyboard (W/S forward/back, "
                         "A/D strafe, Up/Down height, Left/Right rotate); "
                         "with --record the run is offered for saving after landing")
    ap.add_argument("--human-gain", type=float, default=0.6,
                    help="--mode human: fraction of full deflection a held key commands "
                         "(0.6 = rc 18 of the 30 clamp). The expert is continuous, so a "
                         "full-deflection key would record actions it never produces")
    ap.add_argument("--human-yaw-gain", type=float, default=1.0,
                    help="--mode human: the same, for the rotate keys only (1.0 = rc 30, the "
                         "clamp; 0.6 = rc 18, as the other keys). Rotation in place goes "
                         "nowhere, so it can be quicker than the translation keys")
    ap.add_argument("--keys", choices=["pygame", "udp"], default="pygame",
                    help="--mode human: where held keys come from. udp = the dashboard "
                         "sends them to 127.0.0.1:--keys-port")
    ap.add_argument("--keys-port", type=int, default=0)
    ap.add_argument("--save-answer", choices=["prompt", "file"], default="prompt",
                    help="--mode human --record: answer the post-landing save questions "
                         "in the terminal, or from save_answer.json written by the dashboard")
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
    ap.add_argument("--max-airborne", type=float, default=150.0,
                    help="seconds in the air before landing (default 150, was 40: room for many "
                         "laps; the battery gate still lands a flight that runs low)")
    ap.add_argument("--min-height", type=int, default=10,
                    help="abort below this height in cm while airborne (was 25: a sinking "
                         "policy is now held up by --floor-height instead of aborted)")
    ap.add_argument("--floor-height", type=int, default=40,
                    help="soft floor in cm: below it descent is refused, 10 cm lower a climb "
                         "is forced (logged as trimmed 'floor' / 'floor_climb'); 0 = off")
    ap.add_argument("--start-height", type=int, default=80,
                    help="after takeoff and the hover, climb to this height in cm before the "
                         "controller takes over (every mode but hover; the simulator's start "
                         "height); 0 = off. The low-height abort applies only after this")
    ap.add_argument("--max-height", type=int, default=190,
                    help="abort above this height in cm (default 190, was 160: the hoop hangs "
                         "high). Must clear the ring's centre height with room for overshoot, "
                         "and stay under the cage roof")
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

    ap.add_argument("--record", action="store_true",
                    help="save a demonstration .npz (--mode human records by default and "
                         "asks after landing whether to keep it)")
    ap.add_argument("--no-record", action="store_true",
                    help="--mode human: a practice flight, nothing is recorded or asked")
    ap.add_argument("--camera", action="store_true",
                    help="start with the dashboard camera feed on (camera_enabled.flag)")
    ap.add_argument("--preview-before-takeoff", action="store_true",
                    help="also write dashboard camera frames during the countdown and climb "
                         "(off by default: it adds work before the takeoff command)")
    ap.add_argument("--probe-sdk", action="store_true",
                    help="ask the drone its SDK version (sdk?), log it, and skip setbitrate "
                         "on SDK 2.0 drones (off by default: an extra command before takeoff)")
    ap.add_argument("--config", type=Path, default=None, help="kit TOML settings file")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-clearml", action="store_true",
                    help="don't upload the flight to ClearML after landing (also: CLEARML_OFF=1)")
    a = ap.parse_args(argv)

    if a.mode == "policy" and a.model is None:
        ap.error("--mode policy needs --model")
    if a.keys == "udp" and not 0 < a.keys_port < 65536:
        ap.error("--keys udp needs --keys-port")
    if not 0.0 < a.human_gain <= 1.0:
        ap.error("--human-gain must be in (0, 1]")
    if not 0.0 < a.human_yaw_gain <= 1.0:
        ap.error("--human-yaw-gain must be in (0, 1]")
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


def wants_recording(a) -> bool:
    """A human flight is recorded unless told otherwise: the pass / fail / discard
    question after landing only exists when there is a recording to ask about, and
    without --record it used to be skipped with no word. --no-record opts out."""
    return (a.record or a.mode == "human") and not a.no_record


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
            "keys_port": a.keys_port if a.keys == "udp" else None,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "final_status": None,
        })
        if a.camera:
            (d / "camera_enabled.flag").touch()
        print(f"[flight] {d.name}  mode={a.mode} dry_run={a.dry_run}"
              + (f" profile={a.profile}" if a.dry_run else ""))
        if not a.dry_run:
            print("REAL FLIGHT: props guarded, cage clear, hand on Ctrl+C.")

        flight = Flight(a, settings, d)
        if wants_recording(a):
            from imitation.record_demo import DemoRecorder
            flight.recorder = DemoRecorder(
                meta={"source": "onboard_expert" if a.mode == "expert" else a.mode,
                      "dry_run": bool(a.dry_run), "profile": a.profile,
                      "flight_id": d.name},
                obs_dim=flight.builder.width, task=spec.TASK_NAME,
                obs_shape=(flight.builder.width,))

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
                         + (" (touched)" if o.contact else "")
                         + (f" radial {o.radial_m:.3f} m" if o.done else ""))
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
            # Only now, with the drone on the ground: upload the flight to ClearML from the
            # files above. Never during the flight, so it cannot delay the 15 Hz loop.
            if not a.no_clearml:
                _clearml_record(d, meta, status)

    if a.dry_run and a.attempts > 1:
        passed = sum(1 for r in results if r.get("sim_outcome") == "passed")
        print(f"\n[dry-run] {passed}/{a.attempts} passed through the ring")
    return results


if __name__ == "__main__":
    main()
