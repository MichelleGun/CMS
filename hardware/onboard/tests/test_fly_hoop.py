"""
The flight loop and its safety layers.

These are the tests that matter most, because this is the code that commands a real
drone. The central one is the kit's own pattern: whatever goes wrong, the drone is
landed. Modelled on tello_lab's test_failure_or_interrupt_attempts_landing, which asserts
the exact tail of the command sequence.

Everything here runs against sim_drone.SimDrone, which is the same object --dry-run
flies, so these exercise the real loop rather than a parallel one.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from onboard import spec
from onboard.fly_hoop import Flight, load_detector, new_flight_dir, parse
from onboard.kit import FlightError, Settings
from onboard.sim_drone import SimDrone

pytestmark = pytest.mark.usefixtures("fresh_telemetry", "fast_clock")


def args(*extra):
    return parse(["--dry-run", "--countdown", "0", "--settle", "0.2", *extra])


def run(tmp_path, *extra, drone=None, settings=None, seed=0):
    a = args(*extra)
    d = tmp_path / "flight"
    d.mkdir(parents=True, exist_ok=True)
    flight = Flight(a, settings or Settings(), d)
    drone = drone or SimDrone(profile=a.profile, seed=seed, detector=flight.detector)
    return flight, drone, d


def events_in(d):
    return [json.loads(line) for line in
            (d / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def event_names(d):
    return [e["event"] for e in events_in(d)]


def assert_landed(drone):
    """The guarantee every failure path has to satisfy.

    Note `land` is NOT the second-to-last call: the finally block lands FIRST and tears
    the video stream down afterwards, which is the right order -- get the drone down,
    then clean up. So this checks ordering rather than position.
    """
    assert "land" in drone.calls, f"never landed: {drone.calls}"
    assert drone.calls[-1] == "end", f"connection not closed: {drone.calls}"
    assert drone.calls.index("land") < drone.calls.index("end")


# --------------------------------------------------------------- the landing guarantee

def test_a_normal_flight_takes_off_and_lands(tmp_path):
    """Also the shape of the command sequence: connect first, end last, exactly one
    takeoff, and land before end."""
    flight, drone, d = run(tmp_path, seed=0)
    res = flight.run(drone)
    assert drone.calls[0] == "connect"
    assert drone.calls.count("takeoff") == 1
    assert_landed(drone)
    assert res["landing_acknowledged"] is True
    assert "land_complete" in event_names(d)


@pytest.mark.parametrize("fail_at", ["takeoff", "rc"])
def test_a_failure_still_lands_the_drone(tmp_path, fail_at):
    """The core guarantee. A lost takeoff acknowledgement is the dangerous case: the
    drone may already be airborne, so takeoff_attempted is set BEFORE the call."""
    flight, drone, d = run(tmp_path)
    drone.fail_at = fail_at
    with pytest.raises(RuntimeError):
        flight.run(drone)
    assert_landed(drone)
    assert "flight_aborted" in event_names(d)


def test_a_keyboard_interrupt_still_lands_the_drone(tmp_path):
    flight, drone, d = run(tmp_path)
    original = drone.send_rc_control
    calls = {"n": 0}

    def interrupt(*a, **k):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt
        return original(*a, **k)
    drone.send_rc_control = interrupt

    with pytest.raises(KeyboardInterrupt):
        flight.run(drone)
    assert_landed(drone)


def test_a_failed_landing_is_reported_as_an_error_not_a_success(tmp_path):
    flight, drone, d = run(tmp_path)
    drone.fail_at = "land"
    with pytest.raises(FlightError, match="Landing was not confirmed"):
        flight.run(drone)
    assert "land_failed" in event_names(d)


def test_the_video_stream_is_always_torn_down(tmp_path):
    flight, drone, _d = run(tmp_path)
    drone.fail_at = "rc"
    with pytest.raises(RuntimeError):
        flight.run(drone)
    assert "streamoff" in drone.calls


# --------------------------------------------------------------- refusals before takeoff

def test_a_flat_battery_refuses_to_take_off(tmp_path):
    flight, drone, _d = run(tmp_path)
    drone.sim.battery = 12.0
    with pytest.raises(FlightError, match="requires at least"):
        flight.run(drone)
    assert "takeoff" not in drone.calls


def test_an_overheating_mainboard_refuses_to_take_off(tmp_path):
    flight, drone, _d = run(tmp_path)
    real_state = drone.get_current_state

    def hot():
        s = real_state()
        s["temph"] = 95
        return s
    drone.get_current_state = hot
    with pytest.raises(FlightError, match="this flight limits it to"):
        flight.run(drone)
    assert "takeoff" not in drone.calls


def test_no_video_refuses_to_take_off(tmp_path):
    """Refusing to fly blind: the whole flight depends on the camera."""
    flight, drone, _d = run(tmp_path)
    drone.get_frame_read = lambda **k: type("Dead", (), {"frame": None,
                                                          "stop": lambda s: None})()
    with pytest.raises(FlightError, match="No camera frame"):
        flight.run(drone)
    assert "takeoff" not in drone.calls


def test_a_cage_too_short_for_the_takeoff_height_is_refused():
    """ensure_takeoff_clearance runs before any drone object exists."""
    from onboard.kit import ensure_takeoff_clearance
    with pytest.raises(FlightError, match="roof clearance"):
        ensure_takeoff_clearance(Settings(cage_height_cm=100, cage_clearance_cm=30))
    ensure_takeoff_clearance(Settings(cage_height_cm=200, cage_clearance_cm=30))


def test_a_control_flag_set_before_takeoff_prevents_it(tmp_path):
    flight, drone, d = run(tmp_path, "--countdown", "2")
    (d / "land.flag").touch()
    with pytest.raises(FlightError, match="control flag"):
        flight.run(drone)
    assert "takeoff" not in drone.calls


# --------------------------------------------------------------- in-flight stops

def test_the_land_flag_ends_the_flight(tmp_path):
    flight, drone, d = run(tmp_path)
    original = drone.send_rc_control
    n = {"i": 0}

    def tick(*a, **k):
        n["i"] += 1
        if n["i"] == 4:
            (d / "land.flag").touch()
        return original(*a, **k)
    drone.send_rc_control = tick

    res = flight.run(drone)
    assert res["outcome"] == "land_requested"
    assert "land_flag" in event_names(d)
    assert_landed(drone)


def test_the_kill_flag_cuts_the_motors(tmp_path):
    flight, drone, d = run(tmp_path)
    original = drone.send_rc_control
    n = {"i": 0}

    def tick(*a, **k):
        n["i"] += 1
        if n["i"] == 4:
            (d / "kill.flag").touch()
        return original(*a, **k)
    drone.send_rc_control = tick

    res = flight.run(drone)
    assert res["outcome"] == "killed"
    assert "emergency" in drone.calls
    assert "kill_flag" in event_names(d)


def test_the_airborne_budget_ends_the_flight(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "3", "--mode", "hover")
    res = flight.run(drone)
    assert res["outcome"] == "airborne_budget"
    assert "airborne_budget" in event_names(d)


def test_a_battery_that_dies_in_flight_lands(tmp_path):
    flight, drone, _d = run(tmp_path)
    original = drone.send_rc_control
    n = {"i": 0}

    def tick(*a, **k):
        n["i"] += 1
        if n["i"] == 5:
            drone.sim.battery = 5.0
        return original(*a, **k)
    drone.send_rc_control = tick

    with pytest.raises(FlightError, match="requires at least"):
        flight.run(drone)
    assert_landed(drone)


def test_losing_video_in_flight_lands(tmp_path):
    flight, drone, _d = run(tmp_path)
    original = drone.send_rc_control
    n = {"i": 0}

    def tick(*a, **k):
        n["i"] += 1
        if n["i"] == 5:
            drone._streaming = False      # frame becomes None
        return original(*a, **k)
    drone.send_rc_control = tick

    with pytest.raises(FlightError, match="No camera frame for"):
        flight.run(drone)
    assert_landed(drone)


# --------------------------------------------------------------- the travel budget

def test_the_travel_budget_cuts_forward_but_keeps_flying(tmp_path):
    """The substitute for position feedback: there is no overhead camera, so the only
    way to bound where the drone goes is to bound how far it has been told to go."""
    flight, drone, d = run(tmp_path, "--max-travel", "0.15")
    flight.run(drone)
    assert flight.travel_m <= 0.15 + spec.RC_MAX / 125.0 * spec.CTRL_DT + 1e-6
    assert any(t.get("trimmed") and "travel_budget" in t["trimmed"]
               for t in ticks_in(d)), "the trim must be logged, not silent"


def ticks_in(d):
    return [json.loads(line) for line in
            (d / "ticks.jsonl").read_text(encoding="utf-8").splitlines()]


def test_no_forward_never_commands_a_forward_rc(tmp_path):
    """Stage 2 must be unable to translate, whatever the controller asks for."""
    flight, drone, d = run(tmp_path, "--no-forward", "--max-airborne", "6")
    flight.run(drone)
    assert drone.rc_calls(), "it did fly"
    assert all(fb == 0 for (_lr, fb, _ud, _yaw) in drone.rc_calls())
    assert not drone.outcome.crossed


def test_hover_mode_commands_nothing_at_all(tmp_path):
    flight, drone, _d = run(tmp_path, "--mode", "hover", "--max-airborne", "4")
    flight.run(drone)
    assert all(rc == (0, 0, 0, 0) for rc in drone.rc_calls())


def test_every_rc_channel_stays_inside_the_clamp(tmp_path):
    flight, drone, _d = run(tmp_path)
    flight.run(drone)
    for rc in drone.rc_calls():
        assert all(abs(v) <= spec.RC_MAX for v in rc)


def test_a_policy_returning_nonsense_is_clamped_to_a_hover(tmp_path):
    """A NaN from a model must not reach the drone, or crash the loop."""
    flight, drone, _d = run(tmp_path, "--max-airborne", "4")
    flight.controller = lambda ring, state: (np.array([np.nan, np.inf, 1e9, -1e9]), None)
    flight.run(drone)
    for rc in drone.rc_calls():
        assert all(abs(v) <= spec.RC_MAX for v in rc)


# --------------------------------------------------------------- outputs

def test_the_flight_directory_gets_the_files_the_dashboard_reads(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    flight.run(drone)
    for name in ("events.jsonl", "ticks.jsonl", "telemetry.csv", "status.json",
                 "summary.json"):
        assert (d / name).is_file(), name
    status = json.loads((d / "status.json").read_text(encoding="utf-8"))
    for key in ("phase", "battery", "height_cm", "ring_found", "range_m", "travel_m"):
        assert key in status, key


def test_the_camera_frame_is_written_only_when_asked_for(tmp_path):
    """Gated on the flag file so the JPEG encode costs nothing when nobody is watching."""
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    flight.run(drone)
    assert not (d / "camera_latest.jpg").exists()

    flight2, drone2, d2 = run(tmp_path / "second", "--max-airborne", "4")
    (d2 / "camera_enabled.flag").touch()
    flight2.run(drone2)
    assert (d2 / "camera_latest.jpg").is_file()


def test_tick_rows_carry_what_the_dashboard_plots(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    flight.run(drone)
    rows = ticks_in(d)
    assert rows and all(r["type"] == "tick" for r in rows)
    for key in ("phase", "found", "cx", "cy", "action", "rc", "travel_m", "bat"):
        assert key in rows[0], key


def test_recording_produces_a_loadable_demo(tmp_path):
    from imitation.record_demo import DemoRecorder, load_demo
    flight, drone, d = run(tmp_path, "--max-airborne", "6")
    flight.recorder = DemoRecorder(meta={"source": "test"}, obs_dim=spec.OBS_DIM,
                                   task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    flight.run(drone)
    demo = load_demo(d / "demo.npz")
    assert demo["obs"].shape[1] == spec.OBS_DIM
    assert demo["intent"].shape[1] == spec.ACT_DIM
    assert demo["meta"]["task"] == spec.TASK_NAME
    assert demo["meta"]["obs_shape"] == [spec.OBS_DIM]


# --------------------------------------------------------------- the dashboard camera

def test_the_camera_is_live_before_takeoff(tmp_path):
    """Frames used to be written only from the control loop, so a flight that never got
    airborne never produced one and the dashboard pane stayed blank."""
    flight, drone, d = run(tmp_path, "--countdown", "1", "--preview-before-takeoff")
    drone.fail_at = "takeoff_refused"
    (d / "camera_enabled.flag").touch()
    with pytest.raises(RuntimeError, match="'error'"):
        flight.run(drone)
    assert (d / "camera_latest.jpg").is_file()


def test_no_camera_temp_file_survives_a_flight(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    (d / "camera_enabled.flag").touch()
    flight.run(drone)
    assert (d / "camera_latest.jpg").is_file()
    assert not (d / "camera_latest.jpg.tmp").exists()


def test_a_failing_camera_write_is_logged_once_and_cleaned_up(tmp_path, monkeypatch):
    """A frame write that fails for a reason other than a lock (a locked file has its own
    fallback). It must not stop the flight, must not be silent (it was), and
    must not write an event for every frame."""
    from onboard import fly_hoop
    real_replace = fly_hoop.os.replace

    def locked(src, dst):
        if str(dst).endswith("camera_latest.jpg"):
            raise OSError(28, "No space left on device")   # not a lock: no fallback applies
        return real_replace(src, dst)
    monkeypatch.setattr(fly_hoop.os, "replace", locked)

    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    (d / "camera_enabled.flag").touch()
    res = flight.run(drone)
    assert res["outcome"] == "airborne_budget"
    assert event_names(d).count("camera_write_failing") == 1
    assert res["camera_write_failures"] > 1
    assert not (d / "camera_latest.jpg.tmp").exists()


# --------------------------------------------------------------- a drifting height reading

def _with_height_reading(drone, h=None, tof=None):
    """Make the drone's telemetry report a wrong `h` and/or `tof` once it is airborne."""
    original = drone.get_current_state

    def state():
        s = original()
        if drone._airborne:
            if h is not None:
                s["h"] = h
            if tof is not None:
                s["tof"] = tof
        return s
    drone.get_current_state = state


def test_a_drone_whose_h_reads_wrong_but_whose_range_sensor_is_fine_still_flies(tmp_path):
    """Seen on the real drone: h = -10 cm while hovering at tof = 75 cm, because the
    barometer behind h drifts as it warms. The flight was aborted as 'never left the
    ground' though it was hovering."""
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    _with_height_reading(drone, h=-10, tof=75)
    res = flight.run(drone)
    assert res["outcome"] == "airborne_budget"
    assert next(e for e in events_in(d) if e["event"] == "climbed")["height_cm"] == 75
    assert_landed(drone)


def test_no_height_from_either_sensor_still_stops_the_flight(tmp_path):
    """The relaxation is only for when the range sensor can vouch for the height."""
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    _with_height_reading(drone, h=0, tof=6553)
    with pytest.raises(FlightError, match="neither the height reading nor"):
        flight.run(drone)
    assert_landed(drone)


def test_the_height_ceiling_follows_the_range_sensor_when_h_is_wrong(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "4", "--max-height", "100")
    _with_height_reading(drone, h=50, tof=130)       # h says fine, the floor says 130 cm
    with pytest.raises(FlightError, match="outside"):
        flight.run(drone)
    assert_landed(drone)


# --------------------------------------------------------------- low height: soft floor, not abort

def test_the_soft_floor_holds_a_sinking_controller_up(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "2")
    down = np.array([0.0, 0.0, -0.8, 0.0])
    a, why = flight._safety_overrides(down, None, 35)          # below 40 cm: no descent
    assert a[spec.ACT_UD] == 0.0 and why == ["floor"]
    a, why = flight._safety_overrides(down, None, 25)          # below 30 cm: made to climb
    assert a[spec.ACT_UD] == pytest.approx(0.4) and why == ["floor_climb"]
    a, why = flight._safety_overrides(down, None, 70)          # high enough: untouched
    assert a[spec.ACT_UD] == pytest.approx(-0.8) and why == []


def test_hovering_low_no_longer_aborts_the_flight(tmp_path):
    """Seen on the real drone: 'Airborne height 15 cm is outside this flight's 25-160 cm
    range' ended flights. Between 10 cm and the floor the flight now goes on."""
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    _with_height_reading(drone, h=20, tof=20)
    res = flight.run(drone)
    assert res["outcome"] == "airborne_budget"
    assert_landed(drone)


def test_really_touching_the_floor_still_aborts(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    _with_height_reading(drone, h=8, tof=8)
    with pytest.raises(FlightError):
        flight.run(drone)
    assert_landed(drone)


def test_policy_start_climbs_to_the_simulators_start_height(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "2")
    h = {"tof": 60}
    sent = []

    class Rec:
        def snapshot(self):
            return dict(h), 0.0

    class Drone:
        def send_rc_control(self, lr, fb, ud, yaw):
            sent.append(ud)
            h["tof"] += 5 if ud > 0 else 0

    class Ev:
        def __init__(self):
            self.rows = []

        def write(self, name, **kw):
            self.rows.append((name, kw))
    ev = Ev()
    flight._preview = lambda s: True
    flight._pump = lambda: None
    flight._check = lambda *a, **k: {}         # the climb's safety checks need real telemetry
    flight._climb_to_start(Drone(), None, Rec(), ev)
    assert h["tof"] >= 80 and sent[-1] == 0                    # climbed, then stopped climbing
    assert ev.rows[-1][0] == "start_height" and ev.rows[-1][1]["reached"]


# --------------------------------------------------------------- ground start (2026-10-14)

def test_a_low_takeoff_hovers_and_climbs_instead_of_aborting(tmp_path):
    """The real crash: 'Airborne height 15 cm is outside this flight's 25-160 cm range'. The
    Tello's own takeoff stopped at 15 cm and the height check aborted during the hover. Now the
    hover and the climb only enforce the ceiling, and the controller gets the sticks after."""
    flight, drone, d = run(tmp_path, "--min-height", "25", "--max-airborne", "3")
    original = drone.takeoff

    def low_takeoff():
        original()
        drone.sim.pos[2] = 0.15
    drone.takeoff = low_takeoff
    res = flight.run(drone)
    names = event_names(d)
    assert res["outcome"] in ("airborne_budget", "aborted_by_controller")
    assert names.index("settled") < names.index("start_height") < names.index("control_handover")
    start = next(e for e in events_in(d) if e["event"] == "start_height")
    assert start["height_cm"] >= 25
    assert_landed(drone)


def test_the_climb_to_the_start_height_is_for_every_mode_but_hover(tmp_path):
    for mode, climbs in (("expert", True), ("hover", False)):
        flight, drone, d = run(tmp_path / mode, "--mode", mode, "--max-airborne", "1")
        flight.run(drone)
        assert ("start_height" in event_names(d)) is climbs
        assert "control_handover" in event_names(d)


def test_the_default_ceiling_clears_a_tall_hoop(tmp_path):
    assert args().max_height == 190 and args().max_airborne >= 120
    flight, drone, d = run(tmp_path, "--max-airborne", "2")
    _with_height_reading(drone, h=185, tof=185)
    assert flight.run(drone)["outcome"] == "airborne_budget"


def test_the_handover_restarts_the_controllers_memory(tmp_path):
    flight, drone, d = run(tmp_path)
    resets = []

    class Ctl:
        def reset(self):
            resets.append(1)
    flight.controller = Ctl()
    rows = []

    class Ev:
        def write(self, name, **kw):
            rows.append(name)
    flight._handover(drone, Ev())
    assert resets == [1] and rows == ["control_handover"]
    assert drone.calls[-1] == ("rc", (0, 0, 0, 0))


def test_a_human_flies_every_tick_even_between_camera_frames(tmp_path):
    """Keys used to be read only on ticks with a new frame: with a 10-30 Hz jittery stream that
    was much of the key lag. Half the frames are dropped here; the keys still drive every tick."""
    flight, drone, d = run(tmp_path, "--max-airborne", "2", "--start-height", "0")
    flight.a.mode = "human"
    calls = []

    class Keys:
        finished = kill_requested = False
        phase, reason, status_extra = "human", "", ""

        def __call__(self, ring, state=None):
            calls.append(1)
            return np.zeros(spec.ACT_DIM), None
    flight.controller = Keys()
    real = drone.get_frame_read

    class Flaky:
        """Every frame until the controller is flying, then only one read in four."""
        def __init__(self, r):
            self.r, self.n, self.fresh_in_flight = r, 0, 0

        @property
        def frame(self):
            if not calls:
                return self.r.frame
            self.n += 1
            if self.n % 4:
                return None
            self.fresh_in_flight += 1
            return self.r.frame

        def stop(self):
            self.r.stop()
    readers = []
    drone.get_frame_read = lambda **kw: readers.append(Flaky(real(**kw))) or readers[-1]
    flight.run(drone)
    assert len(calls) == len(ticks_in(d))
    assert len(calls) > 2 * (readers[0].fresh_in_flight + 1)


def test_the_clearml_upload_does_not_hold_the_flight_process(tmp_path, monkeypatch):
    """35 s of synchronous upload kept the process (and UDP port 8889, and the dashboard's
    Start button) busy after every flight. It is now a detached process."""
    import subprocess
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import clearml_sync
    started = []
    monkeypatch.setattr(clearml_sync, "enabled", lambda *a, **k: True)
    monkeypatch.setattr(clearml_sync, "record",
                        lambda *a, **k: pytest.fail("must not upload in the flight process"))
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: started.append(cmd))
    from onboard.fly_hoop import _clearml_record
    _clearml_record(tmp_path, {}, "landed")
    assert started and started[0][-5:] == ["record", "flights", str(tmp_path), "--status", "landed"]


# --------------------------------------------------------------- a refused takeoff

def test_takeoff_start_records_the_ground_state(tmp_path):
    """When the drone refuses takeoff, this is the only evidence of why."""
    flight, drone, d = run(tmp_path, "--max-airborne", "2")
    flight.run(drone)
    start = next(e for e in events_in(d) if e["event"] == "takeoff_start")
    assert isinstance(start["bat"], int) and isinstance(start["temph"], int)
    assert start["h"] == 0


def test_a_refused_takeoff_is_named_and_does_not_blame_the_landing(tmp_path, capsys):
    """The real failure: takeoff answered 'error', then land answered 'error' as well and
    was printed as LANDING NOT CONFIRMED -- though the drone never left the ground."""
    flight, drone, d = run(tmp_path)
    drone.fail_at = "takeoff_refused"
    with pytest.raises(RuntimeError, match="'error'"):
        flight.run(drone)
    assert_landed(drone)                     # still asked to land, as always
    failed = next(e for e in events_in(d) if e["event"] == "takeoff_failed")
    assert failed["refused_on_ground"] is True and "bat" in failed
    names = event_names(d)
    assert "land_refused_on_ground" in names and "land_failed" not in names
    assert "LANDING NOT CONFIRMED" not in capsys.readouterr().out


def test_a_takeoff_without_an_error_reply_is_not_called_a_refusal(tmp_path):
    """A lost acknowledgement is not a refusal: the drone may be climbing."""
    flight, drone, d = run(tmp_path)
    drone.fail_at = "takeoff"                # no 'error' reply in the message
    with pytest.raises(RuntimeError):
        flight.run(drone)
    failed = next(e for e in events_in(d) if e["event"] == "takeoff_failed")
    assert failed["refused_on_ground"] is False
    assert_landed(drone)


# --------------------------------------------------------------- sdk and video bitrate

def test_a_tello_edu_is_not_sent_setbitrate_when_probing(tmp_path):
    """setbitrate is an SDK 3.0 (RoboMaster TT) command. The Tello EDU answers 'unknown
    command', which the log used to show as a failure on every flight."""
    flight, drone, d = run(tmp_path, "--max-airborne", "2", "--probe-sdk")
    flight.run(drone)
    assert ("bitrate", 1) not in drone.calls
    names = event_names(d)
    assert "video_bitrate_unsupported" in names and "video_bitrate_failed" not in names
    assert next(e for e in events_in(d) if e["event"] == "sdk_version")["sdk"] == 20


def test_the_default_startup_is_the_one_flown_before_the_sdk_probe(tmp_path):
    """Takeoff was refused more often once `sdk?` was added before it, so it is opt-in.
    By default the drone is sent exactly the commands it always was: no `sdk?`, and the
    bitrate request (which an EDU refuses, harmlessly)."""
    flight, drone, d = run(tmp_path, "--max-airborne", "2")
    flight.run(drone)
    assert "sdk?" not in drone.calls
    assert ("bitrate", 1) in drone.calls
    names = event_names(d)
    assert "sdk_version" not in names and "video_bitrate_failed" in names


def test_no_camera_frame_is_written_before_takeoff_by_default(tmp_path):
    flight, drone, d = run(tmp_path, "--countdown", "1")
    (d / "camera_enabled.flag").touch()
    drone.fail_at = "takeoff_refused"
    with pytest.raises(RuntimeError):
        flight.run(drone)
    assert not (d / "camera_latest.jpg").exists()


def test_a_camera_file_the_dashboard_holds_open_is_still_updated(tmp_path, monkeypatch):
    """Windows refuses the rename over a file another process has open; a plain write to
    it is allowed, so that is the fallback and the frame keeps updating."""
    from onboard import fly_hoop
    real_replace = fly_hoop.os.replace

    def locked(src, dst):
        if str(dst).endswith("camera_latest.jpg"):
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)
    monkeypatch.setattr(fly_hoop.os, "replace", locked)
    flight, drone, d = run(tmp_path, "--max-airborne", "4")
    (d / "camera_enabled.flag").touch()
    res = flight.run(drone)
    assert (d / "camera_latest.jpg").stat().st_size > 1000
    assert not (d / "camera_latest.jpg.tmp").exists()
    assert "camera_write_failing" not in event_names(d)
    assert "camera_write_failures" not in res


def test_a_refused_takeoff_does_not_blame_the_ring(tmp_path, capsys):
    from imitation.record_demo import DemoRecorder
    flight, drone, d = run(tmp_path, "--mode", "human")
    flight.recorder = DemoRecorder(meta={"source": "human"}, obs_dim=spec.OBS_DIM,
                                   task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    drone.fail_at = "takeoff_refused"
    with pytest.raises(RuntimeError):
        flight.run(drone)
    out = capsys.readouterr().out
    assert "REFUSED takeoff" in out and "Power-cycle" in out
    assert "ring was never detected" not in out
    assert "did not end normally" in out


def test_an_sdk_3_drone_still_gets_the_lower_bitrate(tmp_path):
    flight, drone, d = run(tmp_path, "--max-airborne", "2", "--probe-sdk")
    drone.sdk = "30"
    flight.run(drone)
    assert ("bitrate", 1) in drone.calls
    assert "video_bitrate" in event_names(d)


# --------------------------------------------------------------- keys during a video stall

class _AsksDuringStall:
    """A controller whose land or kill request arrives only through latch() -- the path the
    flight loop takes on a tick with no video frame, when the controller is not called."""

    def __init__(self, request):
        self.request = request
        self.finished = self.kill_requested = False
        self.phase, self.reason = "human", ""

    def latch(self):
        setattr(self, self.request, True)

    def __call__(self, ring, state=None):
        return np.zeros(spec.ACT_DIM), None


@pytest.mark.parametrize("request_, outcome", [("finished", "aborted_by_controller"),
                                               ("kill_requested", "killed")])
def test_land_and_kill_are_heard_during_a_video_stall(tmp_path, request_, outcome):
    """Without a frame the controller is skipped, and it is the only thing reading the
    keyboard: a land or a motor cut pressed during a stall of up to 3 s was never seen,
    and the flight ended by the video watchdog instead."""
    flight, drone, d = run(tmp_path)
    flight.controller = _AsksDuringStall(request_)
    original = drone.takeoff

    def takeoff_then_lose_video():
        original()
        drone._streaming = False             # every frame from here on is None
    drone.takeoff = takeoff_then_lose_video

    res = flight.run(drone)                  # no FlightError from the video watchdog
    assert res["outcome"] == outcome
    assert_landed(drone)


# --------------------------------------------------------------- cli / config

def test_detector_defaults_are_used_when_no_tuning_file_exists(tmp_path):
    p = load_detector(tmp_path / "missing.json")
    assert p.allow_edge is True


def test_a_corrupt_detector_file_falls_back_instead_of_crashing(tmp_path):
    bad = tmp_path / "detector.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_detector(bad).allow_edge is True


def test_detector_file_is_honoured(tmp_path):
    from onboard.ring_detect import DetectorParams
    p = tmp_path / "detector.json"
    p.write_text(json.dumps(DetectorParams(residual_max=0.42).to_json()),
                 encoding="utf-8")
    assert load_detector(p).residual_max == pytest.approx(0.42)


def test_policy_mode_requires_a_model():
    with pytest.raises(SystemExit):
        parse(["--mode", "policy"])


def test_multiple_attempts_are_refused_on_real_hardware():
    """Real attempts are flown one at a time, with a human watching each."""
    with pytest.raises(SystemExit):
        parse(["--attempts", "5"])
    parse(["--dry-run", "--attempts", "5"])


def test_a_nonpositive_travel_budget_is_refused():
    with pytest.raises(SystemExit):
        parse(["--dry-run", "--max-travel", "0"])
