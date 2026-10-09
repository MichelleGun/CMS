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
