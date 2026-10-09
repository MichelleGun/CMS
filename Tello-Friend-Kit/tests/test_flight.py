import csv
import json

import pytest

import tello_lab.drone as controller
from tello_lab.config import Settings
from tello_lab.drone import FlightError, assert_ready, run_flight
from tello_lab.recording import TelemetryRecorder


class FakeDrone:
    def __init__(self, battery=80, failure=None):
        self.battery = battery
        self.failure = failure
        self.calls = []
        self.is_flying = False

    def connect(self, wait_for_state=True):
        self.calls.append("connect")
        if self.failure == "connect":
            raise RuntimeError("no response")

    def get_current_state(self):
        return {"bat": self.battery, "h": 80 if self.is_flying else 0}

    def set_speed(self, speed):
        self.calls.append(("speed", speed))

    def takeoff(self):
        self.calls.append("takeoff")
        if self.failure == "takeoff":
            # The SDK flag is still False if a successful takeoff's ACK was lost.
            raise RuntimeError("takeoff ACK lost")
        self.is_flying = True
        if self.failure == "battery_in_air":
            self.battery = 15

    def move_forward(self, distance):
        self.calls.append(("forward", distance))
        if self.failure == "movement":
            raise RuntimeError("movement timeout")
        if self.failure == "interrupt":
            raise KeyboardInterrupt

    def move_right(self, distance):
        self.calls.append(("right", distance))

    def move_back(self, distance):
        self.calls.append(("back", distance))

    def move_left(self, distance):
        self.calls.append(("left", distance))

    def land(self):
        self.calls.append("land")
        if self.failure == "landing":
            raise RuntimeError("landing ACK lost")
        self.is_flying = False

    def end(self):
        self.calls.append("end")


@pytest.fixture(autouse=True)
def instant_wait(monkeypatch):
    def sample_and_check(seconds, recorder, minimum):
        recorder.sample()
        assert_ready(recorder, minimum)

    monkeypatch.setattr(controller, "monitored_wait", sample_and_check)


def events_in(directory):
    path = next(directory.glob("*.jsonl"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hover_and_logs(tmp_path):
    drone = FakeDrone()
    run_flight(drone, Settings(), "hover", tmp_path)
    assert drone.calls == ["connect", ("speed", 20), "takeoff", "land", "end"]
    assert events_in(tmp_path)[-1]["event"] == "land_complete"
    with next(tmp_path.glob("*.csv")).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["bat"] == "80"
    assert "state_age_s" in rows[0]


def test_square_has_expected_relative_movements(tmp_path):
    drone = FakeDrone()
    run_flight(drone, Settings(), "square", tmp_path)
    assert drone.calls == [
        "connect",
        ("speed", 20),
        "takeoff",
        ("forward", 50),
        ("right", 50),
        ("back", 50),
        ("left", 50),
        "land",
        "end",
    ]


def test_low_battery_never_takes_off(tmp_path):
    drone = FakeDrone(battery=29)
    with pytest.raises(FlightError, match="Battery"):
        run_flight(drone, Settings(), "hover", tmp_path)
    assert drone.calls == ["connect", "end"]


@pytest.mark.parametrize("battery", [None, True, "80", -1, 101])
def test_invalid_battery_never_takes_off(tmp_path, battery):
    drone = FakeDrone(battery=battery)
    with pytest.raises(FlightError, match="valid battery"):
        run_flight(drone, Settings(), "hover", tmp_path)
    assert "takeoff" not in drone.calls


@pytest.mark.parametrize("failure", ["takeoff", "movement", "interrupt", "battery_in_air"])
def test_failure_or_interrupt_attempts_landing(tmp_path, failure):
    drone = FakeDrone(failure=failure)
    expected = KeyboardInterrupt if failure == "interrupt" else RuntimeError
    with pytest.raises(expected):
        run_flight(drone, Settings(), "square", tmp_path)
    assert drone.calls[-2:] == ["land", "end"]
    assert drone.calls.count("takeoff") == 1
    assert any(event["event"] == "test_aborted" for event in events_in(tmp_path))


def test_low_battery_during_flight_stops_before_movement(tmp_path):
    drone = FakeDrone(failure="battery_in_air")
    with pytest.raises(FlightError):
        run_flight(drone, Settings(), "square", tmp_path)
    assert ("forward", 50) not in drone.calls
    assert drone.calls[-2:] == ["land", "end"]


def test_countdown_interrupt_does_not_land_or_take_off(tmp_path, monkeypatch):
    def cancel(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "monitored_wait", cancel)
    drone = FakeDrone()
    with pytest.raises(KeyboardInterrupt):
        run_flight(drone, Settings(), "hover", tmp_path)
    assert "takeoff" not in drone.calls
    assert "land" not in drone.calls
    assert drone.calls[-1] == "end"


def test_landing_failure_is_not_reported_as_success(tmp_path):
    drone = FakeDrone(failure="landing")
    with pytest.raises(FlightError, match="Landing was not confirmed"):
        run_flight(drone, Settings(), "hover", tmp_path)
    assert events_in(tmp_path)[-1]["event"] == "land_failed"
    assert drone.calls[-1] == "end"


def test_connection_failure_closes_drone_without_takeoff(tmp_path):
    drone = FakeDrone(failure="connect")
    with pytest.raises(RuntimeError, match="no response"):
        run_flight(drone, Settings(), "hover", tmp_path)
    assert drone.calls == ["connect", "end"]


def test_unchanged_cached_state_becomes_stale(tmp_path, monkeypatch):
    packet = {"bat": 80, "h": 0}
    drone = FakeDrone()
    monkeypatch.setattr(drone, "get_current_state", lambda: packet)
    clock = [100.0]
    monkeypatch.setattr("tello_lab.recording.time.monotonic", lambda: clock[0])
    recorder = TelemetryRecorder(drone, tmp_path / "state.csv", 5)
    recorder.sample()
    assert_ready(recorder, 30)
    clock[0] += 4
    recorder.sample()
    with pytest.raises(FlightError, match="stale"):
        assert_ready(recorder, 30)
    # A new packet with identical contents is fresh.
    monkeypatch.setattr(drone, "get_current_state", lambda: dict(packet))
    recorder.sample()
    assert_ready(recorder, 30)


def test_recording_failure_after_takeoff_still_lands(tmp_path, monkeypatch):
    drone = FakeDrone()
    original = controller.EventLog.write

    def disk_full(self, event, **fields):
        if event == "command_complete":
            raise OSError("disk full")
        return original(self, event, **fields)

    monkeypatch.setattr(controller.EventLog, "write", disk_full)
    with pytest.raises(OSError, match="disk full"):
        run_flight(drone, Settings(), "hover", tmp_path)
    assert drone.calls[-2:] == ["land", "end"]
