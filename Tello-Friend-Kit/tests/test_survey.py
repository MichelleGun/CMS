import json

import pytest

import tello_lab.cli as cli
import tello_lab.observe as observation
import tello_lab.survey as survey
from tello_lab.config import Settings
from tello_lab.drone import FlightError


class Drone:
    def __init__(self, temperature=65, failure=None):
        self.temperature = temperature
        self.failure = failure
        self.flying = False
        self.height = 80
        self.pad_id = -1
        self.pad_x = 0
        self.pad_y = 0
        self.calls = []

    def connect(self, **kwargs):
        self.calls.append("connect")

    def get_current_state(self):
        return {
            "h": self.height if self.flying else 0,
            "tof": self.height + 5 if self.flying else 10,
            "bat": 65,
            "temph": self.temperature,
            "mid": self.pad_id,
            "x": self.pad_x,
            "y": self.pad_y,
            "z": self.height,
        }

    def enable_mission_pads(self):
        self.calls.append("mon")

    def set_mission_pad_detection_direction(self, value):
        self.calls.append("mdirection")

    def disable_mission_pads(self):
        self.calls.append("moff")

    def streamon(self):
        self.calls.append("streamon")

    def streamoff(self):
        self.calls.append("streamoff")

    def get_frame_read(self, **kwargs):
        return self

    def stop(self):
        self.calls.append("reader_stop")

    def takeoff(self):
        self.calls.append("takeoff")
        if self.failure == "takeoff":
            raise RuntimeError("takeoff ACK lost")
        self.flying = True

    def rotate_clockwise(self, value):
        self.calls.append(("rotate", value))
        if self.failure == "rotate":
            raise RuntimeError("rotation ACK lost")

    def rotate_counter_clockwise(self, value):
        self.calls.append(("rotate_ccw", value))

    def set_video_bitrate(self, value):
        self.calls.append(("bitrate", value))

    def set_speed(self, value):
        self.calls.append(("speed", value))

    def move_up(self, value):
        self.calls.append(("up", value))
        if self.failure == "rise":
            raise RuntimeError("rise ACK lost")
        self.height += value

    def go_xyz_speed_mid(self, *args):
        self.calls.append(("pad_go", *args))
        if self.failure == "pad_go":
            raise RuntimeError("pad return ACK lost")
        self.height = args[2]

    def land(self):
        self.calls.append("land")
        self.flying = False

    def end(self):
        self.calls.append("end")


@pytest.fixture
def fast_survey(tmp_path, monkeypatch):
    monkeypatch.setattr(survey, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(survey.time, "sleep", lambda *args: None)
    clock = [0]

    def monotonic():
        clock[0] += 0.05
        return clock[0]

    monkeypatch.setattr(survey.time, "monotonic", monotonic)

    def capture(reader, path, recorder, battery, **kwargs):
        # Keep the fresh packet monitor in step with our synthetic clock.
        recorder.sample()
        return {
            "image": str(path),
            "width": 960,
            "height": 720,
            "state": reader.get_current_state(),
        }

    monkeypatch.setattr(survey, "capture", capture)
    # Countdown tests need a fresh snapshot at each call, just as live UDP state provides.
    original = survey.health

    def health(recorder, battery, **kwargs):
        recorder.sample()
        return original(recorder, battery, **kwargs)

    monkeypatch.setattr(survey, "health", health)
    return tmp_path


def test_survey_preview_never_creates_drone(monkeypatch):
    monkeypatch.setattr(cli, "create_drone", lambda *args: pytest.fail("Opened hardware"))
    assert cli.main(["survey"]) == 0


def test_hot_drone_does_not_start_motors(fast_survey):
    drone = Drone(temperature=87)
    with pytest.raises(FlightError, match="temperature|Mainboard"):
        survey.run_survey(drone, Settings())
    assert drone.calls == ["connect", "end"]


def test_four_headings_and_landing(fast_survey):
    drone = Drone()
    result = survey.run_survey(drone, Settings())
    assert result["status"] == "survey_complete"
    assert result["landing_acknowledged"] is True
    assert drone.calls.count(("rotate", 90)) == 4
    assert not drone.flying
    assert "land" in drone.calls
    assert drone.calls[-1] == "end"


@pytest.mark.parametrize("failure", ["takeoff", "rotate"])
def test_failed_ack_lands_and_closes_stream(fast_survey, failure):
    drone = Drone(failure=failure)
    with pytest.raises(RuntimeError, match="ACK lost"):
        survey.run_survey(drone, Settings())
    assert drone.calls.count("takeoff") == 1
    assert drone.calls.count("land") == 1
    assert "streamoff" in drone.calls and "moff" in drone.calls
    summary = next((fast_survey / "data" / "logs").glob("*.summary.json"))
    assert json.loads(summary.read_text())["status"] == "aborted"


def test_observation_rejects_heat_before_camera(fast_survey, monkeypatch):
    monkeypatch.setattr(observation, "PROJECT_ROOT", fast_survey)
    drone = Drone(temperature=88)
    with pytest.raises(FlightError, match="Mainboard"):
        observation.observe(drone, Settings(), 1, display=False)
    assert drone.calls == ["connect", "end"]


def test_observation_stream_failure_disables_pads(fast_survey, monkeypatch):
    monkeypatch.setattr(observation, "PROJECT_ROOT", fast_survey)
    drone = Drone()

    def fail_stream():
        raise RuntimeError("stream unavailable")

    monkeypatch.setattr(drone, "streamon", fail_stream)
    with pytest.raises(RuntimeError, match="stream unavailable"):
        observation.observe(drone, Settings(), 1, display=False)
    assert "streamoff" in drone.calls and "moff" in drone.calls
    assert "takeoff" not in drone.calls and "land" not in drone.calls
    assert drone.calls[-1] == "end"


@pytest.mark.parametrize("field,value", [("h", 0), ("h", 130), ("tof", 6553)])
def test_invalid_airborne_sensors_stop_survey(field, value):
    class Recorder:
        def snapshot(self):
            return {"h": 80, "tof": 85, "bat": 65, "temph": 65, field: value}, 0

    with pytest.raises(FlightError, match="height|range"):
        survey.health(Recorder(), 20, airborne=True)


def test_forward_search_rises_turns_and_returns_to_pad(fast_survey):
    drone = Drone()
    drone.pad_id = 3
    result = survey.run_survey(drone, Settings(), search_forward=True, landing_pad_id=3)
    assert result["pad_return_acknowledged"] == 3
    assert ("up", 20) in drone.calls
    assert drone.calls.count(("rotate", 45)) == 2
    assert ("rotate_ccw", 90) in drone.calls
    assert ("pad_go", 0, 0, 80, 20, 3) in drone.calls
    assert result["landing_acknowledged"] is True
    assert ("bitrate", 0) in drone.calls


@pytest.mark.parametrize("failure", ["rise", "pad_go"])
def test_search_command_failure_lands(fast_survey, failure):
    drone = Drone(failure=failure)
    drone.pad_id = 3
    with pytest.raises(RuntimeError, match="ACK lost"):
        survey.run_survey(drone, Settings(), search_forward=True, landing_pad_id=3)
    assert drone.calls.count("land") == 1
    assert ("bitrate", 0) in drone.calls


@pytest.mark.parametrize("pad_id,pad_x,pad_y", [(-1, 0, 0), (3, 40, 0), (3, 25, 25)])
def test_search_refuses_return_with_missing_or_distant_pad(fast_survey, pad_id, pad_x, pad_y):
    drone = Drone()
    drone.pad_id, drone.pad_x, drone.pad_y = pad_id, pad_x, pad_y
    with pytest.raises(FlightError, match="detected|return limit"):
        survey.run_survey(drone, Settings(), search_forward=True, landing_pad_id=3)
    assert not any(isinstance(call, tuple) and call[0] == "pad_go" for call in drone.calls)
    assert "land" in drone.calls


def test_small_cage_refuses_search_before_any_drone_command(fast_survey):
    drone = Drone()
    with pytest.raises(FlightError, match="Cage height"):
        survey.run_survey(drone, Settings(cage_height_cm=109.22), search_forward=True)
    assert drone.calls == []
