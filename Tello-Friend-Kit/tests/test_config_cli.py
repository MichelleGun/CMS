import socket

import pytest

import tello_lab.cli as cli
from tello_lab.config import Settings, load_settings


@pytest.mark.parametrize("pattern", ["hover", "square"])
def test_preview_cannot_access_hardware(pattern, monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("Offline preview tried to open sockets or create a drone")

    monkeypatch.setattr(cli, "create_drone", fail)
    monkeypatch.setattr(socket, "socket", fail)
    assert cli.main(["flight", pattern]) == 0
    assert "OFFLINE PREVIEW" in capsys.readouterr().out


@pytest.mark.parametrize(
    "settings",
    [
        {"host": "invalid"},
        {"speed_cm_s": 0},
        {"speed_cm_s": 31},
        {"square_side_cm": 19},
        {"square_side_cm": 101},
        {"square_side_cm": 50.5},
        {"hover_seconds": float("nan")},
        {"hover_seconds": 11},
        {"minimum_battery_percent": 29},
        {"abort_battery_percent": 30},
        {"countdown_seconds": 0},
        {"sample_hz": True},
    ],
)
def test_invalid_config_is_rejected(settings):
    with pytest.raises(ValueError):
        Settings(**settings)


def test_typo_in_config_is_rejected(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text("[flight]\nspeeed_cm_s = 20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown setting"):
        load_settings(path)


@pytest.mark.parametrize("command", ["telemetry", "video"])
def test_invalid_duration_is_rejected_before_connection(command, monkeypatch):
    monkeypatch.setattr(cli, "create_drone", lambda *args: pytest.fail("Tried to connect"))
    assert cli.main([command, "--seconds", "nan"]) == 1


def test_land_does_not_require_state_or_battery(monkeypatch):
    calls = []

    class LandingDrone:
        def connect(self, wait_for_state=True):
            assert wait_for_state is False
            calls.append("connect")

        def land(self):
            calls.append("land")

        def end(self):
            calls.append("end")

    monkeypatch.setattr(cli, "create_drone", lambda *args: LandingDrone())
    assert cli.main(["land"]) == 0
    assert calls == ["connect", "land", "end"]


@pytest.mark.parametrize("command", [["flight", "hover", "--fly"], ["survey", "--fly"]])
def test_measured_cage_blocks_flight_before_creating_drone(command, monkeypatch):
    monkeypatch.setattr(cli, "create_drone", lambda *args: pytest.fail("Opened hardware"))
    monkeypatch.setattr(cli, "load_settings", lambda *args: Settings(cage_height_cm=109.22))
    assert cli.main(command) == 1


def test_cage_measurements_load(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        "[cage]\nwidth_cm=109.22\nlength_cm=109.22\nheight_cm=109.22\nclearance_cm=30\n"
    )
    settings = load_settings(path)
    assert settings.cage_width_cm == settings.cage_length_cm == settings.cage_height_cm == 109.22


def test_cage_can_fit_hover_but_not_a_rise():
    from tello_lab.drone import FlightError, ensure_takeoff_clearance

    settings = Settings(cage_height_cm=120)
    ensure_takeoff_clearance(settings)
    with pytest.raises(FlightError, match="needs nominal height 100"):
        ensure_takeoff_clearance(settings, rise_cm=20)
