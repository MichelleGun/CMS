"""Calibration scripts end-to-end against FakeTello: do they recover what the fake drone actually does?"""
import json

import numpy as np
import pytest
import yaml

from calibration import calibrate_axes, calibrate_speed
from calibration.analysis import fit_rc_per_mps
from common.config import load_config, _merge
from common.fake_tello import FakeOverheadCamera, FakeTello, SimClock
from common.flight_session import FlightSession

BASE = ["--dry-run", "--no-hotkeys", "--yes", "--write"]


def read(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


def test_axes_calibration_confirms_matching_conventions(tmp_path):
    out = tmp_path / "local.yaml"
    res = calibrate_axes.main(BASE + ["--profile", "typical", "--seed", "3", "--config-out", str(out)])
    assert res.ok, res.problems
    conv = read(out)["conventions"]
    cfg = load_config()["conventions"]
    assert conv["axes_calibrated"] is True
    assert conv["rc_signs"] == cfg["rc_signs"] and conv["vel_signs"] == cfg["vel_signs"]
    assert conv["att_signs"] == cfg["att_signs"]
    assert conv["vel_to_mps"] == pytest.approx(0.1)
    assert res.notes == []                                         # nothing needed flipping


def test_axes_calibration_detects_mirrored_left_right_and_fixes_it(tmp_path):
    out = tmp_path / "local.yaml"
    res = calibrate_axes.main(BASE + ["--profile", "typical", "--seed", "3", "--fake-axis-flip", "--config-out", str(out)])
    assert res.ok, res.problems
    assert any("lr" in n and "flipping" in n for n in res.notes)
    conv = read(out)["conventions"]
    assert conv["rc_signs"]["lr"] == -load_config()["conventions"]["rc_signs"]["lr"]
    assert conv["rc_signs"]["fb"] == load_config()["conventions"]["rc_signs"]["fb"]      # others untouched

    # and with the corrected conventions, commanding +y really moves the (mirrored) drone to +y
    cfg = _merge(load_config(), {"conventions": conv})
    clock = SimClock()
    fake = FakeTello(cfg, "typical", seed=5, clock=clock, axis_flip=True)
    cam = FakeOverheadCamera(fake, seed=6)
    fake.connect()
    s = FlightSession(cfg, fake, cam)
    for _ in range(60):
        s.tick()
    assert s.request_takeoff() == []
    for _ in range(300):
        if s.tick().phase == "flying":
            break
    y0 = fake.pos[1]
    for _ in range(60):
        s.tick(manual_action=np.array([0.0, 1.0, 0.0, 1.0]))
    assert fake.pos[1] - y0 > 0.1
    assert "axis_mismatch" not in s.tick().decision.reasons


def test_axes_calibration_refuses_to_certify_when_a_channel_is_unresolved(tmp_path, monkeypatch):
    """If the data cannot establish a channel, axes_calibrated must stay false."""
    out = tmp_path / "local.yaml"
    from calibration import analysis
    real = analysis.analyze_axes

    def broken(trials, current):
        trials = dict(trials)
        trials.pop("yaw")
        return real(trials, current)
    monkeypatch.setattr(calibrate_axes, "analyze_axes", broken)
    res = calibrate_axes.main(BASE + ["--seed", "3", "--config-out", str(out)])
    assert not res.ok and any("yaw" in p for p in res.problems)
    assert not out.exists()


def test_speed_calibration_recovers_the_fake_drones_true_rc_scale(tmp_path):
    out = tmp_path / "local.yaml"
    k, taus, latency = calibrate_speed.main(BASE + ["--profile", "typical", "--seed", "2", "--config-out", str(out)])
    true_k = 125.0                                                  # FakeTelloProfile('typical').true_rc_per_mps
    assert abs(k - true_k) / true_k < 0.15, k
    assert read(out)["control"]["rc_per_mps"] == pytest.approx(k, abs=0.06)
    assert taus and 0.2 < np.median(taus) < 2.0


def test_fit_rc_per_mps_through_origin():
    assert fit_rc_per_mps([10, 20, 40], [0.1, 0.2, 0.4]) == pytest.approx(100.0)


def test_calibration_flights_can_take_off_on_real_hardware_but_normal_flights_cannot():
    """Regression: real takeoff needs calibrated axes, yet calibrate_axes.py must fly to establish them."""
    from calibration.calib_flight import CalibFlight
    cfg = load_config()
    assert cfg["conventions"]["axes_calibrated"] is False

    def rig():
        clock = SimClock()
        fake = FakeTello(cfg, "typical", seed=9, clock=clock)
        cam = FakeOverheadCamera(fake, seed=10)
        fake.connect()
        return fake, cam

    fake, cam = rig()
    normal = FlightSession(cfg, fake, cam, real=True)                       # a real-hardware production session
    for _ in range(60):
        normal.tick()
    problems = normal.request_takeoff()
    assert any("axes not calibrated" in p for p in problems)
    assert all(k != "takeoff" for k, _ in fake.log)

    fake, cam = rig()
    cf = CalibFlight(cfg, fake, cam, hotkeys=False)
    cf.session.real = True                                                  # same rule set as real hardware
    assert cf.wait_ready()
    assert cf.takeoff()                                                     # calibration is allowed to fly
    assert any(k == "takeoff" for k, _ in fake.log)
