"""
The dashboard's camera calibration (calib_manager.py + /api/calib/*): a real calib_session.py
process on a still image (no drone), slider changes, the click auto-fit, presets, the flight's
--detector flag, and the drone lock between a calibration and a flight.

    python -m pytest webapp/backend/tests/test_calib.py -q
"""
import sys
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# flake8: noqa: E402
import calib_manager as cm
import main
from fastapi.testclient import TestClient

sys.path.insert(0, str(BACKEND.parents[1] / "hardware"))
from onboard.tests.test_calibration import RING_CLICKS, WINDOW_CLICKS, daytime_frame


@pytest.fixture
def env(tmp_path, monkeypatch):
    import cv2
    import json
    from onboard.ring_detect import DetectorParams
    flights = tmp_path / "flights"
    (flights / "tello_20261017-120000").mkdir(parents=True)
    cv2.imwrite(str(flights / "tello_20261017-120000" / "camera_latest.jpg"), daytime_frame())
    monkeypatch.setattr(cm, "CMS_DIR", tmp_path)
    mgr = cm.CalibManager(sessions_dir=tmp_path / "sessions", flights_dir=flights)
    monkeypatch.setattr(main, "calib_manager", mgr)
    monkeypatch.setattr(cm.presets, "PRESET_DIR", tmp_path / "presets")
    default = tmp_path / "detector.json"
    default.write_text(json.dumps(DetectorParams(hsv_lo=(90, 30, 40)).to_json()))
    monkeypatch.setattr(cm.presets, "DEFAULT_FILE", default)
    client = TestClient(main.app)
    yield dict(client=client, mgr=mgr, tmp=tmp_path)
    mgr.stop()


def wait(cond, s=30):
    end = time.time() + s
    while time.time() < end:
        try:
            if cond():
                return True
        except (OSError, ValueError, KeyError, TypeError):
            pass
        time.sleep(0.2)
    return False


def test_a_session_on_an_image_shows_frames_and_saves_a_day_preset(env):
    c = env["client"]
    imgs = c.get("/api/calib/images").json()["images"]
    assert imgs and imgs[0]["path"].endswith("camera_latest.jpg")
    r = c.post("/api/calib/start", json={"source": "image", "image": imgs[0]["path"]})
    assert r.status_code == 201, r.text
    ok = wait(lambda: c.get("/api/calib/status").json()["status"]["frame"], s=90)
    assert ok, c.get("/api/calib/status").json()
    for kind in ("raw", "overlay", "mask"):
        assert wait(lambda: c.get(f"/api/calib/frame/{kind}.jpg").status_code == 200)
    assert c.get("/api/calib/frame/nope.jpg").status_code == 404

    # the real detector.json's shape gates may reject a fully visible ring: start from the
    # standard shape gates (the panel's "reset shape gates" button), then fit the colours
    from onboard.ring_detect import DetectorParams
    shape = {k: v for k, v in DetectorParams().to_json().items() if k not in ("hsv_lo", "hsv_hi")}
    assert c.put("/api/calib/params", json={"params": dict(shape, hsv_lo=[90, 30, 40])}).status_code == 200
    # the window is in the colour mask with the loose default: auto-fit it out
    res = c.post("/api/calib/autofit", json={"ring": RING_CLICKS, "reject": WINDOW_CLICKS}).json()
    assert res["ok"] and res["after"]["rejected_in"] < 0.05 and res["after"]["ring_kept"] >= 0.9
    assert wait(lambda: c.get("/api/calib/status").json()["status"]["found"])

    # a slider change reaches the session
    r = c.put("/api/calib/params", json={"params": {"min_major_px": 30}})
    assert r.status_code == 200 and r.json()["params"]["min_major_px"] == 30
    assert wait(lambda: c.get("/api/calib/status").json()["status"]["params"]["min_major_px"] == 30)
    assert c.put("/api/calib/params", json={"params": {"hsv_lo": [300, 0, 0]}}).status_code == 400

    saved = c.post("/api/calib/presets", json={"name": "Day 17 Oct", "condition": "day",
                                                "notes": "window side"})
    assert saved.status_code == 201 and saved.json()["slug"] == "day-17-oct"
    assert c.get("/api/calib/presets/day-17-oct/thumb.jpg").status_code == 200
    listed = c.get("/api/calib/presets").json()
    assert [p["slug"] for p in listed["presets"]][:2] == ["default", "day-17-oct"]
    assert listed["presets"][1]["params"]["hsv_lo"] == res["params"]["hsv_lo"]

    assert c.post("/api/calib/stop").json()["stopped"]
    assert wait(lambda: not c.get("/api/calib/status").json()["active"])


def test_presets_default_and_delete(env):
    c = env["client"]
    cm.presets.save_preset("Night", {"hsv_lo": [95, 80, 30], "hsv_hi": [130, 255, 255]}, condition="night")
    r = c.post("/api/calib/presets/night/default")
    assert r.status_code == 200 and r.json()["presets"][0]["params"]["hsv_lo"] == [95, 80, 30]
    assert c.delete("/api/calib/presets/default").status_code == 400
    assert c.delete("/api/calib/presets/night").status_code == 200
    assert c.delete("/api/calib/presets/night").status_code == 404


def test_a_flight_carries_the_chosen_preset(env):
    from drone_manager import DroneManager
    cm.presets.save_preset("Day", {"hsv_lo": [100, 140, 50], "hsv_hi": [125, 255, 230]}, condition="day")
    argv = DroneManager(flights_dir=env["tmp"] / "flights").build_argv(mode="hover", detector_preset="day")
    assert argv[argv.index("--detector") + 1] == "day"
    assert "--detector" not in DroneManager(flights_dir=env["tmp"] / "flights").build_argv(mode="hover")
    with pytest.raises(ValueError):
        DroneManager(flights_dir=env["tmp"] / "flights").build_argv(mode="hover", detector_preset="nope")


def test_a_tello_calibration_and_a_flight_exclude_each_other(env, monkeypatch):
    c = env["client"]
    monkeypatch.setattr(main, "_is_local", lambda client: True)       # the test client is not "this computer"
    monkeypatch.setattr(main.drone_manager, "is_flight_active", lambda: True)
    r = c.post("/api/calib/start", json={"source": "tello"})
    assert r.status_code == 409 and "flight" in r.json()["detail"]
    monkeypatch.setattr(main.drone_manager, "is_flight_active", lambda: False)
    monkeypatch.setattr(env["mgr"], "uses_drone", lambda: True)
    r = c.post("/api/drone/flights", json={"mode": "hover", "dry_run": True})
    assert r.status_code == 409 and "calibration" in r.json()["detail"]


def test_nothing_runs_without_a_session_and_images_stay_inside_the_project(env):
    c = env["client"]
    assert c.put("/api/calib/params", json={"params": {"min_major_px": 30}}).status_code == 409
    assert c.post("/api/calib/autofit", json={"ring": [[0.5, 0.5]]}).status_code == 409
    r = c.post("/api/calib/start", json={"source": "image", "image": "../../etc/passwd"})
    assert r.status_code == 400
    assert c.post("/api/calib/start", json={"source": "image", "preset": "nope",
                                            "image": "flights/tello_20261017-120000/camera_latest.jpg"}).status_code == 404
