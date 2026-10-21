"""Camera calibration: auto-fit from clicks (ring_detect.fit_hsv), presets (detector_presets), the
dashboard's calibration session (calib_session.py) and fly_hoop --detector. No drone needed."""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from onboard import detector_presets as dp
from onboard.ring_detect import DetectorParams, blue_mask, detect_rings, fit_hsv

HERE = Path(__file__).resolve().parent
ONBOARD = HERE.parent

RING_CENTRE, RING_R = (420, 250), 90


def daytime_frame():
    """Grey room, a saturated blue noodle ring, and a big bright pale-blue window with a frame
    cross (which gives the window holes, so it can look like a ring to a loose colour window)."""
    img = np.full((480, 640, 3), 95, np.uint8)
    cv2.rectangle(img, (30, 40), (250, 330), (255, 225, 190), -1)        # window: pale sky blue
    cv2.line(img, (140, 40), (140, 330), (95, 95, 95), 6)
    cv2.line(img, (30, 185), (250, 185), (95, 95, 95), 6)
    cv2.circle(img, RING_CENTRE, RING_R, (190, 70, 0), 22)               # the noodle: saturated blue
    return img


def frac(p):
    return [p[0] / 639, p[1] / 479]


RING_CLICKS = [frac((RING_CENTRE[0] + RING_R, RING_CENTRE[1])), frac((RING_CENTRE[0], RING_CENTRE[1] - RING_R)),
               frac((RING_CENTRE[0] - RING_R, RING_CENTRE[1]))]
WINDOW_CLICKS = [frac((80, 100)), frac((200, 260)), frac((80, 260))]


def window_in_mask(img, params):
    m = blue_mask(img, params)
    return float(np.mean(m[60:170, 50:120] > 0))


def test_the_default_colour_window_lets_the_daylight_window_in():
    loose = DetectorParams(hsv_lo=(90, 30, 40), hsv_hi=(135, 255, 255))
    assert window_in_mask(daytime_frame(), loose) > 0.9


def test_autofit_keeps_the_ring_and_rejects_the_window():
    img = daytime_frame()
    loose = DetectorParams(hsv_lo=(90, 30, 40), hsv_hi=(135, 255, 255))
    fitted, stats = fit_hsv(img, RING_CLICKS, WINDOW_CLICKS, base=loose)
    assert stats["before"]["rejected_in"] > 0.9
    assert stats["after"]["rejected_in"] < 0.05 and stats["after"]["ring_kept"] >= 0.9
    assert window_in_mask(img, fitted) < 0.05
    rings = detect_rings(img, fitted)
    assert rings and abs(rings[0]["center_px"][0] - RING_CENTRE[0]) < 8
    assert fitted.min_major_px == loose.min_major_px            # shape gates untouched


def test_autofit_needs_a_click_on_the_ring():
    with pytest.raises(ValueError):
        fit_hsv(daytime_frame(), [], WINDOW_CLICKS)


@pytest.fixture
def presets(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "PRESET_DIR", tmp_path / "presets")
    default = tmp_path / "detector.json"
    default.write_text(json.dumps(DetectorParams().to_json()))
    monkeypatch.setattr(dp, "DEFAULT_FILE", default)
    return tmp_path


def test_presets_save_list_load_default_and_delete(presets):
    p = DetectorParams(hsv_lo=(100, 150, 60), hsv_hi=(120, 255, 230)).to_json()
    saved = dp.save_preset("Day 17 Oct", p, condition="day", notes="window side", thumb_jpg=b"\xff\xd8")
    assert saved["slug"] == "day-17-oct" and saved["params"]["hsv_lo"] == [100, 150, 60]
    names = [d["slug"] for d in dp.list_presets()]
    assert names[0] == "default" and "day-17-oct" in names
    assert dp.get_preset("Day 17 Oct")["condition"] == "day"           # by name or slug
    dp.set_default("day-17-oct")
    assert json.loads(dp.DEFAULT_FILE.read_text())["hsv_lo"] == [100, 150, 60]
    with pytest.raises(ValueError):
        dp.delete_preset("default")
    dp.delete_preset("day-17-oct")
    with pytest.raises(KeyError):
        dp.get_preset("day-17-oct")


def test_bad_presets_are_refused(presets):
    with pytest.raises(ValueError):
        dp.save_preset("x", {"hsv_lo": [200, 0, 0], "hsv_hi": [210, 255, 255]})
    with pytest.raises(ValueError):
        dp.save_preset("default", DetectorParams().to_json())
    with pytest.raises(ValueError):
        dp.save_preset("x", DetectorParams().to_json(), condition="dusk")


def test_the_suggestion_follows_the_clock(presets):
    from datetime import datetime
    dp.save_preset("Night", DetectorParams().to_json(), condition="night")
    dp.save_preset("Day", DetectorParams().to_json(), condition="day")
    ps = dp.list_presets()
    assert dp.suggest(ps, datetime(2026, 10, 17, 14, 0)) == "day"
    assert dp.suggest(ps, datetime(2026, 10, 17, 22, 0)) == "night"


def test_fly_hoop_flies_a_named_preset(presets):
    from onboard.fly_hoop import load_detector, parse
    dp.save_preset("Day", DetectorParams(hsv_lo=(101, 140, 50)).to_json(), condition="day")
    assert load_detector(preset="day").hsv_lo == (101, 140, 50)
    assert parse(["--dry-run", "--detector", "day"]).detector == "day"
    with pytest.raises(SystemExit):
        parse(["--dry-run", "--detector", "no-such-preset"])


def test_a_calibration_session_shows_the_detector_and_follows_the_sliders(tmp_path):
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), daytime_frame())
    d = tmp_path / "session"
    d.mkdir()
    (d / "params.json").write_text(json.dumps(DetectorParams(hsv_lo=(90, 30, 40)).to_json()))
    proc = subprocess.Popen([sys.executable, str(ONBOARD / "calib_session.py"), "--dir", str(d),
                             "--source", "image", "--image", str(img_path), "--hz", "10"])
    try:
        def wait(cond, s=30):
            end = time.time() + s
            while time.time() < end:
                try:
                    if cond():
                        return True
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(0.2)
            return False
        assert wait(lambda: (d / "overlay.jpg").exists() and (d / "status.json").exists())
        for name in ("raw.jpg", "mask.jpg"):
            assert (d / name).stat().st_size > 1000
        # a click-fit request is answered and becomes the live params
        (d / "autofit.json").write_text(json.dumps({"ring": RING_CLICKS, "reject": WINDOW_CLICKS}))
        assert wait(lambda: json.loads((d / "autofit_result.json").read_text())["ok"])
        fitted = json.loads((d / "params.json").read_text())
        assert wait(lambda: json.loads((d / "status.json").read_text())["params"]["hsv_lo"] == fitted["hsv_lo"])
        assert wait(lambda: json.loads((d / "status.json").read_text())["found"])
        (d / "stop.flag").touch()
        proc.wait(timeout=20)
        assert json.loads((d / "status.json").read_text()).get("ended")
    finally:
        if proc.poll() is None:
            proc.kill()
