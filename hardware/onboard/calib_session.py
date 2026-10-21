"""
Camera calibration session for the dashboard. GROUND ONLY: it connects to the camera (the Tello,
a webcam or a still image), shows what the ring detector sees, and never sends takeoff or stick
commands.

    python hardware/onboard/calib_session.py --dir <session folder> --source tello
    python hardware/onboard/calib_session.py --dir <session folder> --source image --image shot.jpg

The dashboard (webapp/backend/calib_manager.py) starts it and talks to it through files in the
session folder:
  written here, about 5 times a second (atomically):
    raw.jpg       the camera frame
    overlay.jpg   the frame with every ring candidate outlined (yellow) and the chosen one (green)
    mask.jpg      the colour mask (white = inside the colour window)
    status.json   fps, candidates, the chosen ring (range, centre, arc), battery, the params in use
  read here:
    params.json   detector params; re-read whenever it changes (the dashboard's sliders)
    autofit.json  {"ring": [[x, y], ...], "reject": [[x, y], ...], "clear": false} as fractions
                  of the frame: sample the pixels at these clicks on the CURRENT frame, add them to
                  the samples gathered so far in this session (so the hoop can be clicked on one
                  frame and the window on another), fit the colour window to all of them
                  (ring_detect.fit_hsv_pixels), write params.json and autofit_result.json, then
                  delete the request. "clear": true forgets the samples first.
    image.json    {"path": ...} (source image only): switch to another saved frame
    snapshot.flag save the current raw frame to snapshots/
    stop.flag     end the session
Why: daytime flights locked onto the window instead of the hoop with the night calibration
(2026-10-14), so the calibration has to be easy to redo for each lighting.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HW_ROOT = Path(__file__).resolve().parents[1]
if str(HW_ROOT) not in sys.path:
    sys.path.insert(0, str(HW_ROOT))

# flake8: noqa: E402
import cv2
import numpy as np

from onboard import spec
from onboard.detector_presets import get_preset
from onboard.ring_detect import (DetectorParams, RingTracker, blue_mask, detect_rings, fit_hsv_pixels,
                                 sample_hsv)
from onboard.tune_detector import Source


def _write_atomic(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    for _ in range(5):                       # OneDrive can hold a file for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)


def _jpg(img, quality=80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def draw_overlay(bgr, candidates, chosen, obs):
    out = bgr.copy()
    for c in candidates:
        (cx, cy), (a, b) = c["center_px"], c["axes_px"]
        cv2.ellipse(out, (int(cx), int(cy)), (int(a / 2), int(b / 2)), c["angle_deg"], 0, 360,
                    (0, 220, 255), 1)
    if chosen is not None:
        (cx, cy), (a, b) = chosen["center_px"], chosen["axes_px"]
        cv2.ellipse(out, (int(cx), int(cy)), (int(a / 2), int(b / 2)), chosen["angle_deg"], 0, 360,
                    (60, 220, 60), 3)
        cv2.drawMarker(out, (int(cx), int(cy)), (60, 220, 60), cv2.MARKER_CROSS, 14, 2)
    text = (f"ring: {'yes' if obs.found else 'no'}  candidates: {len(candidates)}"
            + (f"  range {obs.range_m:.2f} m" if obs.found and np.isfinite(obs.range_m) else ""))
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run(a):
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    params_file = d / "params.json"
    if not params_file.exists():
        start = get_preset(a.preset or "default")["params"]
        params_file.write_text(json.dumps(start, indent=2), encoding="utf-8")
    params = DetectorParams.from_json(_read_json(params_file) or {})
    mtime = params_file.stat().st_mtime
    outer = spec.RING_OPENING_M + 2 * spec.RING_NOODLE_M
    tracker = RingTracker(params, ring_diameter_m=outer, hfov_deg=spec.CAMERA_HFOV_DEG)
    src_args = argparse.Namespace(image=a.image if a.source == "image" else None,
                                  webcam=a.webcam if a.source == "webcam" else None, config=None)
    period, frames, t_fps, fps = 1.0 / a.hz, 0, time.monotonic(), 0.0
    ring_px, reject_px = np.zeros((0, 3), np.int32), np.zeros((0, 3), np.int32)
    image_name = Path(a.image).name if a.image else None
    deadline = time.monotonic() + a.max_minutes * 60
    with Source(src_args) as src:
        while not (d / "stop.flag").exists() and time.monotonic() < deadline:
            t0 = time.monotonic()
            try:
                m = params_file.stat().st_mtime
                if m != mtime:
                    new = _read_json(params_file)
                    if new is not None:
                        params = DetectorParams.from_json(new)
                        tracker.params = params
                        tracker.reset()
                    mtime = m
            except OSError:
                pass
            switch = _read_json(d / "image.json") if (d / "image.json").exists() else None
            if switch is not None and a.source == "image":
                img = cv2.imread(str(switch.get("path", "")))
                if img is not None:
                    src._img = img
                    image_name = Path(switch["path"]).name
                    tracker.reset()
                (d / "image.json").unlink(missing_ok=True)
            bgr = src.read()
            if bgr is None:
                time.sleep(0.05)
                continue
            obs = tracker.update(bgr)
            candidates = detect_rings(bgr, params)
            mask = blue_mask(bgr, params)
            _write_atomic(d / "raw.jpg", _jpg(bgr, 85))
            _write_atomic(d / "overlay.jpg", _jpg(draw_overlay(bgr, candidates, obs.raw if obs.fresh else None, obs)))
            _write_atomic(d / "mask.jpg", _jpg(mask, 70))

            req = _read_json(d / "autofit.json") if (d / "autofit.json").exists() else None
            if req is not None:
                try:
                    if req.get("clear"):
                        ring_px, reject_px = ring_px[:0], reject_px[:0]
                    if not req.get("ring") and not req.get("reject"):
                        raise LookupError("samples forgotten")
                    pos, neg = sample_hsv(bgr, req.get("ring", []), req.get("reject", []))
                    ring_px, reject_px = np.concatenate([ring_px, pos]), np.concatenate([reject_px, neg])
                    fitted, stats = fit_hsv_pixels(ring_px, reject_px, base=params)
                    params_file.write_text(json.dumps(fitted.to_json(), indent=2), encoding="utf-8")
                    result = dict(ok=True, params=fitted.to_json(), **stats)
                except LookupError as e:                     # only a reset was asked for
                    result = dict(ok=True, cleared=True, note=str(e), params=params.to_json())
                except Exception as e:                       # report it to the panel, keep running
                    result = dict(ok=False, error=str(e))
                _write_atomic(d / "autofit_result.json", json.dumps(result).encode())
                (d / "autofit.json").unlink(missing_ok=True)
            if (d / "snapshot.flag").exists():
                snaps = d / "snapshots"
                snaps.mkdir(exist_ok=True)
                cv2.imwrite(str(snaps / f"{datetime.now():%Y%m%d-%H%M%S}.jpg"), bgr)
                (d / "snapshot.flag").unlink(missing_ok=True)

            frames += 1
            if time.monotonic() - t_fps >= 1.0:
                fps, frames, t_fps = frames / (time.monotonic() - t_fps), 0, time.monotonic()
            state = {}
            drone = getattr(src, "_drone", None)
            if drone is not None:
                try:
                    st = drone.get_current_state()
                    state = {k: st.get(k) for k in ("bat", "temph", "h")}
                except Exception:
                    pass
            pick = obs.raw if obs.fresh else None
            status = dict(
                t=datetime.now().isoformat(timespec="seconds"), source=a.source, fps=round(fps, 1),
                frame=[int(bgr.shape[1]), int(bgr.shape[0])], found=bool(obs.found),
                candidates=len(candidates),
                range_m=(round(float(obs.range_m), 2) if obs.found and np.isfinite(obs.range_m) else None),
                center=([round(obs.cx, 3), round(obs.cy, 3)] if obs.found else None),
                arc_deg=(pick or {}).get("arc_deg"), clipped=bool((pick or {}).get("clipped", False)),
                mask_fraction=round(float(obs.blue_fraction), 3), params=params.to_json(),
                samples=dict(ring_px=int(len(ring_px)), reject_px=int(len(reject_px))),
                image=image_name, **state)
            _write_atomic(d / "status.json", json.dumps(status).encode())
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
    final = _read_json(d / "status.json") or {}
    _write_atomic(d / "status.json", json.dumps(dict(final, ended=True)).encode())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--source", choices=("tello", "webcam", "image"), default="tello")
    ap.add_argument("--webcam", type=int, default=0)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--preset", default=None, help="start from this preset (default: detector.json)")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--max-minutes", type=float, default=30.0, help="ends by itself after this long")
    a = ap.parse_args(argv)
    if a.source == "image" and (a.image is None or not a.image.is_file()):
        ap.error("--source image needs --image <file>")
    run(a)


if __name__ == "__main__":
    main()
