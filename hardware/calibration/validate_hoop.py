"""
Look at what the system believes, before flying: live overhead image with the cage reference markers,
the detected drone marker, and the stored hoop position projected back onto the picture.

    python hardware/calibration/validate_hoop.py            # Q to quit
    python hardware/calibration/validate_hoop.py --check    # no window: print hoop.json and its age

Green circle = hoop (from flights/hoop.json), blue = drone marker position, red outline = cage footprint.
If the hoop circle is not on the real hoop's floor spot, do not fly: re-run hoop_locator.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW))
from common.config import FLIGHTS_DIR, load_effective_config   # noqa: E402

HOOP_FILE = FLIGHTS_DIR / "hoop.json"


def describe_hoop_file():
    if not HOOP_FILE.exists():
        return "no hoop.json (run calibration/hoop_locator.py)"
    d = json.loads(HOOP_FILE.read_text())
    age = time.time() - d["unix_time"]
    return f"hoop {d['xyz']} (std {d['std_m'] * 100:.1f} cm, {d['source']}), {age:.0f} s old"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--camera-index", type=int, default=None)
    a = ap.parse_args(argv)
    print("[hoop]", describe_hoop_file())
    if a.check:
        return
    import cv2
    from state_estimation.aruco_tracker import OverheadCamera
    cfg = load_effective_config()
    cam = OverheadCamera(cfg, HW, a.camera_index)
    try:
        while True:
            time.sleep(0.03)
            frame = None if cam.last_frame is None else cam.last_frame.copy()
            if frame is None:
                continue
            H = cam.floor.H
            if H is not None:
                Hinv = np.linalg.inv(H)

                def to_px(xy):
                    p = Hinv @ np.array([xy[0], xy[1], 1.0])
                    return int(p[0] / p[2]), int(p[1] / p[2])
                ref = [tuple(v) for v in cfg["aruco"]["reference"].values()]
                pts = np.array([to_px(v) for v in ref], np.int32)
                cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
                if HOOP_FILE.exists():
                    xyz = json.loads(HOOP_FILE.read_text())["xyz"]
                    cv2.circle(frame, to_px(xyz[:2]), 18, (0, 255, 0), 3)
                    cv2.putText(frame, "hoop", to_px(xyz[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                did = int(cfg["aruco"]["drone_id"])
                if did in cam.last_markers:
                    c = cam.last_markers[did].mean(0)
                    cv2.circle(frame, (int(c[0]), int(c[1])), 14, (255, 128, 0), 3)
            cv2.imshow("validate_hoop (Q quits)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
