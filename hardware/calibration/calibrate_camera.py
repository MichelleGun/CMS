"""
Camera intrinsics (lens distortion) from a printed chessboard.

    python hardware/calibration/calibrate_camera.py --cols 9 --rows 6 --square-mm 25

Hold the printed chessboard in view at different positions / tilts. SPACE saves a frame when the board is
detected (aim for 15+ spread over the whole image, especially the corners), C computes and saves, Q quits.
Writes hardware/calibration/camera_intrinsics.json (small; fine to commit). Use the same camera index and
resolution as flights (config.yaml `camera`).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW))
from common.config import load_effective_config      # noqa: E402


def calibrate(obj_points, img_points, size):
    import cv2
    err, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return err, K, dist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--camera-index", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        import cv2
    except ImportError:
        sys.exit("opencv is needed: pip install opencv-contrib-python")
    cfg = load_effective_config()
    c = cfg["camera"]
    cap = cv2.VideoCapture(c["index"] if a.camera_index is None else a.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
    pattern = (a.cols, a.rows)
    objp = np.zeros((a.rows * a.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * (a.square_mm / 1000.0)
    obj_pts, img_pts, size = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern)
        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern, corners, found)
        cv2.putText(vis, f"saved {len(obj_pts)}  SPACE save  C compute  Q quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibrate_camera", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" ") and found:
            sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_pts.append(objp)
            img_pts.append(sub)
        elif k == ord("c") and len(obj_pts) >= 10:
            err, K, dist = calibrate(obj_pts, img_pts, size)
            out = HW / c["intrinsics_file"]
            out.write_text(json.dumps({"camera_matrix": K.tolist(), "dist_coeffs": dist.tolist(),
                                       "image_size": list(size), "rms_reprojection_px": float(err)}, indent=2))
            print(f"saved {out}   RMS reprojection error {err:.3f} px (below ~0.5 px is good)")
            break
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
