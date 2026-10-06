"""
Measure where the hoop is, once, before flying, so nobody has to type coordinates by hand.

    python hardware/calibration/hoop_locator.py            # camera finds the hoop marker (id from config)
    python hardware/calibration/hoop_locator.py --manual 0.4 -0.3 1.0

Place the hoop marker FLAT ON THE FLOOR directly under the hoop centre (the overhead camera cannot measure
height, so z comes from config `hoop.z`, or --z). The script averages N frames, rejects a noisy reading
(xy std > hoop.xy_std_max_m), checks the position against the range the policy was trained on, and writes
flights/hoop.json with a timestamp. run_policy.py / keyboard_teleop.py --record read it (max age
`hoop.max_age_s`, default 10 min) and ask you to confirm. calibration/validate_hoop.py shows it on the camera image.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
for p in (HW, HW / "control"):
    sys.path.insert(0, str(p))
from common import constants as C                           # noqa: E402
from common.config import FLIGHTS_DIR, load_effective_config   # noqa: E402

HOOP_FILE = FLIGHTS_DIR / "hoop.json"


def aggregate_hoop(xy_samples, z, max_std=0.02, min_samples=10, force=False):
    """-> (xyz or None, std (m), [problems]). Pure: unit-tested without a camera."""
    xy = np.asarray(xy_samples, dtype=float).reshape(-1, 2)
    problems = []
    if len(xy) < min_samples:
        return None, float("nan"), [f"only {len(xy)} detections (need {min_samples}); is the marker visible?"]
    med = np.median(xy, axis=0)
    keep = np.linalg.norm(xy - med, axis=1) < 0.05                 # drop stray detections of other things
    xy = xy[keep]
    if len(xy) < min_samples:
        return None, float("nan"), ["detections too scattered to trust"]
    mean = xy.mean(axis=0)
    std = float(np.max(xy.std(axis=0)))
    if std > max_std:
        problems.append(f"reading is noisy: std {std * 100:.1f} cm > {max_std * 100:.1f} cm (lighting? marker moving?)")
    xyz = np.array([mean[0], mean[1], z])
    if np.any(np.abs(xyz[:2]) > C.HOOP_XY_MAX + 1e-9):
        problems.append(f"xy {xyz[:2].round(2).tolist()} is outside +-{C.HOOP_XY_MAX.tolist()} m (not seen in training)")
    if not (C.HOOP_Z_RANGE[0] <= z <= C.HOOP_Z_RANGE[1]):
        problems.append(f"z {z:.2f} is outside {C.HOOP_Z_RANGE}")
    if problems and not force:
        return None, std, problems
    return xyz, std, problems


def write_hoop(xyz, std, source):
    HOOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOOP_FILE.write_text(json.dumps({"xyz": [round(float(v), 4) for v in xyz], "std_m": round(float(std), 4),
                                     "source": source, "unix_time": time.time(),
                                     "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    return HOOP_FILE


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--z", type=float, default=None, help="hoop height (m); default from config hoop.z")
    ap.add_argument("--camera-index", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_effective_config()
    h = cfg["hoop"]
    z = a.z if a.z is not None else float(h["z"])

    if a.manual:
        xyz = np.array(a.manual, dtype=float)
        problems = []
        if np.any(np.abs(xyz[:2]) > C.HOOP_XY_MAX + 1e-9) or not (C.HOOP_Z_RANGE[0] <= xyz[2] <= C.HOOP_Z_RANGE[1]):
            problems.append("outside the range the policy was trained on")
        if problems and not a.force:
            sys.exit("[hoop] " + "; ".join(problems) + "  (use --force)")
        print("[hoop] wrote", write_hoop(xyz, 0.0, "manual"))
        return xyz

    from state_estimation.aruco_tracker import OverheadCamera, apply_homography
    cam = OverheadCamera(cfg, HW, a.camera_index)
    hoop_id = int(cfg["aruco"]["hoop_id"])
    samples = []
    try:
        t_end = time.time() + 20
        while len(samples) < h["frames"] and time.time() < t_end:
            time.sleep(0.05)
            m, H = cam.last_markers, cam.floor.H
            if H is not None and hoop_id in m:
                samples.append(apply_homography(H, [m[hoop_id].mean(0)])[0])
    finally:
        cam.close()
    xyz, std, problems = aggregate_hoop(samples, z, h["xy_std_max_m"], force=a.force)
    for p in problems:
        print("  !", p)
    if xyz is None:
        sys.exit("[hoop] no trustworthy hoop position; nothing written")
    print(f"[hoop] x={xyz[0]:+.3f} y={xyz[1]:+.3f} z={xyz[2]:+.2f} m (std {std * 100:.1f} cm, {len(samples)} frames)")
    print("[hoop] wrote", write_hoop(xyz, std, f"aruco id {hoop_id}"))
    return xyz


if __name__ == "__main__":
    main()
