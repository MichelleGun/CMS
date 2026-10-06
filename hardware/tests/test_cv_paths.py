"""
The OpenCV / pygame code paths, exercised against a synthetic overhead scene rendered with real ArUco markers.
Skipped automatically where cv2 / pygame are not installed (they are optional for the rest of the suite).
"""
import math
from pathlib import Path
import os
import time

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

from common.config import load_config
from state_estimation.aruco_tracker import (ArucoDetector, FloorMap, OverheadCamera, apply_homography,
                                            height_correct, marker_yaw)
from calibration import gen_markers

CFG = load_config()
A = CFG["aruco"]
CAM_H, F, SIZE, NADIR = 3.6, 900.0, (1280, 720), np.array([0.1, -0.05])


def project(p):
    p = np.atleast_2d(np.asarray(p, dtype=float))
    depth = CAM_H - p[:, 2]
    return np.stack([SIZE[0] / 2 + F * (p[:, 0] - NADIR[0]) / depth, SIZE[1] / 2 - F * (p[:, 1] - NADIR[1]) / depth], 1)


def marker_img(marker_id, px=200):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, A["dictionary"]))
    m = cv2.aruco.generateImageMarker(d, marker_id, px)
    out = np.full((px + 2 * (px // 5), px + 2 * (px // 5)), 255, np.uint8)
    b = px // 5
    out[b:b + px, b:b + px] = m
    return out, b, px


def place(canvas, marker_id, centre_xy, z, side, yaw):
    """Draw a marker lying flat at (x, y, z) with its TOP edge toward the heading `yaw` (CCW from +x)."""
    img, b, px = marker_img(marker_id)
    full = img.shape[0]
    half = side / 2 * full / px                                  # outer border included
    local = np.array([[half, half], [half, -half], [-half, -half], [-half, half]])   # TL, TR, BR, BL (nose = +x_body)
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    world = local @ R.T + np.asarray(centre_xy)
    dst = project(np.hstack([world, np.full((4, 1), z)])).astype(np.float32)
    src = np.array([[0, 0], [full, 0], [full, full], [0, full]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, SIZE, flags=cv2.INTER_LINEAR, borderValue=255)
    mask = cv2.warpPerspective(np.full_like(img, 255), H, SIZE, flags=cv2.INTER_NEAREST, borderValue=0)
    canvas[mask > 0] = warped[mask > 0]


def scene(drone_xy, drone_z, drone_yaw, with_hoop=None):
    canvas = np.full((SIZE[1], SIZE[0]), 200, np.uint8)
    for mid, xy in A["reference"].items():
        place(canvas, int(mid), xy, 0.0, 0.20, 0.0)
    place(canvas, int(A["drone_id"]), drone_xy, drone_z, 0.10, drone_yaw)
    if with_hoop is not None:
        place(canvas, int(A["hoop_id"]), with_hoop, 0.0, 0.18, 0.0)
    return canvas


def run_pipeline(img, z):
    det = ArucoDetector(A["dictionary"])
    markers = det.detect(img)
    fm = FloorMap({int(k): tuple(v) for k, v in A["reference"].items()})
    assert fm.update({i: c.mean(0) for i, c in markers.items()}, SIZE), f"saw markers {sorted(markers)}"
    c = markers[int(A["drone_id"])]
    xy = height_correct(apply_homography(fm.H, [c.mean(0)])[0], z, fm.nadir_xy, CAM_H)
    return xy, marker_yaw(c, fm.H), markers, fm


@pytest.mark.parametrize("xy,z,yaw_deg", [((0.3, 0.2), 0.8, 0), ((-0.5, 0.6), 1.2, 90), ((0.0, -0.4), 0.5, -60), ((0.6, -0.2), 1.5, 170)])
def test_real_aruco_detection_recovers_position_height_and_yaw(xy, z, yaw_deg):
    yaw = math.radians(yaw_deg)
    est_xy, est_yaw, markers, fm = run_pipeline(scene(xy, z, yaw), z)
    assert {0, 1, 2, 3, 10} <= set(markers)
    assert np.linalg.norm(est_xy - np.array(xy)) < 0.02, (est_xy, xy)
    assert abs(math.degrees((est_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)) < 4.0


def test_gen_markers_writes_detectable_pngs(tmp_path, monkeypatch):
    monkeypatch.setattr(gen_markers, "OUT", tmp_path)
    gen_markers.main()
    files = sorted(tmp_path.glob("*.png"))
    assert len(files) == len(A["reference"]) + 2
    det = ArucoDetector(A["dictionary"])
    for f in files:
        mid = int(f.name[:2])
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        img[:100, :] = 255                                              # blank the caption strip
        assert mid in det.detect(img), f.name


def test_overhead_camera_thread_from_a_recorded_video(tmp_path):
    """Full OverheadCamera: grab/stamp/detect/undistort-off/height-correct, reading a video file."""
    path = str(tmp_path / "overhead.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, SIZE, True)
    truth = (0.35, -0.25, 0.9, math.radians(30))
    img = cv2.cvtColor(scene(truth[:2], truth[2], truth[3]), cv2.COLOR_GRAY2BGR)
    for _ in range(40):
        w.write(img)
    w.release()
    cfg = load_config(overrides={"camera": {"height_m": CAM_H, "intrinsics_file": "no_such_file.json"}})
    cam = OverheadCamera(cfg, Path(__file__).parent, camera_index=path)
    cam.set_height_hint(truth[2])
    dets = []
    t0 = time.time()
    while time.time() - t0 < 5 and len(dets) < 10:
        time.sleep(0.05)
        dets += cam.poll(time.monotonic())
    cam.close()
    assert len(dets) >= 5, "no detections came out of the camera thread"
    d = dets[-1]
    # the first frames may have been processed before the height hint, so judge the last one
    assert math.hypot(d.x - truth[0], d.y - truth[1]) < 0.03
    assert abs(math.degrees((d.yaw - truth[3] + math.pi) % (2 * math.pi) - math.pi)) < 4
    assert all(b.t >= a.t for a, b in zip(dets, dets[1:]))              # monotonic grab-time stamps
    assert 0.1 <= d.quality <= 1.0


def test_pygame_input_maps_keys_and_renders_headless():
    pygame = pytest.importorskip("pygame")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from control.inputs import PygameInput
    inp = PygameInput("test", size=(400, 200))
    try:
        for key in (pygame.K_t, pygame.K_l, pygame.K_x, pygame.K_BACKSPACE, pygame.K_SPACE, pygame.K_ESCAPE,
                    pygame.K_1, pygame.K_3, pygame.K_r):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0))
        st = inp.poll()
        assert st.pressed == ["takeoff", "land", "kill", "kill", "hold", "quit", "speed1", "speed3", "record"]
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert "quit" in inp.poll().pressed
        inp.show(["FLYING  MANUAL", "line two"], alert=False)
        inp.show(["LANDING *** fence_hard ***", "x"], alert=True)
        assert inp.poll().held == set()                                  # nothing physically pressed
    finally:
        inp.close()


def test_drone_hiding_a_corner_marker_does_not_break_tracking():
    """Seen from above, a high drone near a corner covers that corner's floor marker. The homography is
    learned while all references are visible and then retained; a frame with a marker hidden must not degrade it."""
    xy, z = (0.6, -0.6), 1.5
    det = ArucoDetector(A["dictionary"])
    fm = FloorMap({int(k): tuple(v) for k, v in A["reference"].items()})
    clear = det.detect(scene((0.0, 0.0), 0.8, 0.0))
    assert fm.update({i: c.mean(0) for i, c in clear.items()}, SIZE)
    H_before = fm.H.copy()
    img = scene(xy, z, math.radians(170))
    seen = det.detect(img)
    assert 1 not in seen and 10 in seen                              # corner 1 is covered by the drone's marker
    assert fm.update({i: c.mean(0) for i, c in seen.items()}, SIZE)  # keeps working on the retained homography
    assert np.allclose(fm.H, H_before)
    est = height_correct(apply_homography(fm.H, [seen[10].mean(0)])[0], z, fm.nadir_xy, CAM_H)
    assert np.linalg.norm(est - np.array(xy)) < 0.02
