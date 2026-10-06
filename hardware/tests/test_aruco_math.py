"""Camera geometry without OpenCV: a synthetic overhead pinhole camera looks down on the cage."""
import math

import numpy as np

from state_estimation.aruco_tracker import (FloorMap, apply_homography, fit_homography, height_correct,
                                            marker_quality, marker_yaw)


def make_camera(height=3.0, f=900.0, size=(1280, 720), tilt=0.0, centre=(0.1, -0.05)):
    """Pinhole camera at (cx, cy, height) looking straight down. Returns project(world_xyz) -> pixel."""
    cx, cy = centre
    w, h = size

    def project(p):
        p = np.atleast_2d(np.asarray(p, dtype=float))
        depth = height - p[:, 2]
        u = w / 2 + f * (p[:, 0] - cx) / depth
        v = h / 2 - f * (p[:, 1] - cy) / depth            # image y points down, world y points up
        return np.stack([u, v], axis=1)
    return project, np.array(centre), size


REFS = {0: (-1.0, -1.0), 1: (1.0, -1.0), 2: (1.0, 1.0), 3: (-1.0, 1.0)}


def test_homography_recovers_floor_plane_mapping():
    proj, nadir, size = make_camera()
    src = proj(np.array([[x, y, 0.0] for x, y in REFS.values()]))
    H = fit_homography(src, np.array(list(REFS.values())))
    test = np.array([[0.3, 0.4], [-0.7, 0.2], [0.9, -0.8]])
    back = apply_homography(H, proj(np.hstack([test, np.zeros((3, 1))])))
    assert np.allclose(back, test, atol=1e-6)


def test_height_correction_recovers_airborne_marker_position():
    proj, nadir, size = make_camera()
    src = proj(np.array([[x, y, 0.0] for x, y in REFS.values()]))
    fm = FloorMap(REFS)
    assert fm.update({i: p for i, p in zip(REFS, src)}, size)
    assert np.allclose(fm.nadir_xy, nadir, atol=1e-6)                 # image centre looks at the camera nadir
    for true in ([0.5, -0.3, 0.8], [-0.6, 0.7, 1.4], [0.0, 0.0, 0.1]):
        px = proj(np.array([true]))[0]
        floor = apply_homography(fm.H, [px])[0]
        uncorrected_err = np.linalg.norm(floor - true[:2])
        xy = height_correct(floor, true[2], fm.nadir_xy, 3.0)
        assert np.allclose(xy, true[:2], atol=1e-6), (true, xy)
        if true[2] > 0.5:
            assert uncorrected_err > 0.05                              # the correction is not a no-op


def test_marker_yaw_ccw_from_plus_x():
    proj, nadir, size = make_camera()
    src = proj(np.array([[x, y, 0.0] for x, y in REFS.values()]))
    fm = FloorMap(REFS)
    fm.update({i: p for i, p in zip(REFS, src)}, size)
    for yaw_deg in (0, 30, 90, -75, 170):
        yaw = math.radians(yaw_deg)
        half = 0.035
        # marker corners in the drone frame (top edge toward the nose): TL, TR, BR, BL; nose = +x_body
        local = np.array([[half, half], [half, -half], [-half, -half], [-half, half]])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        world = local @ R.T + np.array([0.2, 0.1])
        px = proj(np.hstack([world, np.full((4, 1), 0.0)]))
        est = marker_yaw(px, fm.H)
        d = (est - yaw + math.pi) % (2 * math.pi) - math.pi
        assert abs(d) < 1e-6, (yaw_deg, math.degrees(est))


def test_floor_map_needs_four_references_and_is_robust_to_one_bad_frame():
    proj, nadir, size = make_camera()
    src = proj(np.array([[x, y, 0.0] for x, y in REFS.values()]))
    fm = FloorMap(REFS)
    assert not fm.update({0: src[0], 1: src[1], 2: src[2]}, size)
    for _ in range(9):
        fm.update({i: p for i, p in zip(REFS, src)}, size)
    bad = src.copy()
    bad[2] += 60                                                      # one frame with a mislocated corner
    fm.update({i: p for i, p in zip(REFS, bad)}, size)
    probe = proj(np.array([[0.4, 0.4, 0.0]]))[0]
    assert np.allclose(apply_homography(fm.H, [probe])[0], [0.4, 0.4], atol=0.02)


def test_marker_quality_scales_with_size():
    small = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
    big = small * 6
    assert marker_quality(small) < marker_quality(big) <= 1.0
