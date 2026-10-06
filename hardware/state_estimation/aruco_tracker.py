"""
Overhead camera -> cage-frame drone position via ArUco markers.

Geometry (all plain numpy, unit-tested without OpenCV):
  - Reference markers lie flat on the floor at known cage-frame (x, y). Their image centres give a
    homography image -> floor plane (DLT, least squares over >= 4 markers).
  - The drone's marker is ~0.8 m above the floor, so its floor-plane projection is pushed outward
    from the camera's nadir. With the camera height H and the drone height z (from the filter),
    true_xy = nadir + (floor_xy - nadir) * (H - z) / H.
  - Marker yaw = direction from the marker's bottom edge to its top edge, in floor coordinates.
    Mount the drone marker with its TOP edge toward the drone's nose; yaw 0 = cage +x, CCW positive.

OpenCV (cv2) is imported lazily, only by the classes that touch a physical camera.
Frames are stamped with time.monotonic() at grab(), before decode: that stamp is what the filter
corrects for latency.
"""
import json
import math
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .est_types import CameraDetection


# -- geometry -------------------------------------------------------------------------------
def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear transform. src, dst: (N, 2), N >= 4. Returns H with dst ~ H @ src (homogeneous)."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    if len(src) < 4:
        raise ValueError("need at least 4 point pairs")
    # normalise for conditioning
    def norm(p):
        c = p.mean(0)
        s = math.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return (np.hstack([p, np.ones((len(p), 1))]) @ T.T)[:, :2], T
    s_n, Ts = norm(src)
    d_n, Td = norm(dst)
    A = []
    for (x, y), (u, v) in zip(s_n, d_n):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.array(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def height_correct(xy_floor, z, nadir_xy, cam_height) -> np.ndarray:
    """Undo perspective for a marker at height z: scale its floor-plane offset from the camera nadir."""
    xy_floor, nadir_xy = np.asarray(xy_floor, dtype=float), np.asarray(nadir_xy, dtype=float)
    k = (cam_height - z) / cam_height
    return nadir_xy + (xy_floor - nadir_xy) * k


def marker_yaw(corners_px: np.ndarray, H: np.ndarray) -> float:
    """corners_px: (4, 2) in ArUco order (top-left, top-right, bottom-right, bottom-left)."""
    f = apply_homography(H, corners_px)
    top = 0.5 * (f[0] + f[1])
    bottom = 0.5 * (f[3] + f[2])
    d = top - bottom                                         # bottom edge -> top edge = the nose direction
    return math.atan2(d[1], d[0])


def marker_quality(corners_px: np.ndarray, nominal_side_px=40.0) -> float:
    side = np.mean([np.linalg.norm(corners_px[i] - corners_px[(i + 1) % 4]) for i in range(4)])
    return float(np.clip(side / nominal_side_px, 0.1, 1.0))


class FloorMap:
    """Homography from reference markers, refined over frames (the camera is static)."""

    def __init__(self, reference: Dict[int, Tuple[float, float]], keep=30):
        self.reference = {int(k): np.array(v, dtype=float) for k, v in reference.items()}
        self.H: Optional[np.ndarray] = None
        self._hist: List[np.ndarray] = []
        self.keep = keep
        self.nadir_xy: Optional[np.ndarray] = None
        self.image_size: Optional[Tuple[int, int]] = None

    def update(self, centres_px: Dict[int, np.ndarray], image_size=None) -> bool:
        ids = [i for i in centres_px if i in self.reference]
        if len(ids) < 4:
            return self.H is not None
        src = np.array([centres_px[i] for i in ids])
        dst = np.array([self.reference[i] for i in ids])
        H = fit_homography(src, dst)
        self._hist.append(H)
        self._hist = self._hist[-self.keep:]
        self.H = np.median(np.stack(self._hist), axis=0)       # robust to a bad frame
        if image_size is not None:
            self.image_size = tuple(image_size)
            self.nadir_xy = apply_homography(self.H, [[image_size[0] / 2, image_size[1] / 2]])[0]
        return True


# -- OpenCV-facing parts --------------------------------------------------------------------------
def _load_intrinsics(path: Path):
    if not path.exists():
        return None, None
    d = json.loads(path.read_text())
    return np.array(d["camera_matrix"], dtype=float), np.array(d["dist_coeffs"], dtype=float)


class ArucoDetector:
    def __init__(self, dictionary="DICT_4X4_50"):
        import cv2
        self.cv2 = cv2
        self.aruco = cv2.aruco
        self.dict = self.aruco.getPredefinedDictionary(getattr(self.aruco, dictionary))
        if hasattr(self.aruco, "ArucoDetector"):
            self._det = self.aruco.ArucoDetector(self.dict, self.aruco.DetectorParameters())
        else:                                                    # OpenCV < 4.7
            self._det = None
            self._params = self.aruco.DetectorParameters_create()

    def detect(self, gray) -> Dict[int, np.ndarray]:
        if self._det is not None:
            corners, ids, _ = self._det.detectMarkers(gray)
        else:
            corners, ids, _ = self.aruco.detectMarkers(gray, self.dict, parameters=self._params)
        out = {}
        if ids is not None:
            for c, i in zip(corners, ids.reshape(-1)):
                out[int(i)] = np.asarray(c, dtype=float).reshape(4, 2)
        return out


class OverheadCamera:
    """Background thread: grab -> stamp -> detect -> CameraDetection. `poll(now)` drains the queue."""

    def __init__(self, cfg: dict, hw_root: Path, camera_index=None, show=False):
        import cv2
        self.cv2 = cv2
        self.cfg = cfg
        a, c = cfg["aruco"], cfg["camera"]
        self.drone_id = int(a["drone_id"])
        self.cam_height = float(c["height_m"])
        self.detector = ArucoDetector(a["dictionary"])
        self.floor = FloorMap({int(k): tuple(v) for k, v in a["reference"].items()})
        self.K, self.dist = _load_intrinsics(hw_root / c["intrinsics_file"])
        if self.K is None:
            print("[camera] WARNING: no camera intrinsics found; run calibration/calibrate_camera.py "
                  "(positions will be less accurate, especially near the image edges)")
        src = c["index"] if camera_index is None else camera_index
        # an int is a webcam index; a path string plays a recorded video (handy for tests and offline replays)
        self.cap = cv2.VideoCapture(src if isinstance(src, str) else int(src))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {src!r}")
        self._q: "queue.SimpleQueue[CameraDetection]" = queue.SimpleQueue()
        self._z_hint = 0.8
        self._stop = threading.Event()
        self._show = show
        self.last_frame = None
        self.last_markers: Dict[int, np.ndarray] = {}
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def set_height_hint(self, z):
        if z is not None:
            self._z_hint = float(z)

    def poll(self, now) -> List[CameraDetection]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out

    def _undistort(self, pts):
        if self.K is None:
            return pts
        u = self.cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float32), self.K, self.dist, P=self.K)
        return u.reshape(-1, 2).astype(float)

    def _run(self):
        cv2 = self.cv2
        while not self._stop.is_set():
            if not self.cap.grab():                              # stamp BEFORE decode
                time.sleep(0.005)
                continue
            t = time.monotonic()
            ok, frame = self.cap.retrieve()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            markers = self.detector.detect(gray)
            markers = {i: self._undistort(c) for i, c in markers.items()}
            self.last_frame, self.last_markers = frame, markers
            h, w = gray.shape[:2]
            self.floor.update({i: c.mean(0) for i, c in markers.items()}, (w, h))
            if self.floor.H is None or self.drone_id not in markers or self.floor.nadir_xy is None:
                continue
            c = markers[self.drone_id]
            floor_xy = apply_homography(self.floor.H, [c.mean(0)])[0]
            xy = height_correct(floor_xy, self._z_hint, self.floor.nadir_xy, self.cam_height)
            self._q.put(CameraDetection(t=t, x=float(xy[0]), y=float(xy[1]),
                                        yaw=marker_yaw(c, self.floor.H), quality=marker_quality(c)))

    def close(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self.cap.release()
