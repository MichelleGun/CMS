"""
Tune the ring detector against the real camera and the real ring. GROUND ONLY.

    python hardware/onboard/tune_detector.py                 # live, from the drone
    python hardware/onboard/tune_detector.py --save           # write detector.json
    python hardware/onboard/tune_detector.py --image shot.jpg # from a saved frame
    python hardware/onboard/tune_detector.py --webcam 0       # any USB camera

This is stage 1 of the bring-up and it is not optional. The detector's thresholds have
never been validated against a real camera frame in any lighting -- the kit's own
docstring says so ("HSV thresholds need validation against the actual cage lighting"),
and its defaults additionally assume a near-perfect circle, which a pool noodle bent into
a ring is not. Flying before tuning means flying a detector nobody has seen work.

It sends NO flight commands. It connects, starts the video stream, and shows the mask and
the candidates with sliders. Nothing spins.

What to check before saving:
  * the ring is outlined, and nothing else is;
  * the outline survives being moved to the edge of the frame (that is the `clipped`
    path, which the approach depends on);
  * the printed range roughly matches a tape measure at two distances. If it is off by a
    constant factor, the error is --ring-opening/--ring-noodle or --hfov, not the
    thresholds: range scales linearly with the assumed outer diameter and with the
    assumed focal length.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HW_ROOT = Path(__file__).resolve().parents[1]
if str(HW_ROOT) not in sys.path:
    sys.path.insert(0, str(HW_ROOT))

# flake8: noqa: E402
from onboard import spec
from onboard.ring_detect import DetectorParams, RingTracker, blue_mask, detect_rings

DETECTOR_FILE = Path(__file__).resolve().parent / "detector.json"
WINDOW = "ring detector tuning (ground only)"

# (slider label, DetectorParams field, index within the field or None, max value)
SLIDERS = [
    ("hue lo", "hsv_lo", 0, 179),
    ("hue hi", "hsv_hi", 0, 179),
    ("sat lo", "hsv_lo", 1, 255),
    ("val lo", "hsv_lo", 2, 255),
    ("residual x100", "residual_max", None, 100),
    ("min major px", "min_major_px", None, 200),
    ("min arc deg", "min_arc_deg", None, 360),
    ("fill lo x100", "fill_ratio", 0, 200),
    ("fill hi x100", "fill_ratio", 1, 200),
    ("open lo x100", "opening_ratio", 0, 100),
]


def _get(params, field, index):
    v = getattr(params, field)
    return v[index] if index is not None else v


def _set(params, field, index, value):
    if index is None:
        setattr(params, field, value)
        return
    cur = list(getattr(params, field))
    cur[index] = value
    setattr(params, field, tuple(cur))


def build_sliders(cv2, params):
    for label, field, index, hi in SLIDERS:
        v = _get(params, field, index)
        scaled = int(round(v * 100)) if "x100" in label else int(round(v))
        cv2.createTrackbar(label, WINDOW, min(max(scaled, 0), hi), hi, lambda _v: None)


def read_sliders(cv2, params):
    for label, field, index, _hi in SLIDERS:
        raw = cv2.getTrackbarPos(label, WINDOW)
        val = raw / 100.0 if "x100" in label else raw
        _set(params, field, index, val)
    # Keep the paired bounds ordered, or inRange silently matches nothing.
    if params.hsv_hi[0] < params.hsv_lo[0]:
        _set(params, "hsv_hi", 0, params.hsv_lo[0])
    if params.fill_ratio[1] < params.fill_ratio[0]:
        _set(params, "fill_ratio", 1, params.fill_ratio[0])
    return params


class Source:
    """A frame source: the drone's camera, a webcam, or one still image."""

    def __init__(self, args):
        self.args = args
        self.kind = "image" if args.image else ("webcam" if args.webcam is not None
                                                else "tello")
        self._img = None
        self._cap = None
        self._drone = None
        self._reader = None
        self._ctx = None

    def __enter__(self):
        import cv2
        a = self.args
        if self.kind == "image":
            self._img = cv2.imread(str(a.image))
            if self._img is None:
                raise SystemExit(f"cannot read {a.image}")
        elif self.kind == "webcam":
            self._cap = cv2.VideoCapture(int(a.webcam))
            if not self._cap.isOpened():
                raise SystemExit(f"cannot open webcam {a.webcam}")
        else:
            from onboard.kit import connected, create_drone, load_settings
            settings = load_settings(a.config) if a.config else load_settings()
            self._drone = create_drone(settings)
            self._ctx = connected(self._drone)
            self._ctx.__enter__()
            print(f"connected; battery {self._drone.get_battery()}%")
            try:
                self._drone.set_video_bitrate(1)
            except Exception as e:
                print(f"(bitrate not set: {e})")
            self._drone.streamon()
            self._reader = self._drone.get_frame_read(with_queue=True, max_queue_len=2)
            end = time.monotonic() + 8
            while self._reader.frame is None:
                if time.monotonic() > end:
                    raise SystemExit("no camera frame within 8 s")
                time.sleep(0.05)
        return self

    def read(self):
        import cv2
        if self.kind == "image":
            return self._img.copy()
        if self.kind == "webcam":
            ok, frame = self._cap.read()
            return frame if ok else None
        rgb = self._reader.frame
        return None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()
        if self._reader is not None:
            try:
                self._reader.stop()
            except Exception:
                pass
        if self._drone is not None:
            try:
                self._drone.streamoff()
            except Exception:
                pass
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, help="tune against a saved frame instead")
    ap.add_argument("--webcam", type=int, default=None, help="use this webcam index")
    ap.add_argument("--save", action="store_true",
                    help="write detector.json on exit (S also saves at any time)")
    ap.add_argument("--out", type=Path, default=DETECTOR_FILE)
    ap.add_argument("--ring-opening", type=float, default=spec.RING_OPENING_M)
    ap.add_argument("--ring-noodle", type=float, default=spec.RING_NOODLE_M)
    ap.add_argument("--hfov", type=float, default=spec.CAMERA_HFOV_DEG)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--no-window", action="store_true",
                    help="print detections without a GUI (for a headless check)")
    a = ap.parse_args(argv)

    import cv2
    params = DetectorParams()
    if a.out.is_file():
        try:
            params = DetectorParams.from_json(json.loads(a.out.read_text(encoding="utf-8")))
            print(f"starting from {a.out}")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    outer = a.ring_opening + 2 * a.ring_noodle
    tracker = RingTracker(params, ring_diameter_m=outer, hfov_deg=a.hfov)
    print(f"ring outer diameter {outer*100:.0f} cm (opening {a.ring_opening*100:.0f} + "
          f"2 x noodle {a.ring_noodle*100:.0f}), hfov {a.hfov:.0f} deg")
    print("keys: S save, R reset tracker, Q/Esc quit")

    if not a.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        build_sliders(cv2, params)

    saved = False
    with Source(a) as src:
        last_print = 0.0
        while True:
            bgr = src.read()
            if bgr is None:
                time.sleep(0.02)
                continue
            if not a.no_window:
                read_sliders(cv2, params)
            tracker.params = params
            tracker.ring_diameter_m = outer
            tracker.hfov_deg = a.hfov

            cands = detect_rings(bgr, params)
            ring = tracker.update(bgr)
            mask = blue_mask(bgr, params)

            now = time.monotonic()
            if now - last_print > 0.5:
                last_print = now
                if cands:
                    c = cands[0]
                    print(f"major {c['major_px']:6.1f}px  arc {c['arc_deg']:5.1f}  "
                          f"resid {c['ellipse_residual']:.3f}  fill {c['fill_ratio']:.2f} "
                          f"open {c['opening_area_ratio']:.2f}  "
                          f"clipped {str(c['clipped']):5s}  "
                          f"-> range {ring.range_m:5.2f} m  "
                          f"cx {ring.cx:+.3f} cy {ring.cy:+.3f} ar {ring.axis_ratio:.2f}"
                          f"  ({len(cands)} cand)")
                else:
                    print(f"no ring   blue {ring.blue_fraction*100:5.1f}% of frame")

            if a.no_window:
                continue

            view = bgr.copy()
            h, w = view.shape[:2]
            for i, c in enumerate(cands):
                centre = tuple(int(round(v)) for v in c["center_px"])
                axes = tuple(max(1, int(round(v / 2))) for v in c["axes_px"])
                col = (0, 255, 255) if i == 0 else (120, 120, 120)
                if c["clipped"]:
                    col = (0, 140, 255)
                cv2.ellipse(view, centre, axes, c["angle_deg"], 0, 360, col, 2)
                cv2.drawMarker(view, centre, col, cv2.MARKER_CROSS, 16, 2)
            cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            rng = f"{ring.range_m:.2f} m" if np.isfinite(ring.range_m) else "--"
            cv2.putText(view, f"{len(cands)} cand   {rng}   "
                              f"cx {ring.cx:+.2f} cy {ring.cy:+.2f} ar {ring.axis_ratio:.2f}"
                              f"   {'CLIPPED' if ring.clipped else ''}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(view, "GROUND ONLY - no flight commands", (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            both = np.hstack([view, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            cv2.imshow(WINDOW, both)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                a.out.write_text(json.dumps(params.to_json(), indent=2), encoding="utf-8")
                saved = True
                print(f"saved {a.out}")
            if k == ord("r"):
                tracker.reset()
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    if not a.no_window:
        cv2.destroyAllWindows()
    if a.save and not saved:
        a.out.write_text(json.dumps(params.to_json(), indent=2), encoding="utf-8")
        print(f"saved {a.out}")
    return params


if __name__ == "__main__":
    main()
