"""Ground camera check with optional local recordings."""

import json
import time

from .config import PROJECT_ROOT
from .drone import connected
from .recording import output_path


def capture_snapshot(drone):
    """Save a fresh front-camera image and ground telemetry without taking off."""
    import cv2

    reader = None
    stream_attempted = False
    with connected(drone):
        try:
            stream_attempted = True
            drone.streamon()
            reader = drone.get_frame_read(with_queue=True, max_queue_len=2)
            started = time.monotonic()
            frame = None
            while time.monotonic() - started < 5:
                current = reader.frame
                if current is not None:
                    frame = current
                    if time.monotonic() - started >= 1:
                        break
                time.sleep(0.02)
            if frame is None:
                raise RuntimeError("No camera frame received within five seconds")
            path = output_path(PROJECT_ROOT / "data" / "photos", "scene", ".jpg")
            if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not save camera snapshot: {path}")
            metadata = {
                "image": str(path),
                "width": frame.shape[1],
                "height": frame.shape[0],
                "state": dict(drone.get_current_state()),
                "takeoff_sent": False,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(json.dumps(metadata, indent=2))
            return path
        finally:
            if reader is not None:
                reader.stop()
            if stream_attempted:
                drone.streamoff()


def show_video(drone, *, record: bool, seconds: float | None):
    import cv2

    writer = None
    reader = None
    stream_started = False
    with connected(drone):
        try:
            drone.streamon()
            stream_started = True
            reader = drone.get_frame_read(with_queue=True, max_queue_len=2)
            deadline = time.monotonic() + seconds if seconds is not None else None
            last_frame_at = time.monotonic()
            print("Camera check only. Q/Esc: close; S: save photo; Ctrl+C: stop.")
            while deadline is None or time.monotonic() < deadline:
                rgb_frame = reader.frame
                if rgb_frame is None:
                    if time.monotonic() - last_frame_at > 5:
                        raise RuntimeError("No camera frame received for five seconds")
                    time.sleep(0.01)
                    continue
                last_frame_at = time.monotonic()
                # DJITelloPy/PyAV produces RGB; OpenCV displays and writes BGR.
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                if record and writer is None:
                    path = output_path(PROJECT_ROOT / "data" / "videos", "camera", ".mp4")
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV could not open the MP4 video writer")
                    print(f"Recording (nominal 20 fps): {path}")
                if writer is not None:
                    writer.write(frame)
                cv2.imshow("Tello EDU camera - Q to close", frame)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = output_path(PROJECT_ROOT / "data" / "photos", "camera", ".jpg")
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError(f"Could not save photo: {path}")
                    print(f"Saved: {path}")
                if cv2.getWindowProperty("Tello EDU camera - Q to close", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            if reader is not None:
                reader.stop()
            if stream_started:
                drone.streamoff()
