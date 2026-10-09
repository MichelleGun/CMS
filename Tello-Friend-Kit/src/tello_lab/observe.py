"""Finite live target observation from the ground; no motor commands."""

import time

import cv2

from .config import PROJECT_ROOT
from .drone import connected
from .perception import annotate, blue_ring_candidates
from .recording import EventLog, TelemetryRecorder, output_path
from .survey import health


def observe(drone, settings, seconds, *, display=True, bitrate_mbps=None):
    path = output_path(PROJECT_ROOT / "data" / "logs", "observe", ".csv")
    images = PROJECT_ROOT / "data" / "photos" / path.stem
    images.mkdir(parents=True, exist_ok=True)
    events = EventLog(path.with_suffix(".jsonl"))
    recorder = TelemetryRecorder(drone, path, settings.sample_hz)
    reader = None
    pads_attempted = stream_attempted = bitrate_attempted = False
    count = 0
    try:
        with connected(drone):
            try:
                recorder.start()
                health(recorder, settings.minimum_battery_percent, airborne=False)
                if bitrate_mbps is not None:
                    bitrate_attempted = True
                    drone.set_video_bitrate(bitrate_mbps)
                    events.write("video_bitrate", mbps=bitrate_mbps)
                pads_attempted = True
                drone.enable_mission_pads()
                drone.set_mission_pad_detection_direction(2)
                stream_attempted = True
                drone.streamon()
                reader = drone.get_frame_read(with_queue=True, max_queue_len=2)
                started = last_frame = last_saved = time.monotonic()
                print("Live ground observation; Q/Esc or Ctrl+C stops. No takeoff.", flush=True)
                while time.monotonic() - started < seconds:
                    state = health(recorder, settings.minimum_battery_percent, airborne=False)
                    rgb = reader.frame
                    now = time.monotonic()
                    if rgb is None:
                        if now - last_frame > 3:
                            raise RuntimeError("Camera produced no frame for three seconds")
                        time.sleep(0.01)
                        continue
                    last_frame = now
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    candidates = blue_ring_candidates(bgr)
                    rendered = annotate(bgr, candidates, state)
                    count += 1
                    if now - last_saved >= 1:
                        image = images / f"frame_{count:05d}.jpg"
                        if not cv2.imwrite(str(image), rendered):
                            raise RuntimeError(f"Could not save {image}")
                        events.write(
                            "observation",
                            frame_number=count,
                            image=str(image),
                            candidates=candidates,
                            state=state,
                        )
                        print(
                            f"Frame {count}: {len(candidates)} blue ring candidates, "
                            f"mission pad {state.get('mid')}",
                            flush=True,
                        )
                        last_saved = now
                    if display:
                        cv2.imshow("Tello live target observation", rendered)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                        if (
                            cv2.getWindowProperty(
                                "Tello live target observation", cv2.WND_PROP_VISIBLE
                            )
                            < 1
                        ):
                            break
            finally:
                if reader is not None:
                    reader.stop()
                try:
                    if stream_attempted:
                        drone.streamoff()
                finally:
                    try:
                        if pads_attempted:
                            drone.disable_mission_pads()
                    finally:
                        try:
                            if bitrate_attempted:
                                drone.set_video_bitrate(0)  # Restore SDK automatic selection.
                        finally:
                            recorder.close()
    finally:
        if display:
            cv2.destroyAllWindows()
        events.close()
        print(f"Observation record: {path.with_suffix('.jsonl')}", flush=True)
    return count
