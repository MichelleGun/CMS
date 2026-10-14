"""A finite camera sweep over the takeoff position for target inspection."""

import json
import math
import time

from .config import PROJECT_ROOT
from .drone import (FlightError, assert_ready, connected, ensure_takeoff_clearance,
                    try_set_video_bitrate)
from .perception import annotate, blue_ring_candidates
from .recording import EventLog, TelemetryRecorder, output_path

# Conservative project test limits, not manufacturer temperature specifications.
MAX_START_TEMPERATURE_C = 80
MAX_AIR_TEMPERATURE_C = 85


def health(recorder, minimum_battery, *, airborne):
    assert_ready(recorder, minimum_battery)
    state, _ = recorder.snapshot()
    temperature = state.get("temph")
    if isinstance(temperature, bool) or not isinstance(temperature, int):
        raise FlightError("No valid mainboard temperature telemetry")
    limit = MAX_AIR_TEMPERATURE_C if airborne else MAX_START_TEMPERATURE_C
    if temperature > limit:
        raise FlightError(f"Mainboard reports {temperature} C; this survey limits it to {limit} C")
    if airborne:
        height = state.get("h")
        distance = state.get("tof")
        if isinstance(height, bool) or not isinstance(height, int) or not 30 <= height <= 120:
            raise FlightError("Reported airborne height is outside this survey's 30–120 cm range")
        if isinstance(distance, bool) or not isinstance(distance, int) or not 0 < distance < 6553:
            raise FlightError("Downward range telemetry is invalid; stopping the survey")
    if not airborne and state.get("h") != 0:
        raise FlightError("Survey must start with the drone on the ground")
    return state


def capture(reader, path, recorder, minimum_battery, *, airborne, small_camera=False):
    import cv2

    # Let motion and video settle, continuously discarding old decoded frames.
    # The first hardware survey showed distortion immediately after yaw commands.
    settle_until = time.monotonic() + 1
    while time.monotonic() < settle_until:
        health(recorder, minimum_battery, airborne=airborne)
        _ = reader.frame
        time.sleep(0.02)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = health(recorder, minimum_battery, airborne=airborne)
        frame = reader.frame
        if frame is not None and (frame.shape[1] <= 400) == small_camera:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f"Could not save {path}")
            candidates = blue_ring_candidates(bgr)
            annotated_path = path.with_name(path.stem + "_annotated.jpg")
            if not cv2.imwrite(str(annotated_path), annotate(bgr, candidates, state)):
                raise RuntimeError(f"Could not save {annotated_path}")
            return {
                "image": str(path),
                "annotated_image": str(annotated_path),
                "blue_ring_candidates": candidates,
                "width": frame.shape[1],
                "height": frame.shape[0],
                "state": state,
            }
        time.sleep(0.02)
    raise RuntimeError("No fresh frame from the selected camera within three seconds")


def run_survey(drone, settings, *, search_forward=False, landing_pad_id=None):
    ensure_takeoff_clearance(settings, rise_cm=20 if search_forward else 0)
    path = output_path(PROJECT_ROOT / "data" / "logs", "survey", ".csv")
    image_dir = PROJECT_ROOT / "data" / "photos" / path.stem
    image_dir.mkdir(parents=True, exist_ok=True)
    events = EventLog(path.with_suffix(".jsonl"))
    recorder = TelemetryRecorder(drone, path, settings.sample_hz)
    reader = None
    takeoff_attempted = False
    stream_attempted = False
    pads_attempted = False
    images = []
    result = {"status": "started", "images": images, "telemetry": str(path)}
    landing_error = None
    bitrate_attempted = False
    try:
        with connected(drone):
            try:
                recorder.start()
                health(recorder, settings.minimum_battery_percent, airborne=False)
                events.write("connected", purpose="identify targets using a camera sweep")
                if search_forward:
                    bitrate_attempted = try_set_video_bitrate(drone, 1, events)
                    drone.set_speed(settings.speed_cm_s)
                pads_attempted = True
                drone.enable_mission_pads()
                drone.set_mission_pad_detection_direction(0)
                stream_attempted = True
                drone.streamon()
                reader = drone.get_frame_read(with_queue=True, max_queue_len=2)
                images.append(
                    capture(
                        reader,
                        image_dir / "ground_front.jpg",
                        recorder,
                        settings.minimum_battery_percent,
                        airborne=False,
                    )
                )
                for remaining in (3, 2, 1):
                    print(f"Camera survey takeoff in {remaining}s; Ctrl+C cancels.", flush=True)
                    deadline = time.monotonic() + 1
                    while time.monotonic() < deadline:
                        health(recorder, settings.minimum_battery_percent, airborne=False)
                        time.sleep(0.05)
                events.write("takeoff_start")
                takeoff_attempted = True
                drone.takeoff()
                started = time.monotonic()
                events.write("takeoff_complete")
                if search_forward:
                    state = health(recorder, settings.abort_battery_percent, airborne=True)
                    if state["h"] > 90:
                        raise FlightError(
                            "No clearance within the 120 cm survey limit for a 20 cm rise"
                        )
                    events.write("rise_start", cm=20)
                    drone.move_up(20)
                    events.write("rise_complete", cm=20)
                headings = (0, 45, -45) if search_forward else (0, 90, 180, 270)
                rotations = (45, -90, 45) if search_forward else (90, 90, 90, 90)
                for heading, rotation in zip(headings, rotations, strict=True):
                    if time.monotonic() - started > 40:
                        raise FlightError("Camera sweep exceeded its 40-second airborne limit")
                    health(recorder, settings.abort_battery_percent, airborne=True)
                    frame = capture(
                        reader,
                        image_dir / f"front_{heading:03d}.jpg",
                        recorder,
                        settings.abort_battery_percent,
                        airborne=True,
                    )
                    frame["commanded_heading_offset_deg"] = heading
                    images.append(frame)
                    events.write("image_saved", **frame)
                    print(
                        f"Saved heading {heading} image; pad ID {frame['state'].get('mid')}",
                        flush=True,
                    )
                    if rotation > 0:
                        drone.rotate_clockwise(rotation)
                    else:
                        drone.rotate_counter_clockwise(-rotation)
                    events.write("rotation_complete", degrees=rotation)
                if landing_pad_id is not None:
                    state = health(recorder, settings.abort_battery_percent, airborne=True)
                    if state.get("mid") != landing_pad_id:
                        raise FlightError(f"Landing pad {landing_pad_id} is no longer detected")
                    for coordinate in ("x", "y"):
                        value = state.get(coordinate)
                        if isinstance(value, bool) or not isinstance(value, int) or abs(value) > 30:
                            raise FlightError(
                                "Pad-relative position exceeds the 30 cm return limit"
                            )
                    if math.hypot(state["x"], state["y"]) > 30:
                        raise FlightError("Pad-relative position exceeds the 30 cm return limit")
                    pad_height = state.get("z")
                    if not isinstance(pad_height, int) or not 30 <= pad_height <= 120:
                        raise FlightError("Landing pad height is outside this survey's range")
                    events.write("pad_return_start", pad_id=landing_pad_id, x=0, y=0, z=80)
                    drone.go_xyz_speed_mid(0, 0, 80, settings.speed_cm_s, landing_pad_id)
                    events.write("pad_return_complete", pad_id=landing_pad_id)
                    result["pad_return_acknowledged"] = landing_pad_id
                result["status"] = "survey_complete"
            except BaseException as exc:
                result["status"] = "aborted"
                result["reason"] = f"{type(exc).__name__}: {exc}"
                try:
                    events.write("survey_aborted", reason=result["reason"])
                except OSError:
                    pass
                raise
            finally:
                if takeoff_attempted:
                    print("Survey finished; requesting landing...", flush=True)
                    try:
                        drone.land()
                        result["landing_acknowledged"] = True
                    except Exception as exc:
                        landing_error = exc
                        result["landing_acknowledged"] = False
                        result["landing_error"] = str(exc)
                        print(f"LANDING NOT CONFIRMED: {exc}", flush=True)
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
                                try_set_video_bitrate(drone, 0)
                        finally:
                            recorder.close()
    finally:
        events.close()
        path.with_suffix(".summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Survey record: {path.with_suffix('.summary.json')}", flush=True)
    if landing_error is not None:
        raise FlightError(f"Survey landing was not confirmed: {landing_error}") from landing_error
    return result
