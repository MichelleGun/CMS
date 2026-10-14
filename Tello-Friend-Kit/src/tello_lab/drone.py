"""SDK connections and bounded flight plans."""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .recording import MAX_STATE_AGE_SECONDS, EventLog, TelemetryRecorder, output_path


class FlightError(RuntimeError):
    pass


def ensure_takeoff_clearance(settings: Settings, *, rise_cm=0):
    """Reject a plan whose nominal automatic takeoff/rise would consume roof clearance.

    80 cm is the observed automatic takeoff height, not a guaranteed firmware ceiling.
    This check does not establish horizontal clearance or authorise a flight.
    """
    if settings.cage_height_cm is None:
        return
    available = settings.cage_height_cm - settings.cage_clearance_cm
    planned = 80 + rise_cm
    if planned > available:
        raise FlightError(
            f"Cage height {settings.cage_height_cm:.2f} cm minus "
            f"{settings.cage_clearance_cm:g} cm roof clearance leaves {available:.2f} cm; "
            f"this plan needs nominal height {planned} cm. Ground camera checks remain available."
        )


def create_drone(settings: Settings):
    # Importing or previewing a plan never creates an SDK instance or opens sockets.
    from djitellopy import Tello

    Tello.LOGGER.setLevel(logging.WARNING)
    # A lost acknowledgement must not replay a relative movement or takeoff.
    return Tello(host=settings.host, retry_count=1)


@contextmanager
def connected(drone, *, wait_for_state: bool = True):
    try:
        drone.connect(wait_for_state=wait_for_state)
        yield drone
    finally:
        drone.end()


def try_set_video_bitrate(drone, mbps: int, events=None) -> bool:
    """Request a video bitrate, carrying on if the drone refuses.

    setbitrate is an SDK 3.0 (RoboMaster TT) command. A Tello EDU on SDK 2.0 answers
    "unknown command", which aborted the survey and observation modes outright on that
    aircraft. A lower bitrate only reduces video corruption, so it is never worth
    stopping for. Returns whether it was accepted, so a restore is only sent after a
    successful set.
    """
    try:
        drone.set_video_bitrate(mbps)
    except Exception as exc:
        if events is not None:
            events.write("video_bitrate_failed", mbps=mbps, error=str(exc))
        return False
    if events is not None:
        events.write("video_bitrate", mbps=mbps)
    return True


@dataclass(frozen=True)
class Step:
    command: str
    value: float | int = 0


def flight_plan(pattern: str, settings: Settings) -> list[Step]:
    steps = [Step("takeoff"), Step("hover", settings.hover_seconds)]
    if pattern == "square":
        steps.extend(
            Step(direction, settings.square_side_cm)
            for direction in ("forward", "right", "back", "left")
        )
    elif pattern != "hover":
        raise ValueError(f"Unknown flight pattern: {pattern}")
    return [*steps, Step("land")]


def battery_value(state: dict) -> int:
    battery = state.get("bat")
    if isinstance(battery, bool) or not isinstance(battery, int) or not 0 <= battery <= 100:
        raise FlightError("No valid battery telemetry; refusing flight")
    return battery


def assert_ready(recorder: TelemetryRecorder, minimum: int):
    state, age = recorder.snapshot()
    if not state or age > MAX_STATE_AGE_SECONDS:
        raise FlightError("Telemetry is missing or stale; stopping the test")
    battery = battery_value(state)
    if battery < minimum:
        raise FlightError(f"Battery is {battery}%; this phase requires at least {minimum}%")


def monitored_wait(seconds: float, recorder: TelemetryRecorder, minimum: int):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        assert_ready(recorder, minimum)
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    assert_ready(recorder, minimum)


def run_flight(drone, settings: Settings, pattern: str, log_dir: Path):
    ensure_takeoff_clearance(settings)
    plan = flight_plan(pattern, settings)
    csv_path = output_path(log_dir, pattern, ".csv")
    events = EventLog(csv_path.with_suffix(".jsonl"))
    recorder = TelemetryRecorder(drone, csv_path, settings.sample_hz)
    takeoff_attempted = False
    landing_error = None
    primary_error = None
    try:
        with connected(drone):
            try:
                recorder.start()
                events.write("connected", host=settings.host, pattern=pattern)
                assert_ready(recorder, settings.minimum_battery_percent)
                drone.set_speed(settings.speed_cm_s)
                for remaining in range(settings.countdown_seconds, 0, -1):
                    print(f"Takeoff in {remaining}s; Ctrl+C cancels.", flush=True)
                    monitored_wait(1, recorder, settings.minimum_battery_percent)
                for step in plan:
                    if step.command == "land":
                        break  # The finally block owns every landing attempt.
                    assert_ready(
                        recorder,
                        settings.abort_battery_percent
                        if takeoff_attempted
                        else settings.minimum_battery_percent,
                    )
                    events.write("command_start", command=step.command, value=step.value)
                    if step.command == "takeoff":
                        # A timeout can occur after the drone has actually taken off.
                        takeoff_attempted = True
                        drone.takeoff()
                    elif step.command == "hover":
                        monitored_wait(step.value, recorder, settings.abort_battery_percent)
                    else:
                        getattr(drone, f"move_{step.command}")(int(step.value))
                    events.write("command_complete", command=step.command)
            except BaseException as exc:
                primary_error = exc
                # Logging must never prevent the landing in the finally block.
                try:
                    events.write("test_aborted", reason=f"{type(exc).__name__}: {exc}")
                except OSError:
                    pass
                raise
            finally:
                if takeoff_attempted:
                    print("Requesting landing...", flush=True)
                    try:
                        drone.land()
                    except Exception as exc:
                        landing_error = exc
                        print(f"LANDING NOT CONFIRMED: {exc}", flush=True)
                    try:
                        events.write(
                            "land_failed" if landing_error else "land_complete",
                            reason=str(landing_error) if landing_error else "",
                        )
                    except OSError:
                        pass
                recorder.close()
    finally:
        events.close()
        print(f"Telemetry: {csv_path}\nEvents: {csv_path.with_suffix('.jsonl')}")
    if landing_error is not None and primary_error is None:
        raise FlightError(f"Landing was not confirmed: {landing_error}") from landing_error
