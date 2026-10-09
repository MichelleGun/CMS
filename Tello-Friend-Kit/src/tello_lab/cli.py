"""Run with python -m tello_lab; hardware access is always an explicit subcommand."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, PROJECT_ROOT, bounded, load_settings
from .drone import (
    connected,
    create_drone,
    ensure_takeoff_clearance,
    flight_plan,
    monitored_wait,
    run_flight,
)
from .recording import TelemetryRecorder, output_path


def doctor() -> int:
    print(f"Python: {sys.version.split()[0]}\nInterpreter: {sys.executable}")
    print(f"Project: {PROJECT_ROOT}")
    ok = sys.version_info[:2] == (3, 11)
    for distribution, module in (
        ("djitellopy", "djitellopy"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("av", "av"),
        ("pytest", "pytest"),
        ("ruff", None),
    ):
        try:
            version = importlib.metadata.version(distribution)
            if module is not None:
                importlib.import_module(module)
            print(f"OK {distribution}: {version}")
        except (ImportError, importlib.metadata.PackageNotFoundError, OSError) as exc:
            ok = False
            print(f"FAIL {distribution}: {exc}")
    print("No drone connection or flight commands were sent.")
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Single Tello EDU connection and flight testing")
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML settings file")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check Python and package imports offline")
    commands.add_parser("check", help="Connect and print battery/state; no takeoff")
    commands.add_parser("snapshot", help="Save a fresh camera image and ground state; no takeoff")
    telemetry = commands.add_parser("telemetry", help="Record stationary telemetry; no takeoff")
    telemetry.add_argument("--seconds", type=float, default=10)
    telemetry.add_argument("--mission-pads", action="store_true", help="Enable EDU pad detection")
    telemetry.add_argument(
        "--direction",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Pad detection: 0 downward, 1 forward, 2 both",
    )
    video = commands.add_parser("video", help="Open the camera preview; no takeoff")
    video.add_argument("--record", action="store_true", help="Save preview to MP4")
    video.add_argument("--seconds", type=float, help="Close the preview after this duration")
    observe = commands.add_parser(
        "observe", help="Live blue ring and mission-pad observations on the ground"
    )
    observe.add_argument(
        "--seconds", type=float, default=20, help="Observation duration, 1–60 seconds"
    )
    observe.add_argument(
        "--no-window", action="store_true", help="Save observations without a preview window"
    )
    observe.add_argument(
        "--bitrate-mbps",
        type=int,
        choices=range(1, 6),
        help="Temporarily set video bitrate; restores automatic selection on exit",
    )
    flight = commands.add_parser("flight", help="Preview a plan offline, or fly with --fly")
    flight.add_argument("pattern", choices=("hover", "square"))
    flight.add_argument("--fly", action="store_true", help="Execute real takeoff and movement")
    survey = commands.add_parser(
        "survey", help="Preview or fly a camera sweep over the takeoff spot"
    )
    survey.add_argument(
        "--fly", action="store_true", help="Take off, photograph four headings, land"
    )
    survey.add_argument(
        "--search-forward",
        action="store_true",
        help="Rise 20 cm after takeoff and inspect forward/45 degrees right/45 degrees left",
    )
    survey.add_argument(
        "--landing-pad",
        type=int,
        choices=range(1, 9),
        help="Centre over this detected pad before landing; limited to 30 cm horizontal correction",
    )
    commands.add_parser("land", help="Send normal landing; no battery or state check required")
    return result


def record_telemetry(drone, settings, args):
    path = output_path(PROJECT_ROOT / "data" / "logs", "ground", ".csv")
    recorder = TelemetryRecorder(drone, path, settings.sample_hz)
    pads_attempted = False
    with connected(drone):
        try:
            if args.mission_pads:
                pads_attempted = True
                drone.enable_mission_pads()
                drone.set_mission_pad_detection_direction(args.direction)
            recorder.start()
            # No battery threshold is needed for a stationary telemetry check.
            monitored_wait(args.seconds, recorder, 0)
        finally:
            recorder.close()
            if pads_attempted:
                drone.disable_mission_pads()
    print(f"Saved telemetry: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        settings = load_settings(args.config)
        if args.command in ("flight", "survey") and settings.cage_height_cm is not None:
            print(
                f"Cage roof: {settings.cage_height_cm:.2f} cm; "
                f"height after roof clearance: "
                f"{settings.cage_height_cm - settings.cage_clearance_cm:.2f} cm."
            )
        if args.command == "survey" and not args.fly:
            if args.search_forward:
                print("OFFLINE PREVIEW: take off, rise 20 cm, inspect forward and 45 degrees")
                print("right/left, then restore heading and land. Uses 1 Mbps video.")
            else:
                print("OFFLINE PREVIEW: hover over the takeoff spot, photograph four headings,")
                print("rotate clockwise by 90 degrees four times to restore heading, then land.")
            if args.landing_pad is not None:
                print(
                    f"Before landing, centre at (0, 0, 80 cm) over detected pad {args.landing_pad}."
                )
            print("Includes mainboard temperature, battery, telemetry and height checks.")
            print("Add --fly to execute the camera survey.")
            return 0
        if args.command == "flight" and not args.fly:
            print("OFFLINE PREVIEW: no connection, takeoff, or movement.")
            print(f"Host: {settings.host}; speed: {settings.speed_cm_s} cm/s")
            for step in flight_plan(args.pattern, settings):
                unit = " s" if step.command == "hover" else " cm"
                detail = f" {step.value}{unit}" if step.value else ""
                print(f"  {step.command}{detail}")
            print("Add --fly to execute this plan after connecting to the drone Wi-Fi.")
            return 0
        if args.command in ("telemetry", "video") and args.seconds is not None:
            bounded("seconds", args.seconds, 1, 300)
        if args.command == "observe":
            bounded("seconds", args.seconds, 1, 60)
        if args.command in ("flight", "survey") and args.fly:
            ensure_takeoff_clearance(
                settings,
                rise_cm=20 if args.command == "survey" and args.search_forward else 0,
            )
        drone = create_drone(settings)
        if args.command == "check":
            with connected(drone):
                print(f"Connected to Tello at {settings.host}; battery: {drone.get_battery()}%")
                print(json.dumps(dict(drone.get_current_state()), indent=2))
                # Query SDK version only after the mandatory communication check succeeds.
                try:
                    print(f"Reported SDK version: {drone.query_sdk_version()}")
                except Exception as exc:
                    print(f"SDK version query unavailable: {exc}")
        elif args.command == "telemetry":
            record_telemetry(drone, settings, args)
        elif args.command == "video":
            from .video import show_video

            show_video(drone, record=args.record, seconds=args.seconds)
        elif args.command == "snapshot":
            from .video import capture_snapshot

            capture_snapshot(drone)
        elif args.command == "observe":
            from .observe import observe

            observe(
                drone,
                settings,
                args.seconds,
                display=not args.no_window,
                bitrate_mbps=args.bitrate_mbps,
            )
        elif args.command == "flight":
            print("REAL FLIGHT: keep the drone in sight and the test area clear.")
            run_flight(drone, settings, args.pattern, PROJECT_ROOT / "data" / "logs")
        elif args.command == "survey":
            from .survey import run_survey

            run_survey(
                drone,
                settings,
                search_forward=args.search_forward,
                landing_pad_id=args.landing_pad,
            )
        elif args.command == "land":
            with connected(drone, wait_for_state=False):
                drone.land()
                print("Landing acknowledged.")
        return 0
    except KeyboardInterrupt:
        print(
            "Stopped by Ctrl+C. If takeoff was attempted, landing was requested.", file=sys.stderr
        )
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "For connection issues, see docs/setup.md. Run only one hardware script at a time.",
            file=sys.stderr,
        )
        return 1
