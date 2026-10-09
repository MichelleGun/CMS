"""
The only module that couples this package to Tello-Friend-Kit.

Everything else in hardware/onboard/ imports its drone primitives from here, so the
coupling is one import line in one file. If the kit moves or is replaced, this is the
only thing to change.

Why reuse the kit at all: its safety skeleton is the only code in this repository that
has actually flown this drone (`tello_lab flight hover --fly` succeeded). Re-deriving
the battery gating, the telemetry freshness check and the landing-in-finally pattern
would mean re-earning that confidence. The kit is also a sealed artifact with a
SHA-256 MANIFEST.json, so it is imported and never edited.

What is deliberately NOT reused: tello_lab.perception. Three of its rejection gates have
to become tunable for a bent pool-noodle ring and for close approach, so ring_detect.py
adapts it instead. See that module's docstring.
"""
import sys
from pathlib import Path

HW_ROOT = Path(__file__).resolve().parents[1]           # CMS/hardware
CMS_ROOT = HW_ROOT.parent                               # CMS
KIT_SRC = CMS_ROOT / "Tello-Friend-Kit" / "src"

if not KIT_SRC.is_dir():
    raise ImportError(
        f"Tello-Friend-Kit not found at {KIT_SRC}. hardware/onboard/ needs it for the "
        "drone connection and safety primitives."
    )
if str(KIT_SRC) not in sys.path:
    sys.path.insert(0, str(KIT_SRC))

# flake8: noqa: E402  -- the sys.path insert above has to run first
from tello_lab.config import Settings, bounded, load_settings            # noqa: F401
from tello_lab.drone import (                                           # noqa: F401
    FlightError,
    assert_ready,
    battery_value,
    connected,
    create_drone,
    ensure_takeoff_clearance,
    monitored_wait,
)
from tello_lab.recording import (                                       # noqa: F401
    MAX_STATE_AGE_SECONDS,
    EventLog,
    TelemetryRecorder,
    output_path,
    timestamp,
)
from tello_lab.survey import (                                          # noqa: F401
    MAX_AIR_TEMPERATURE_C,
    MAX_START_TEMPERATURE_C,
    health,
)

KIT_CONFIG = CMS_ROOT / "Tello-Friend-Kit" / "config" / "tello.toml"

__all__ = [
    "Settings", "bounded", "load_settings",
    "FlightError", "assert_ready", "battery_value", "connected", "create_drone",
    "ensure_takeoff_clearance", "monitored_wait",
    "MAX_STATE_AGE_SECONDS", "EventLog", "TelemetryRecorder", "output_path", "timestamp",
    "MAX_AIR_TEMPERATURE_C", "MAX_START_TEMPERATURE_C", "health",
    "KIT_CONFIG", "KIT_SRC", "HW_ROOT", "CMS_ROOT",
]
