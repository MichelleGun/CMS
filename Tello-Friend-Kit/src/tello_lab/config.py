"""Validated settings for a small, single-drone flight test."""

import ipaddress
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "tello.toml"


def bounded(name: str, value: object, low: float, high: float, *, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


@dataclass(frozen=True)
class Settings:
    host: str = "192.168.10.1"
    minimum_battery_percent: int = 30
    abort_battery_percent: int = 20
    speed_cm_s: int = 20
    hover_seconds: float = 5.0
    square_side_cm: int = 50
    countdown_seconds: int = 5
    sample_hz: float = 5.0
    cage_width_cm: float | None = None
    cage_length_cm: float | None = None
    cage_height_cm: float | None = None
    cage_clearance_cm: float = 30.0

    def __post_init__(self):
        ipaddress.IPv4Address(self.host)
        bounded("minimum_battery_percent", self.minimum_battery_percent, 30, 100, integer=True)
        bounded("abort_battery_percent", self.abort_battery_percent, 10, 99, integer=True)
        if self.abort_battery_percent >= self.minimum_battery_percent:
            raise ValueError("abort_battery_percent must be below minimum_battery_percent")
        bounded("speed_cm_s", self.speed_cm_s, 10, 30, integer=True)
        bounded("hover_seconds", self.hover_seconds, 1, 10)
        bounded("square_side_cm", self.square_side_cm, 20, 100, integer=True)
        bounded("countdown_seconds", self.countdown_seconds, 3, 10, integer=True)
        bounded("sample_hz", self.sample_hz, 1, 20)
        for field in ("cage_width_cm", "cage_length_cm", "cage_height_cm"):
            value = getattr(self, field)
            if value is not None:
                bounded(field, value, 50, 1000)
        bounded("cage_clearance_cm", self.cage_clearance_cm, 10, 100)


def load_settings(path: Path = DEFAULT_CONFIG) -> Settings:
    with path.open("rb") as source:
        data = tomllib.load(source)
    allowed = {
        "drone": {"host"},
        "flight": {
            "minimum_battery_percent",
            "abort_battery_percent",
            "speed_cm_s",
            "hover_seconds",
            "square_side_cm",
            "countdown_seconds",
        },
        "telemetry": {"sample_hz"},
        "cage": {"width_cm", "length_cm", "height_cm", "clearance_cm"},
    }
    values = {}
    for section, entries in data.items():
        if section not in allowed or not isinstance(entries, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        for key, value in entries.items():
            if key not in allowed[section]:
                raise ValueError(f"Unknown setting: {section}.{key}")
            values[f"cage_{key}" if section == "cage" else key] = value
    return Settings(**values)
