"""Local flight events and sampled state packets; no commands are sent here."""

import csv
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

STATE_FIELDS = (
    "bat",
    "h",
    "tof",
    "pitch",
    "roll",
    "yaw",
    "vgx",
    "vgy",
    "vgz",
    "templ",
    "temph",
    "baro",
    "time",
    "agx",
    "agy",
    "agz",
    "mid",
    "x",
    "y",
    "z",
)
MAX_STATE_AGE_SECONDS = 3.0


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def output_path(directory: Path, label: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    return directory / f"{stamp}_{label}{suffix}"


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("x", encoding="utf-8")

    def write(self, event: str, **fields):
        self.file.write(json.dumps({"utc": timestamp(), "event": event, **fields}) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TelemetryRecorder:
    """Sample SDK state in the background while movement calls block the main thread.

    DJITelloPy 2.5 replaces its state dict for each received UDP packet. Retaining the
    previous dict lets us distinguish fresh packets from an unchanged cached value.
    The reported age is an estimate at the sampling resolution.
    """

    def __init__(self, drone, path: Path, sample_hz: float):
        self.drone = drone
        self.path = path
        self.interval = 1.0 / sample_hz
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._packet = None
        self._last_seen = time.monotonic()
        self._state = {}
        self._error = None
        self._file = None
        self._thread = None

    def sample(self) -> tuple[dict, float]:
        packet = self.drone.get_current_state()
        now = time.monotonic()
        with self._lock:
            if packet and packet is not self._packet:
                self._packet = packet
                self._last_seen = now
                self._state = dict(packet)
            return dict(self._state), max(0.0, now - self._last_seen)

    def snapshot(self) -> tuple[dict, float]:
        if self._error is not None:
            raise RuntimeError(f"Telemetry recording failed: {self._error}") from self._error
        with self._lock:
            return dict(self._state), max(0.0, time.monotonic() - self._last_seen)

    def start(self):
        self._file = self.path.open("x", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            self._file, fieldnames=("utc", "elapsed_s", "state_age_s", *STATE_FIELDS)
        )
        writer.writeheader()
        started = time.monotonic()
        # The initial sample is synchronous so the caller can preflight immediately.
        try:
            state, age = self.sample()
            writer.writerow(
                {
                    "utc": timestamp(),
                    "elapsed_s": 0,
                    "state_age_s": age,
                    **{key: state.get(key, "") for key in STATE_FIELDS},
                }
            )
            self._file.flush()
        except BaseException:
            self._file.close()
            raise

        def record():
            try:
                while not self._stop.wait(self.interval):
                    state, age = self.sample()
                    writer.writerow(
                        {
                            "utc": timestamp(),
                            "elapsed_s": round(time.monotonic() - started, 3),
                            "state_age_s": round(age, 3),
                            **{key: state.get(key, "") for key in STATE_FIELDS},
                        }
                    )
                    self._file.flush()
            except Exception as exc:
                self._error = exc

        self._thread = threading.Thread(target=record, name="tello-telemetry", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("Telemetry thread did not stop")
        if self._file is not None:
            self._file.close()
