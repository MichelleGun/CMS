"""Per-flight JSONL log under hardware/flights/. One line per control tick plus event lines."""
import json
import time
from pathlib import Path

import numpy as np

from .config import FLIGHTS_DIR


def _clean(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 5).tolist()
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return round(o, 5) if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class FlightLogger:
    def __init__(self, tag="flight", directory=None, enabled=True):
        self.enabled = enabled
        self.path = None
        self._f = None
        if enabled:
            d = Path(directory) if directory else FLIGHTS_DIR
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            self._f = open(self.path, "w", encoding="utf-8")

    def tick(self, **fields):
        self._write({"type": "tick", **fields})

    def event(self, name, **fields):
        self._write({"type": "event", "name": name, **fields})

    def _write(self, d):
        if self._f is None:
            return
        self._f.write(json.dumps(_clean(d)) + "\n")
        self._f.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
