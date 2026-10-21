"""
The dashboard's camera calibration: run hardware/onboard/calib_session.py (ground only: it never
sends takeoff or stick commands), relay its frames and status, pass slider changes and click
auto-fits to it, and keep the named presets (hardware/onboard/detector_presets/).

Why: daytime flights locked onto the window instead of the hoop with the night calibration
(2026-10-14), so a calibration per lighting (day / night / other) is saved and chosen per flight.

Same approach as drone_manager.py: the session is a subprocess, and the session folder
(hardware/calibration/sessions/<id>/) is the only channel. This module never imports OpenCV.
One session at a time. A Tello session and a flight exclude each other (both need the drone's
UDP ports); main.py checks both ways.
"""
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import CMS_DIR, FLIGHTS_DIR, HARDWARE_DIR
from jobs import PYTHON_EXE, graceful_stop, read_json, spawn, write_json_atomic

if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))
from onboard import detector_presets as presets  # noqa: E402  (plain JSON, no OpenCV)

CALIB_SCRIPT = HARDWARE_DIR / "onboard" / "calib_session.py"
SESSIONS_DIR = HARDWARE_DIR / "calibration" / "sessions"
SOURCES = ("tello", "webcam", "image")
FRAMES = ("raw", "overlay", "mask")


class CalibManager:
    def __init__(self, sessions_dir: Path = SESSIONS_DIR, flights_dir: Path = FLIGHTS_DIR):
        self.sessions_dir = Path(sessions_dir)
        self.flights_dir = Path(flights_dir)
        self.proc = None
        self.dir: Optional[Path] = None
        self.source: Optional[str] = None

    # -- session --------------------------------------------------------------------
    def active(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def uses_drone(self) -> bool:
        return self.active() and self.source == "tello"

    def start(self, source="tello", webcam_index: int = 0, image: Optional[str] = None,
              preset: Optional[str] = None, flight_active: bool = False) -> str:
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        if self.active():
            raise RuntimeError("a calibration session is already running")
        if source == "tello" and flight_active:
            raise RuntimeError("a flight is active: the drone's camera is in use")
        if preset:
            presets.get_preset(preset)                    # KeyError -> 404 in main.py
        sid = f"calib_{datetime.now():%Y%m%d-%H%M%S}"
        d = self.sessions_dir / sid
        d.mkdir(parents=True, exist_ok=True)
        argv = [PYTHON_EXE, str(CALIB_SCRIPT), "--dir", str(d), "--source", source]
        if source == "webcam":
            argv += ["--webcam", str(int(webcam_index))]
        if source == "image":
            img = self.resolve_image(image)
            argv += ["--image", str(img)]
        if preset:
            argv += ["--preset", preset]
        self.proc = spawn(argv, d / "session.log", CMS_DIR)
        self.dir, self.source = d, source
        return sid

    def stop(self) -> bool:
        if self.dir is None:
            return False
        (self.dir / "stop.flag").touch()
        if self.proc is not None:
            end = time.monotonic() + 6
            while self.proc.poll() is None and time.monotonic() < end:
                time.sleep(0.1)
            if self.proc.poll() is None:
                graceful_stop(self.proc, timeout=5)
        return True

    def status(self) -> dict:
        st = read_json(self.dir / "status.json") if self.dir else None
        res = read_json(self.dir / "autofit_result.json") if self.dir else None
        log_tail = None
        if self.dir and not self.active() and (self.dir / "session.log").is_file():
            try:
                log_tail = (self.dir / "session.log").read_text(encoding="utf-8", errors="replace")[-1500:]
            except OSError:
                pass
        return dict(active=self.active(), session=self.dir.name if self.dir else None,
                    source=self.source, status=st, autofit=res, log_tail=log_tail)

    def frame_path(self, kind: str) -> Optional[Path]:
        if kind not in FRAMES or self.dir is None:
            return None
        p = self.dir / f"{kind}.jpg"
        return p if p.is_file() else None

    def _require(self):
        if not self.active():
            raise RuntimeError("no calibration session is running")

    def current_params(self) -> dict:
        if self.dir is not None:
            p = read_json(self.dir / "params.json")
            if p:
                return p
        return presets.default_preset()["params"]

    def set_params(self, changes: dict) -> dict:
        """Merge slider changes into the session's params (the session re-reads them)."""
        self._require()
        merged = presets.clean_params(dict(self.current_params(), **changes))
        write_json_atomic(self.dir / "params.json", merged)
        return merged

    def autofit(self, ring: list, reject: list, clear: bool = False, timeout: float = 8.0) -> dict:
        """Clicks on the CURRENT frame join the samples of earlier clicks (the hoop can be clicked
        on one frame and the window on another). clear=True forgets them first; clear=True with
        no clicks only forgets."""
        self._require()
        for pts in (ring, reject):
            for p in pts:
                if len(p) != 2 or not all(0.0 <= float(v) <= 1.0 for v in p):
                    raise ValueError("points are [x, y] fractions of the frame (0-1)")
        res_path = self.dir / "autofit_result.json"
        res_path.unlink(missing_ok=True)
        write_json_atomic(self.dir / "autofit.json", {"ring": ring, "reject": reject, "clear": bool(clear)})
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            res = read_json(res_path)
            if res is not None:
                return res
            time.sleep(0.1)
        raise RuntimeError("the calibration session did not answer the auto-fit in time")

    def switch_image(self, image: str) -> None:
        """Source 'image' only: show another saved frame (samples and params are kept)."""
        self._require()
        if self.source != "image":
            raise ValueError("only a session on saved frames can switch images")
        write_json_atomic(self.dir / "image.json", {"path": str(self.resolve_image(image))})

    def snapshot(self) -> None:
        self._require()
        (self.dir / "snapshot.flag").touch()

    # -- images to calibrate against (source "image") ------------------------------------
    def images(self, limit: int = 30) -> list:
        """Recent camera frames from flights and calibration snapshots, newest first."""
        found = []
        for p in list(self.flights_dir.glob("*/camera_latest.jpg")) + list(
                self.sessions_dir.glob("*/snapshots/*.jpg")):
            try:
                found.append((p.stat().st_mtime, p))
            except OSError:
                pass
        found.sort(reverse=True)
        return [dict(path=str(p.relative_to(CMS_DIR)).replace("\\", "/"),
                     taken=datetime.fromtimestamp(t).isoformat(timespec="minutes"))
                for t, p in found[:limit]]

    def resolve_image(self, image: Optional[str]) -> Path:
        if not image:
            raise ValueError("source 'image' needs an image (pick one from the list)")
        p = (CMS_DIR / image).resolve()
        allowed = (self.flights_dir.resolve(), self.sessions_dir.resolve())
        if not any(str(p).startswith(str(a)) for a in allowed) or not p.is_file():
            raise ValueError("pick an image from the list (flight camera frames or snapshots)")
        return p

    # -- presets -----------------------------------------------------------------------
    def list_presets(self) -> dict:
        ps = presets.list_presets()
        for d in ps:
            d.pop("path", None)
        return dict(presets=ps, suggested=presets.suggest(ps), conditions=list(presets.CONDITIONS))

    def save_preset(self, name: str, condition: str = "other", notes: str = "") -> dict:
        params = self.current_params()
        thumb = None
        if self.dir is not None and (self.dir / "overlay.jpg").is_file():
            thumb = (self.dir / "overlay.jpg").read_bytes()
        d = presets.save_preset(name, params, condition=condition, notes=notes, thumb_jpg=thumb)
        d.pop("path", None)
        return d

    def preset_thumb(self, slug: str) -> Optional[Path]:
        p = presets.PRESET_DIR / f"{presets.slugify(slug)}.jpg"
        return p if p.is_file() else None

    def load_preset_into_session(self, slug: str) -> dict:
        self._require()
        params = presets.get_preset(slug)["params"]
        write_json_atomic(self.dir / "params.json", params)
        return params
