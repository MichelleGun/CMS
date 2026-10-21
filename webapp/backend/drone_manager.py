"""
The real-drone side of the dashboard: start flights, list them, read their live state,
forward keyboard keys, land, kill and answer the post-landing save question.

Same "the directory tree IS the database" approach as run_manager.py, so a backend
restart needs no recovery code. It reads what hardware/onboard/fly_hoop.py writes under
hardware/flights/<flight_id>/ and never imports djitellopy, OpenCV or the flight code.

SAFETY DECISIONS BUILT IN HERE, not in the UI, so they cannot be clicked around:

1. Real flights CAN be started from the dashboard (the user chose one confirm click). What
   is still enforced: the request must carry confirm=True, it must come from this computer
   (main.py checks the client address), and no other flight may be active, including one
   started from a terminal.

2. Land and kill are FLAG FILES, not a socket or a signal. fly_hoop.py polls land.flag and
   kill.flag every tick. That is fail-safe in the direction that matters: if this backend
   dies, no flag appears and the flight's own safety layer still lands the drone.

3. Human-flight keys go over UDP to 127.0.0.1 with a deadman in the flight
   (onboard/human.py UdpKeys): no packet for 0.4 s means nothing is held and the drone
   hovers. A closed tab or a dead backend ends in a hover, never a stuck key.
"""
import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import CMS_DIR, FLIGHTS_DIR, FLY_HOOP_SCRIPT
from jobs import (PYTHON_EXE, graceful_stop, pid_alive, read_json as _read_json,
                  spawn, write_json_atomic)

FLIGHT_PREFIX = "tello_"
LAND_FLAG = "land.flag"
KILL_FLAG = "kill.flag"
CAMERA_FLAG = "camera_enabled.flag"
SAVE_ANSWER = "save_answer.json"
HELD_KEYS = {"fb+", "fb-", "lr+", "lr-", "ud+", "ud-", "yaw+", "yaw-"}
KEYS_META_TTL_S = 1.0     # how stale send_keys' cached keys port may get

MODES = [
    {"id": "expert", "label": "Scripted expert"},
    {"id": "policy", "label": "Trained policy"},
    {"id": "hover", "label": "Hover and detect only"},
    {"id": "human", "label": "Fly it yourself (keyboard)"},
]
PROFILES = [
    {"id": "ideal", "label": "Ideal"},
    {"id": "typical", "label": "Typical"},
    {"id": "harsh", "label": "Harsh"},
]
MODE_IDS = {m["id"] for m in MODES}
PROFILE_IDS = {p["id"] for p in PROFILES}


def _demo_label(d: Path) -> Optional[str]:
    if (d / "demo.npz").is_file():
        return "pass"
    if (d / "demo_failed.npz").is_file():
        return "miss"
    return None


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DroneManager:
    def __init__(self, flights_dir: Path = FLIGHTS_DIR):
        self.flights_dir = flights_dir
        self._procs: dict = {}
        self._keys_ports: dict = {}     # flight_id -> (port or None, monotonic time read)

    # ------------------------------------------------------------------ state

    @staticmethod
    def _proc_alive(proc) -> bool:
        return proc.poll() is None

    def is_flight_active(self) -> bool:
        """True if a flight this backend started is still running, or the newest flight's
        own recorded pid is still alive (which covers a flight started from a terminal)."""
        if any(self._proc_alive(p) for p in self._procs.values()):
            return True
        newest = self.newest_flight_id()
        return bool(newest and self._pid_alive(newest))

    def _pid_alive(self, flight_id: str) -> bool:
        meta = _read_json(self.flights_dir / flight_id / "meta.json") or {}
        if meta.get("final_status"):
            return False          # it recorded its own ending
        return pid_alive(meta.get("pid"))

    def _flight_dirs(self):
        if not self.flights_dir.is_dir():
            return []
        dirs = [p for p in self.flights_dir.iterdir()
                if p.is_dir() and p.name.startswith(FLIGHT_PREFIX)]
        return sorted(dirs, key=lambda p: p.name, reverse=True)   # name is a timestamp

    def newest_flight_id(self) -> Optional[str]:
        dirs = self._flight_dirs()
        return dirs[0].name if dirs else None

    def list_flights(self, limit: Optional[int] = 25) -> dict:
        flights = []
        for d in self._flight_dirs()[:limit]:
            meta = _read_json(d / "meta.json") or {}
            summary = _read_json(d / "summary.json") or {}
            flights.append({
                "flight_id": d.name,
                "mode": meta.get("mode"),
                "dry_run": meta.get("dry_run"),
                "status": meta.get("final_status") or ("running" if self._pid_alive(d.name)
                                                       else "unknown"),
                "outcome": summary.get("outcome"),
                "sim_outcome": summary.get("sim_outcome"),
                "travel_m": summary.get("travel_m"),
                "created_at": meta.get("created_at"),
                "demo": _demo_label(d),
                "model": meta.get("model"),
            })
        return {"flights": flights, "current_flight_id": self.newest_flight_id(),
                "active": self.is_flight_active()}

    def get_flight(self, flight_id: str) -> Optional[dict]:
        d = self._dir(flight_id, must_exist=False)
        if not d.is_dir():
            return None
        return {
            "flight_id": flight_id,
            "meta": _read_json(d / "meta.json"),
            "status": _read_json(d / "status.json"),
            "summary": _read_json(d / "summary.json"),
            "running": self._pid_alive(flight_id),
            "camera_enabled": (d / CAMERA_FLAG).exists(),
            "land_requested": (d / LAND_FLAG).exists(),
            "kill_requested": (d / KILL_FLAG).exists(),
            "save_answered": (d / SAVE_ANSWER).exists(),
            "has_camera_frame": (d / "camera_latest.jpg").exists(),
            # the detached ClearML upload (fly_hoop -> clearml_sync.py record) is still running
            "clearml_uploading": (d / "clearml_uploading.flag").exists(),
            "demo": _demo_label(d),
        }

    def events(self, flight_id: str) -> list:
        return self._jsonl(self._dir(flight_id) / "events.jsonl")

    def ticks(self, flight_id: str, max_points: int = 1500) -> list:
        rows = [r for r in self._jsonl(self._dir(flight_id) / "ticks.jsonl")
                if r.get("type") == "tick"]
        step = max(1, len(rows) // max_points)
        return rows[::step]

    @staticmethod
    def _jsonl(path: Path) -> list:
        out = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass          # the line being written right now
        except OSError:
            pass
        return out

    def status(self) -> dict:
        """What the panel polls: the newest flight, whatever its state."""
        fid = self.newest_flight_id()
        if fid is None:
            return {"flight_id": None, "active": False, "status": None,
                    "modes": MODES, "profiles": PROFILES}
        f = self.get_flight(fid) or {}
        return {**f, "active": self.is_flight_active(), "modes": MODES,
                "profiles": PROFILES}

    # ------------------------------------------------------------------ control

    def _dir(self, flight_id: str, must_exist=True) -> Path:
        # Resolve and confine: flight_id reaches here from a URL path segment, so a
        # traversal attempt ("../..") must not let a file be written anywhere else.
        d = (self.flights_dir / flight_id).resolve()
        if d.parent != self.flights_dir.resolve() or not flight_id.startswith(FLIGHT_PREFIX):
            raise ValueError(f"not a flight id: '{flight_id}'")
        if must_exist and not d.is_dir():
            raise ValueError(f"no such flight '{flight_id}'")
        return d

    def _write_flag(self, flight_id: str, name: str, enabled: bool):
        flag = self._dir(flight_id) / name
        if enabled:
            flag.touch(exist_ok=True)
        else:
            flag.unlink(missing_ok=True)

    def request_land(self, flight_id: str):
        """Always allowed, even if the flight looks finished: refusing on a stale
        'not running' reading would be the one failure mode that actually matters."""
        self._write_flag(flight_id, LAND_FLAG, True)

    def request_kill(self, flight_id: str):
        """Cut the motors. The drone drops, so this is for 'it is out of control' only."""
        self._write_flag(flight_id, KILL_FLAG, True)

    def set_camera(self, flight_id: str, enabled: bool):
        self._write_flag(flight_id, CAMERA_FLAG, enabled)

    def _keys_port(self, flight_id: str) -> Optional[int]:
        """The flight's keys port from meta.json, or None once it has ended.

        Re-read at most every KEYS_META_TTL_S. This runs for every key message -- ten a
        second from the heartbeat alone, inside the websocket's event loop -- and reading
        a OneDrive-synced file that often both stalled the loop and, whenever the read
        failed, silently dropped that key change. A failed re-read keeps the last answer.
        """
        now = time.monotonic()
        hit = self._keys_ports.get(flight_id)
        if hit is not None and now - hit[1] < KEYS_META_TTL_S:
            return hit[0]
        meta = _read_json(self._dir(flight_id) / "meta.json")
        if meta is None:
            return hit[0] if hit is not None else None
        port = meta.get("keys_port")
        if not isinstance(port, int) or meta.get("final_status"):
            port = None
        self._keys_ports[flight_id] = (port, now)
        return port

    def send_keys(self, flight_id: str, held) -> bool:
        """Forward held keys to a live human flight. Unknown names are dropped here and
        again in the flight. Returns False if the flight takes no keys."""
        port = self._keys_port(flight_id)
        if port is None:
            return False
        names = [k for k in held if k in HELD_KEYS]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(json.dumps({"held": names}).encode(), ("127.0.0.1", port))
        return True

    def answer_save(self, flight_id: str, save: bool, passed: bool):
        """The post-landing 'save this run?' answer, read by fly_hoop --save-answer file."""
        write_json_atomic(self._dir(flight_id) / SAVE_ANSWER,
                          {"save": bool(save), "passed": bool(passed)})

    def camera_frame_path(self, flight_id: str) -> Path:
        return self._dir(flight_id) / "camera_latest.jpg"

    # ------------------------------------------------------------------ starting

    def build_argv(self, mode="expert", profile="typical", seed=0, attempts=1,
                   no_forward=False, no_commit=False, max_travel=None,
                   max_airborne=150.0, record=False, model=None,
                   dry_run: bool = True, confirm: bool = False,
                   human_gain: float = 0.6, keys_port: Optional[int] = None,
                   max_height: int = 190, min_height: int = 10, start_height: int = 80,
                   keys: str = "popup", clearml: bool = True,
                   detector_preset: Optional[str] = None) -> list:
        """fly_hoop.py's command line for a dashboard flight.

        keys="popup" (default since 2026-10-14): a human flight opens fly_hoop's own pygame
        window, the same one a terminal flight uses -- click it and fly from it. Keys then go
        straight from the OS to the flight loop, with none of the browser -> websocket -> UDP
        hops (and the 1 s status poll before they engage) that made dashboard flying lag.
        "browser" keeps the old path (keys typed in the dashboard)."""
        if not dry_run and not confirm:
            raise ValueError("a real flight needs confirm=true")
        if mode not in MODE_IDS:
            raise ValueError(f"unknown mode '{mode}'")
        if profile not in PROFILE_IDS:
            raise ValueError(f"unknown profile '{profile}'")
        if mode == "policy" and not model:
            raise ValueError("mode 'policy' needs a model")
        if model:
            model = Path(model) if Path(model).is_absolute() else CMS_DIR / model
            if not model.is_file():
                raise ValueError(f"no model file at '{model}'")
        if not 0.0 < float(human_gain) <= 1.0:
            raise ValueError("human_gain must be in (0, 1]")
        if max_travel is not None and float(max_travel) <= 0:
            raise ValueError("max_travel must be positive")
        if not 0 < int(min_height) < int(max_height) <= 250:
            raise ValueError("need 0 < min_height < max_height <= 250 cm")
        if not 0 <= int(start_height) < int(max_height):
            raise ValueError("start_height must be below max_height")
        if keys not in ("popup", "browser"):
            raise ValueError("keys must be 'popup' or 'browser'")

        argv = [PYTHON_EXE, str(FLY_HOOP_SCRIPT),
                "--mode", mode, "--seed", str(int(seed)),
                "--max-airborne", str(float(max_airborne)), "--camera",
                # the camera shows from the countdown on, not only after takeoff
                "--preview-before-takeoff",
                "--max-height", str(int(max_height)), "--min-height", str(int(min_height)),
                "--start-height", str(int(start_height))]
        if dry_run:
            argv += ["--dry-run", "--profile", profile,
                     "--attempts", str(max(1, int(attempts))),
                     "--countdown", "1", "--settle", "1.0"]
        else:
            argv += ["--countdown", "3"]        # time to step back from the cage
        if no_forward:
            argv.append("--no-forward")
        if no_commit:
            argv.append("--no-commit")
        if max_travel is not None:
            argv += ["--max-travel", str(float(max_travel))]
        if mode == "human":
            if keys == "browser":
                argv += ["--keys", "udp", "--keys-port", str(keys_port or _free_udp_port())]
            else:
                argv += ["--keys", "pygame"]
            argv += ["--human-gain", str(float(human_gain)), "--save-answer", "file"]
        if not clearml:
            argv.append("--no-clearml")
        # fly_hoop records human flights by default; the dashboard's checkbox decides.
        argv.append("--record" if record else "--no-record")
        if model:
            argv += ["--model", str(model)]
        if detector_preset and detector_preset != "default":
            from calib_manager import presets
            try:
                argv += ["--detector", presets.get_preset(detector_preset)["slug"]]
            except KeyError as e:
                raise ValueError(str(e).strip("'\""))
        return argv

    def start_flight(self, **kw) -> str:
        """Start a flight. A real one needs confirm=True; main.py also requires the
        request to come from this computer."""
        argv = self.build_argv(**kw)
        if self.is_flight_active():
            raise RuntimeError("a flight is already active")
        if not FLY_HOOP_SCRIPT.is_file():
            raise RuntimeError(f"missing {FLY_HOOP_SCRIPT}")
        self.flights_dir.mkdir(parents=True, exist_ok=True)
        prefix = "dryrun" if kw.get("dry_run", True) else "flight"
        log_path = self.flights_dir / f"{prefix}_{datetime.now():%Y%m%d-%H%M%S}.log"
        # fly_hoop.py makes its own timestamped directory, so the id is not known until
        # it appears. The caller polls /api/drone/status for it.
        self._procs[log_path.stem] = spawn(argv, log_path, CMS_DIR)
        return log_path.stem

    def stop_flight(self, flight_id: Optional[str] = None) -> bool:
        """Stop a dry run this backend started, or ask a real flight to land.

        A real flight is never killed by signal from here: landing is a sequence the
        flight itself has to carry out, and killing the process mid-air would leave the
        drone hovering with no supervisor until its own battery timeout.
        """
        if flight_id:
            meta = _read_json(self.flights_dir / flight_id / "meta.json") or {}
            if meta.get("dry_run") is False:
                self.request_land(flight_id)
                return True
        stopped = False
        for key, proc in list(self._procs.items()):
            if self._proc_alive(proc):
                graceful_stop(proc, timeout=15)
                stopped = True
            self._procs.pop(key, None)
        if not stopped and flight_id:
            self.request_land(flight_id)
            stopped = True
        return stopped


drone_manager = DroneManager()
