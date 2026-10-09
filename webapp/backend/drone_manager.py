"""
The real-drone side of the dashboard: list flights, read their live state, and send the
two commands a browser is allowed to send.

Mirrors run_manager.py deliberately -- same subprocess handling, same "the directory tree
IS the database" approach, so a backend restart needs no recovery code. It reads what
hardware/onboard/fly_hoop.py writes under hardware/flights/<flight_id>/ and never imports
djitellopy, OpenCV or the flight code itself.

TWO SAFETY DECISIONS ARE BUILT IN HERE, not in the UI, so they cannot be clicked around.

1. The browser cannot take off a real drone. start_flight refuses anything but a dry run.
   A real flight is started from a terminal by a person standing next to the cage, and the
   dashboard attaches to it by reading the newest flight directory. The reason is latency
   and staleness: a page left open in a forgotten tab, a double-click, or a reconnecting
   socket must not be able to spin propellers in another room.

2. Commands are FLAG FILES, not a socket or a signal. fly_hoop.py polls for land.flag and
   kill.flag every tick. This is the same mechanism the training side already uses for its
   camera toggle, and it is fail-safe in the direction that matters: if this backend dies,
   no flag appears and the flight's own safety layer still lands the drone; if the flight
   process dies, a flag is simply never read and nothing is left half-commanded. Both
   flags only ever make the drone safer, so there is no authorisation question -- the worst
   a stranger with access to the page can do is land a drone.
"""
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

from config import CMS_DIR, FLIGHTS_DIR, FLY_HOOP_SCRIPT

PYTHON_EXE = sys.executable          # this backend runs inside the drones env already

FLIGHT_PREFIX = "tello_"
LAND_FLAG = "land.flag"
KILL_FLAG = "kill.flag"
CAMERA_FLAG = "camera_enabled.flag"

MODES = [
    {"id": "expert", "label": "Scripted expert"},
    {"id": "policy", "label": "Trained policy"},
    {"id": "hover", "label": "Hover and detect only"},
]
PROFILES = [
    {"id": "ideal", "label": "Ideal"},
    {"id": "typical", "label": "Typical"},
    {"id": "harsh", "label": "Harsh"},
]
MODE_IDS = {m["id"] for m in MODES}
PROFILE_IDS = {p["id"] for p in PROFILES}


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None          # a half-written atomic replace, or no file yet


def _subprocess_env():
    env = dict(os.environ)
    root = Path(PYTHON_EXE).resolve().parent
    extra = [str(root / "Library" / "bin"), str(root / "Scripts"), str(root)]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _popen_kwargs(log_path: Path):
    # A log file, never subprocess.PIPE: run_manager.py records a confirmed hang when an
    # unread pipe's OS buffer filled, and the flight loop prints every tick with --verbose.
    log_file = open(log_path, "w")
    kwargs = dict(cwd=str(CMS_DIR), env=_subprocess_env(),
                  stdout=log_file, stderr=subprocess.STDOUT)
    if os.name == "nt":
        # So CTRL_BREAK_EVENT can be delivered to the child alone. fly_hoop.py treats a
        # KeyboardInterrupt as "land now", which is why a break is the right stop signal.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return kwargs


class DroneManager:
    def __init__(self):
        self._procs: dict[str, subprocess.Popen] = {}

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
        meta = _read_json(FLIGHTS_DIR / flight_id / "meta.json") or {}
        if meta.get("final_status"):
            return False          # it recorded its own ending
        pid = meta.get("pid")
        if not isinstance(pid, int):
            return False
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            return False

    def _flight_dirs(self):
        if not FLIGHTS_DIR.is_dir():
            return []
        dirs = [p for p in FLIGHTS_DIR.iterdir()
                if p.is_dir() and p.name.startswith(FLIGHT_PREFIX)]
        return sorted(dirs, key=lambda p: p.name, reverse=True)   # name is a timestamp

    def newest_flight_id(self) -> Optional[str]:
        dirs = self._flight_dirs()
        return dirs[0].name if dirs else None

    def list_flights(self, limit: int = 25) -> dict:
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
        }

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
        # traversal attempt ("../..") must not let a flag file be written anywhere else.
        d = (FLIGHTS_DIR / flight_id).resolve()
        if d.parent != FLIGHTS_DIR.resolve() or not flight_id.startswith(FLIGHT_PREFIX):
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
        """Ask the flight to land. Always allowed, even if the flight looks finished:
        refusing on a stale 'not running' reading would be the one failure mode that
        actually matters."""
        self._write_flag(flight_id, LAND_FLAG, True)

    def request_kill(self, flight_id: str):
        """Cut the motors. The drone drops, so this is for 'it is out of control' only --
        but it is still strictly safer than leaving it flying, so the browser may send it."""
        self._write_flag(flight_id, KILL_FLAG, True)

    def set_camera(self, flight_id: str, enabled: bool):
        self._write_flag(flight_id, CAMERA_FLAG, enabled)

    def camera_frame_path(self, flight_id: str) -> Path:
        return self._dir(flight_id) / "camera_latest.jpg"

    # ------------------------------------------------------------------ starting

    def start_flight(self, mode="expert", profile="typical", seed=0, attempts=1,
                     no_forward=False, no_commit=False, max_travel=None,
                     max_airborne=40.0, record=False, model=None,
                     dry_run: bool = True) -> str:
        """Start a DRY RUN. Refuses a real flight -- see this module's docstring."""
        if not dry_run:
            raise ValueError(
                "the dashboard cannot start a real flight. Run it from a terminal next "
                "to the cage: python hardware/onboard/fly_hoop.py  (the dashboard will "
                "pick it up automatically and can land or stop it)")
        if mode not in MODE_IDS:
            raise ValueError(f"unknown mode '{mode}'")
        if profile not in PROFILE_IDS:
            raise ValueError(f"unknown profile '{profile}'")
        if mode == "policy" and not model:
            raise ValueError("mode 'policy' needs a model path")
        if self.is_flight_active():
            raise RuntimeError("a flight is already active")
        if not FLY_HOOP_SCRIPT.is_file():
            raise RuntimeError(f"missing {FLY_HOOP_SCRIPT}")

        argv = [PYTHON_EXE, str(FLY_HOOP_SCRIPT), "--dry-run",
                "--mode", mode, "--profile", profile, "--seed", str(int(seed)),
                "--attempts", str(max(1, int(attempts))),
                "--max-airborne", str(float(max_airborne)),
                "--countdown", "1", "--settle", "1.0"]
        if no_forward:
            argv.append("--no-forward")
        if no_commit:
            argv.append("--no-commit")
        if max_travel is not None:
            argv += ["--max-travel", str(float(max_travel))]
        if record:
            argv.append("--record")
        if model:
            argv += ["--model", str(model)]

        FLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = FLIGHTS_DIR / f"dryrun_{datetime.now():%Y%m%d-%H%M%S}.log"
        kwargs = _popen_kwargs(log_path)
        proc = subprocess.Popen(argv, **kwargs)
        kwargs["stdout"].close()      # the child holds its own duplicated handle now
        # fly_hoop.py makes its own timestamped directory, so the id is not known until
        # it appears. The caller polls /api/drone/status for it.
        self._procs[log_path.stem] = proc
        return log_path.stem

    def stop_flight(self, flight_id: Optional[str] = None) -> bool:
        """Stop a dry run this backend started, or ask a real flight to land.

        A real flight is never killed by signal from here: landing is a sequence the
        flight itself has to carry out, and killing the process mid-air would leave the
        drone hovering with no supervisor until its own battery timeout.
        """
        if flight_id:
            meta = _read_json(FLIGHTS_DIR / flight_id / "meta.json") or {}
            if meta.get("dry_run") is False:
                self.request_land(flight_id)
                return True
        stopped = False
        for key, proc in list(self._procs.items()):
            if self._proc_alive(proc):
                self._graceful_stop(proc)
                stopped = True
            self._procs.pop(key, None)
        if not stopped and flight_id:
            self.request_land(flight_id)
            stopped = True
        return stopped

    @staticmethod
    def _graceful_stop(proc):
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=15)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            try:
                proc.kill()
            except OSError:
                pass


drone_manager = DroneManager()
