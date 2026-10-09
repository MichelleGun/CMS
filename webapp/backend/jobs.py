"""
Subprocess helpers shared by every manager, and a small registry for one-off jobs
(make demos, train a clone, evaluate a policy).

Each job writes its log to hardware/flights/jobs/<job_id>.log and its record to
<job_id>.json, so the job history survives a backend restart: the folder is the
database, the same rule run_manager.py and drone_manager.py follow.
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

from config import CMS_DIR, FLIGHTS_DIR

PYTHON_EXE = sys.executable          # this backend runs inside the drones env already
JOBS_DIR = FLIGHTS_DIR / "jobs"


def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None          # a half-written atomic replace, or no file yet


def write_json_atomic(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def subprocess_env():
    """PATH with the conda env's Library/bin (ffmpeg) and Scripts in front."""
    env = dict(os.environ)
    root = Path(PYTHON_EXE).resolve().parent
    extra = [str(root / "Library" / "bin"), str(root / "Scripts"), str(root)]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    env["PYTHONUNBUFFERED"] = "1"    # so a tailed log shows progress as it happens
    return env


def popen_kwargs(log_path: Path, cwd: Path):
    # A log file, never subprocess.PIPE: an unread pipe's OS buffer fills and stalls the
    # child (a confirmed hang with replay_hoop.py --pov).
    log_file = open(log_path, "w")
    kwargs = dict(cwd=str(cwd), env=subprocess_env(), stdout=log_file,
                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        # So CTRL_BREAK_EVENT reaches the child alone; fly_hoop.py treats it as Ctrl+C,
        # which lands the drone.
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return kwargs


def spawn(argv, log_path: Path, cwd: Path) -> subprocess.Popen:
    kwargs = popen_kwargs(log_path, cwd)
    try:
        return subprocess.Popen(argv, **kwargs)
    finally:
        kwargs["stdout"].close()     # the child holds its own duplicated handle now


def graceful_stop(proc, timeout=10):
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        try:
            proc.kill()
        except OSError:
            pass


def pid_alive(pid) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def tail_text(path: Path, offset: int = 0, max_bytes: int = 65536):
    """-> (text, new_offset). Restarts from 0 if the file shrank."""
    try:
        size = path.stat().st_size
    except OSError:
        return "", offset
    if offset > size:
        offset = 0
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace"), offset + len(data)


class JobManager:
    """One job at a time. A job is a script run with a log and a few output paths."""

    def __init__(self, jobs_dir: Path = JOBS_DIR):
        self.dir = jobs_dir
        self._procs: dict[str, subprocess.Popen] = {}

    def _record_path(self, job_id: str) -> Path:
        if not job_id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"not a job id: '{job_id}'")
        return self.dir / f"{job_id}.json"

    def active_job(self) -> Optional[dict]:
        for job_id, proc in self._procs.items():
            if proc.poll() is None:
                return self.get(job_id)
        return None

    def start(self, kind: str, argv, label: str, outputs: Optional[dict] = None,
              cwd: Path = CMS_DIR) -> str:
        if self.active_job() is not None:
            raise RuntimeError("another job is still running; wait for it or stop it")
        self.dir.mkdir(parents=True, exist_ok=True)
        job_id = f"{kind}_{datetime.now():%Y%m%d-%H%M%S}"
        log = self.dir / f"{job_id}.log"
        proc = spawn([str(a) for a in argv], log, cwd)
        self._procs[job_id] = proc
        write_json_atomic(self._record_path(job_id), {
            "job_id": job_id, "kind": kind, "label": label, "pid": proc.pid,
            "argv": [str(a) for a in argv], "outputs": outputs or {},
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None, "returncode": None,
        })
        return job_id

    def _refresh(self, rec: dict) -> dict:
        """Fill in the ending of a job this process saw finish."""
        if rec.get("returncode") is not None:
            return rec
        proc = self._procs.get(rec["job_id"])
        code = proc.poll() if proc is not None else None
        if proc is not None and code is not None:
            rec.update(returncode=code,
                       finished_at=datetime.now().isoformat(timespec="seconds"))
            try:
                write_json_atomic(self._record_path(rec["job_id"]), rec)
            except OSError:
                pass
        return rec

    def get(self, job_id: str) -> Optional[dict]:
        rec = read_json(self._record_path(job_id))
        if rec is None:
            return None
        rec = self._refresh(rec)
        running = rec.get("returncode") is None and (
            (self._procs.get(job_id) is not None and self._procs[job_id].poll() is None)
            or (job_id not in self._procs and pid_alive(rec.get("pid"))))
        rec["status"] = ("running" if running else
                         "done" if rec.get("returncode") == 0 else
                         "failed" if rec.get("returncode") is not None else "ended")
        return rec

    def list(self, limit: int = 50) -> list:
        if not self.dir.is_dir():
            return []
        # ids are <kind>_<timestamp>; newest first means sorting on the timestamp part
        recs = sorted(self.dir.glob("*.json"), key=lambda p: p.stem.split("_", 1)[-1],
                      reverse=True)[:limit]
        return [r for r in (self.get(p.stem) for p in recs) if r is not None]

    def log(self, job_id: str, offset: int = 0):
        self._record_path(job_id)
        return tail_text(self.dir / f"{job_id}.log", offset)

    def stop(self, job_id: str) -> bool:
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            graceful_stop(proc)
            return True
        rec = read_json(self._record_path(job_id)) or {}
        if rec.get("returncode") is None and pid_alive(rec.get("pid")):
            psutil.Process(rec["pid"]).terminate()
            return True
        return False


job_manager = JobManager()
