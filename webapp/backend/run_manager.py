"""
Starts, stops, and lists training/replay runs. The backend never imports
stable_baselines3/pybullet/torch itself: everything here is subprocess
management plus reading the small JSON/JSONL files those subprocesses write.
The filesystem tree under results/hoop/runs/ is the database, there is no
separate registry file, so this survives a backend restart with no special
recovery code (see the plan's "no shared registry file" note).
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

import jobs
from config import SCRIPTS_DIR, RUNS_DIR, CMS_DIR, ALGORITHM_IDS, bounds_ok

PYTHON_EXE = sys.executable  # the drones env's own interpreter, since this
                              # backend process runs inside that same env


def _popen_kwargs(log_path: Path):
    return jobs.popen_kwargs(log_path, SCRIPTS_DIR)


class RunManager:
    def __init__(self):
        self._training_procs: dict[str, subprocess.Popen] = {}
        self._replay_procs: dict[str, subprocess.Popen] = {}
        self._showcase_procs: dict[str, subprocess.Popen] = {}   # run_id -> proc

    # ---------------------------------------------------------------- runs

    def is_training_active(self) -> bool:
        return any(self._proc_alive(p) for p in self._training_procs.values())

    def is_replay_active(self) -> bool:
        return any(self._proc_alive(p) for p in self._replay_procs.values())

    @staticmethod
    def _proc_alive(proc: subprocess.Popen) -> bool:
        return proc.poll() is None

    def start_run(self, algo: str, seed: Optional[int], start_pos, target_pos,
                  timesteps: int, n_envs: int, eval_freq: int, n_eval_episodes: int,
                  env: str = "net", ring_jitter: float = 1.0, radius_start: float = 1.5,
                  sway_deg: float = 20.0, until_stopped: bool = False) -> str:
        if algo not in ALGORITHM_IDS:
            raise ValueError(f"unknown algo '{algo}'")
        if not bounds_ok(start_pos) or not bounds_ok(target_pos):
            raise ValueError("start_pos/target_pos outside the room bounds")
        if self.is_training_active():
            raise RuntimeError("a training run is already active")

        run_id = f"{algo}_{datetime.now():%Y%m%d-%H%M%S}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        argv = [
            PYTHON_EXE, str(SCRIPTS_DIR / "train_hoop_multi.py"),
            "--algo", algo,
            "--timesteps", str(timesteps),
            "--n-envs", str(n_envs),
            "--eval-freq", str(eval_freq),
            "--n-eval-episodes", str(n_eval_episodes),
            "--run-id", run_id,
            "--out", str(run_dir),
            "--env", env,
        ]
        if env == "net":
            argv += ["--ring-jitter", str(ring_jitter), "--radius-start", str(radius_start),
                     "--sway-deg", str(sway_deg)]
        if until_stopped:
            argv.append("--until-stopped")
        if seed is not None:
            argv += ["--seed", str(seed)]
        if start_pos is not None:
            argv += ["--start-pos", *(str(v) for v in start_pos)]
        if target_pos is not None:
            argv += ["--target-pos", *(str(v) for v in target_pos)]

        log_path = run_dir / "train.log"
        kwargs = _popen_kwargs(log_path)
        proc = subprocess.Popen(argv, **kwargs)
        kwargs["stdout"].close()  # the child has its own duplicated handle now
        self._training_procs[run_id] = proc
        return run_id

    def stop_run(self, run_id: str) -> bool:
        proc = self._training_procs.get(run_id)
        if proc is not None and self._proc_alive(proc):
            self._graceful_stop(proc)
            return True
        # No live handle (e.g. the backend restarted); fall back to the pid
        # meta.json recorded at start.
        meta_path = RUNS_DIR / run_id / "meta.json"
        if meta_path.is_file():
            try:
                pid = json.loads(meta_path.read_text()).get("pid")
                if pid and psutil.pid_exists(pid):
                    psutil.Process(pid).terminate()
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _graceful_stop(proc: subprocess.Popen):
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def list_runs(self):
        runs = []
        if not RUNS_DIR.is_dir():
            return {"runs": [], "current_run_id": None, "best_run_id": None}
        for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime):
            if not d.is_dir():
                continue
            meta = _read_json(d / "meta.json")
            if meta is None:
                continue
            status_doc = _read_json(d / "live_status.json") or {}
            status = self._status(d.name, meta, status_doc)
            best = status_doc.get("best_eval_reward")
            runs.append({
                "run_id": meta.get("run_id", d.name),
                "algo": meta.get("algo"),
                "env": meta.get("env", "legacy"),
                "has_model": any((d / n).is_file() for n in
                                 ("latest_model.zip", "best_model.zip", "final_model.zip")),
                "status": status,
                "current_timestep": (status_doc.get("current_timestep")
                                     or meta.get("final_timesteps") or 0),
                "total_timesteps": meta.get("timesteps_requested"),
                "until_stopped": bool(meta.get("until_stopped")),
                "elapsed_sec": status_doc.get("elapsed_sec"),
                "best_eval_reward": best,
                "created_at": meta.get("created_at"),
            })
        current_run_id = runs[-1]["run_id"] if runs else None
        scored = [r for r in runs if r["best_eval_reward"] is not None]
        best_run_id = max(scored, key=lambda r: r["best_eval_reward"])["run_id"] if scored else None
        return {"runs": runs, "current_run_id": current_run_id, "best_run_id": best_run_id}

    def get_run(self, run_id: str):
        d = RUNS_DIR / run_id
        meta = _read_json(d / "meta.json")
        if meta is None:
            return None
        live_status = _read_json(d / "live_status.json")
        curve = _read_curve(d / "evaluations.npz")
        return {"meta": meta, "live_status": live_status, "curve": curve,
                "status": self._status(run_id, meta, live_status or {})}

    def _status(self, run_id: str, meta: dict, status_doc: dict) -> str:
        """final_status once the run has ended; otherwise "training" while its process is alive,
        even before the first evaluation has written live_status.json (~40 s into a run)."""
        if meta.get("final_status"):
            return meta["final_status"]
        proc = self._training_procs.get(run_id)
        if proc is not None:
            return "training" if self._proc_alive(proc) else (status_doc.get("status") or "unknown")
        pid = meta.get("pid")
        if pid:
            if psutil.pid_exists(pid):
                return "training"
            # process gone without writing final_status: don't leave a stale "training" behind,
            # it would keep the Training panel locked
            return "crashed" if status_doc.get("status") in (None, "training") else status_doc["status"]
        return status_doc.get("status") or "unknown"

    # ------------------------------------------------------------- replay

    def start_replay(self, run_id: str, variant: str, episodes: int, pov: bool) -> str:
        run_dir = RUNS_DIR / run_id
        if not (run_dir / "meta.json").is_file():
            raise ValueError(f"unknown run '{run_id}'")
        if self.is_training_active():
            raise RuntimeError("a training run is active, replay is disabled meanwhile")
        if self.is_replay_active():
            raise RuntimeError("a replay is already being generated")

        replay_id = f"{variant}_{datetime.now():%Y%m%d-%H%M%S}"
        replay_dir = run_dir / "replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            PYTHON_EXE, str(SCRIPTS_DIR / "replay_hoop.py"),
            "--run-dir", str(run_dir),
            "--variant", variant,
            "--replay-id", replay_id,
            "--episodes", str(episodes),
        ]
        if pov:
            argv.append("--pov")
        log_path = replay_dir / f"{replay_id}.log"
        kwargs = _popen_kwargs(log_path)
        proc = subprocess.Popen(argv, **kwargs)
        kwargs["stdout"].close()
        self._replay_procs[replay_id] = proc
        return replay_id

    def get_replay_status(self, run_id: str, replay_id: str, pov: bool = False):
        replay_dir = RUNS_DIR / run_id / "replay"
        video = replay_dir / f"{replay_id}.mp4"
        pov_video = replay_dir / f"{replay_id}_pov.mp4"
        proc = self._replay_procs.get(replay_id)
        still_running = proc is not None and self._proc_alive(proc)
        if video.is_file() and (not pov or pov_video.is_file()):
            return {"status": "done",
                    "video_url": f"/media/{run_id}/replay/{video.name}",
                    "pov_video_url": (f"/media/{run_id}/replay/{pov_video.name}"
                                       if pov_video.is_file() else None)}
        if still_running:
            return {"status": "rendering"}
        return {"status": "failed"}

    # ------------------------------------------------------------- showcase
    # One real-time episode of the run's current policy (scripts/showcase_hoop.py), allowed
    # while training is running: it is a separate process with its own simulator.

    def is_showcase_active(self, run_id: str) -> bool:
        proc = self._showcase_procs.get(run_id)
        return proc is not None and self._proc_alive(proc)

    def start_showcase(self, run_id: str, variant: str = "latest", seed: Optional[int] = None,
                       realtime: bool = False):
        run_dir = RUNS_DIR / run_id
        if not (run_dir / "meta.json").is_file():
            raise ValueError(f"unknown run '{run_id}'")
        if self.is_showcase_active(run_id):
            raise RuntimeError("a showcase is already playing for this run")
        names = ["latest_model.zip", "best_model.zip", "final_model.zip"]
        if variant == "best":
            names = ["best_model.zip", "latest_model.zip", "final_model.zip"]
        if not any((run_dir / n).is_file() for n in names):
            raise FileNotFoundError("no model saved yet: wait for the first evaluation")
        # Truncate first, so a connected socket (which tails from its last offset) sees the
        # file shrink and restarts from the top of this new showcase.
        (run_dir / "showcase.jsonl").write_text("")
        argv = [PYTHON_EXE, str(SCRIPTS_DIR / "showcase_hoop.py"),
                "--run-dir", str(run_dir), "--variant", variant]
        if seed is not None:
            argv += ["--seed", str(seed)]
        if realtime:
            argv.append("--realtime")
        kwargs = _popen_kwargs(run_dir / "showcase.log")
        proc = subprocess.Popen(argv, **kwargs)
        kwargs["stdout"].close()
        self._showcase_procs[run_id] = proc

    # ------------------------------------------------------------- realtime
    # realtime.flag paces the training run so env 0 flies at real speed (PaceCallback in
    # scripts/telemetry_callback.py); switchable mid-run, like the camera flag.

    def set_realtime(self, run_id: str, enabled: bool):
        run_dir = RUNS_DIR / run_id
        if not run_dir.is_dir():
            raise ValueError(f"unknown run '{run_id}'")
        flag = run_dir / "realtime.flag"
        if enabled:
            flag.touch(exist_ok=True)
        else:
            flag.unlink(missing_ok=True)

    # --------------------------------------------------------------- camera

    def set_camera(self, run_id: str, enabled: bool):
        run_dir = RUNS_DIR / run_id
        flag = run_dir / "camera_enabled.flag"
        if not run_dir.is_dir():
            raise ValueError(f"unknown run '{run_id}'")
        if enabled:
            flag.touch(exist_ok=True)
        else:
            flag.unlink(missing_ok=True)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_curve(path: Path):
    if not path.is_file():
        return None
    try:
        import numpy as np
        with np.load(path) as data:
            timesteps = data["timesteps"].tolist()
            results = data["results"]
            mean_reward = results.mean(axis=1).tolist()
            std_reward = results.std(axis=1).tolist()
        return {"timesteps": timesteps, "mean_reward": mean_reward, "std_reward": std_reward}
    except Exception:
        return None


run_manager = RunManager()
