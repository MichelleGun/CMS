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
import run_config
from config import (SCRIPTS_DIR, RUNS_DIR, CMS_DIR, ALGORITHM_IDS, START_MARGIN, FLIGHTS_DIR,
                    HARDWARE_DIR, NET_OBS_DIMS, bounds_ok)

PYTHON_EXE = sys.executable  # the drones env's own interpreter, since this
                              # backend process runs inside that same env

# Several runs may train side by side. Each run's env copies are separate processes (one core
# each, see scripts/vec_util.py), so the limit that matters is cores: a new run is refused if
# the env processes of every live run would exceed the CPU budget, with a hint to use fewer
# envs, rather than silently slowing every run down.
MAX_CONCURRENT_RUNS = 3
CPU_BUDGET = max(1, (os.cpu_count() or 2) - 2)     # leave headroom for the dashboard + trainers
ONBOARD_TRAIN = HARDWARE_DIR / "onboard" / "train_onboard.py"
BC_DIR = FLIGHTS_DIR / "bc"
ONBOARD_TASK = "onboard_hoop"


def _popen_kwargs(log_path: Path):
    return jobs.popen_kwargs(log_path, SCRIPTS_DIR)


def _env_procs(meta: dict) -> int:
    """Cores a run's env copies occupy: one per copy when they run as subprocesses. Runs
    from before meta.json recorded "vec" all stepped their copies in one process."""
    return int(meta.get("n_envs") or 1) if meta.get("vec") == "subproc" else 1


class RunManager:
    def __init__(self):
        self._training_procs: dict[str, subprocess.Popen] = {}
        self._launch_meta: dict[str, dict] = {}    # run_id -> n_envs/vec, until meta.json exists
        self._replay_procs: dict[str, subprocess.Popen] = {}
        self._success_cache: dict[str, tuple] = {}   # run_id -> (npz mtime, best success rate)
        self._showcase_procs: dict[str, subprocess.Popen] = {}   # run_id -> proc

    # ---------------------------------------------------------------- runs

    def is_training_active(self) -> bool:
        return bool(self.live_runs())

    def is_replay_active(self) -> bool:
        return any(self._proc_alive(p) for p in self._replay_procs.values())

    @staticmethod
    def _proc_alive(proc: subprocess.Popen) -> bool:
        return proc.poll() is None

    def live_runs(self) -> list:
        """meta.json of every run whose process is alive: started here, from a terminal, or
        before the backend restarted. All of them count against the limits."""
        out = {}
        if RUNS_DIR.is_dir():
            for d in RUNS_DIR.iterdir():
                meta = _read_json(d / "meta.json") if d.is_dir() else None
                if meta and self._status(d.name, meta, {}) == "training":
                    out[d.name] = meta
        # A run launched a moment ago has not written meta.json yet (importing torch takes
        # seconds) but already counts: otherwise quick clicks could start a 4th run.
        for run_id, proc in self._training_procs.items():
            if run_id not in out and self._proc_alive(proc):
                out[run_id] = self._launch_meta.get(run_id, {})
        return list(out.values())

    def capacity(self) -> dict:
        live = self.live_runs()
        return {"live_runs": len(live), "max_runs": MAX_CONCURRENT_RUNS,
                "cpu_used": sum(_env_procs(m) for m in live), "cpu_budget": CPU_BUDGET}

    def _new_run_id(self, prefix: str) -> str:
        base = f"{prefix}_{datetime.now():%Y%m%d-%H%M%S}"
        run_id, k = base, 2
        while (RUNS_DIR / run_id).exists():            # two runs started in the same second
            run_id, k = f"{base}-{k}", k + 1
        return run_id

    def _check_capacity(self, n_envs: int, vec: str):
        cap = self.capacity()
        if cap["live_runs"] >= MAX_CONCURRENT_RUNS:
            raise RuntimeError(f"{cap['live_runs']} runs are already training (the limit is "
                               f"{MAX_CONCURRENT_RUNS}); stop one first")
        want = n_envs if (vec == "subproc" or (vec == "auto" and n_envs > 1)) else 1
        free = cap["cpu_budget"] - cap["cpu_used"]
        if want > free:
            raise RuntimeError(f"not enough free cores: live runs use {cap['cpu_used']} of "
                               f"{cap['cpu_budget']}, this run wants {want}. Use "
                               f"{max(1, free)} parallel envs or fewer, or stop a run")

    def _init_dir(self, init_run: Optional[str], init_clone: Optional[str]) -> Optional[Path]:
        if init_run and init_clone:
            raise ValueError("continue from a run OR a clone, not both")
        if init_run:
            d = (RUNS_DIR / init_run).resolve()
            if d.parent != RUNS_DIR.resolve() or not (d / "meta.json").is_file():
                raise ValueError(f"unknown run '{init_run}'")
        elif init_clone:
            d = (BC_DIR / init_clone).resolve()
            if d.parent != BC_DIR.resolve() or not (d / "bc_model.zip").is_file():
                raise ValueError(f"unknown clone '{init_clone}'")
        else:
            return None
        if not any((d / n).is_file() for n in ("latest_model.zip", "best_model.zip",
                                                "final_model.zip", "bc_model.zip")):
            raise ValueError(f"'{d.name}' has no saved model yet")
        return d

    @staticmethod
    def _demo_paths(demo_ids) -> list:
        root = FLIGHTS_DIR.resolve()
        out = []
        for i in demo_ids or []:
            p = (FLIGHTS_DIR / i).resolve()
            if root not in p.parents or p.suffix != ".npz" or not p.is_file():
                raise ValueError(f"not a demo file: '{i}'")
            out.append(p)
        return out

    def start_run(self, algo: str, seed: Optional[int], start_pos, target_pos,
                  timesteps: int, n_envs: int, eval_freq: int, n_eval_episodes: int,
                  env: str = "net", ring_jitter: float = 1.0, radius_start: float = 1.5,
                  sway_deg: float = 5.0, until_stopped: bool = False,
                  carry_start: float = 0.7, reverse_prob: float = 0.5,
                  lean_flip_prob: float = 0.3, vec: str = "auto",
                  init_run: Optional[str] = None, init_clone: Optional[str] = None,
                  demo_ids=None, human_weight: float = 3.0, bc_coef: float = 1.0,
                  profile: str = "typical", onboard_dr: float = 1.0, onboard_sway_deg: float = 25.0,
                  hard_start_p: float = 0.5, dr_steps: int = 500_000, bc_augment: bool = True,
                  wind_steps: int = 1_500_000,
                  reward_mode: str = "shaped", budget: float = 3.0, laps: int = 1,
                  reward_weights: Optional[dict] = None, onboard_laps: int = 0,
                  onboard_wash_deg: float = 15.0, wash_deg: float = 15.0,
                  start_z: float = 0.8, same_side: bool = True, start_zones: bool = True,
                  streak: float = 0.5, learning_rate: Optional[float] = None,
                  gamma: Optional[float] = None, gae_lambda: Optional[float] = None,
                  ent_coef: Optional[float] = None, target_kl: Optional[float] = None,
                  batch_size: Optional[int] = None, n_epochs: Optional[int] = None,
                  critic_warmup_steps: Optional[int] = None, log_std_reset: Optional[float] = None,
                  sde: bool = False, ring_tracker: bool = False, streak_cap: Optional[float] = None,
                  bc_coef_min: Optional[float] = None, bc_decay_steps: Optional[int] = None) -> str:
        if env == "onboard":
            algo = "ppo"                            # the onboard trainer is PPO (+ demo term)
        if algo not in ALGORITHM_IDS:
            raise ValueError(f"unknown algo '{algo}'")
        if not bounds_ok(start_pos, START_MARGIN) or not bounds_ok(target_pos):
            raise ValueError("start_pos/target_pos outside the room bounds")
        if init_clone and env != "onboard":
            raise ValueError("a behaviour clone can only start an onboard-camera run")
        init_dir = self._init_dir(init_run, init_clone)
        if init_dir is not None and init_run:
            pm = _read_json(init_dir / "meta.json") or {}
            if pm.get("env", "legacy") != env:
                raise ValueError(f"run '{init_run}' trained on env '{pm.get('env', 'legacy')}', "
                                 f"not '{env}'")
            if pm.get("algo") and pm["algo"] != algo:
                raise ValueError(f"run '{init_run}' is a {pm['algo']} run; pick {pm['algo']} "
                                 "to continue it")
        demos = self._demo_paths(demo_ids) if env == "onboard" else []
        if env == "onboard" and reward_weights:
            bad = sorted(set(reward_weights) - set(run_config.REWARD_DEFAULTS))
            if bad:
                raise ValueError(f"unknown reward weight(s) {bad}; names: {sorted(run_config.REWARD_DEFAULTS)}")
        self._check_capacity(n_envs, vec)

        run_id = self._new_run_id("onboard" if env == "onboard" else algo)
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        common = ["--timesteps", str(timesteps), "--n-envs", str(n_envs), "--vec", vec,
                  "--eval-freq", str(eval_freq), "--n-eval-episodes", str(n_eval_episodes),
                  "--run-id", run_id, "--out", str(run_dir)]
        if until_stopped:
            common.append("--until-stopped")
        if seed is not None:
            common += ["--seed", str(seed)]
        if init_dir is not None:
            common += ["--init", str(init_dir)]

        if env == "onboard":
            argv = run_config.onboard_argv(
                PYTHON_EXE, run_id, run_dir, demos=demos, init_dir=init_dir, p=dict(
                    timesteps=timesteps, n_envs=n_envs, vec=vec, eval_freq=eval_freq,
                    n_eval_episodes=n_eval_episodes, until_stopped=until_stopped, seed=seed,
                    profile=profile, onboard_dr=onboard_dr, onboard_sway_deg=onboard_sway_deg,
                    onboard_wash_deg=onboard_wash_deg, same_side=same_side, start_zones=start_zones,
                    streak=streak, streak_cap=streak_cap, hard_start_p=hard_start_p, dr_steps=dr_steps,
                    onboard_laps=onboard_laps, bc_augment=bc_augment, ring_tracker=ring_tracker,
                    human_weight=human_weight, bc_coef=bc_coef, bc_coef_min=bc_coef_min,
                    bc_decay_steps=bc_decay_steps, learning_rate=learning_rate, gamma=gamma,
                    gae_lambda=gae_lambda, ent_coef=ent_coef, target_kl=target_kl,
                    batch_size=batch_size, n_epochs=n_epochs, critic_warmup_steps=critic_warmup_steps,
                    log_std_reset=log_std_reset, sde=sde, reward_weights=reward_weights))
        else:
            argv = [PYTHON_EXE, str(SCRIPTS_DIR / "train_hoop_multi.py"), "--algo", algo,
                    "--env", env, *common]
            if env == "net":
                argv += ["--ring-jitter", str(ring_jitter), "--radius-start", str(radius_start),
                         "--sway-deg", str(sway_deg), "--wash-deg", str(wash_deg),
                         "--start-z", str(start_z),
                         "--lap-dir", "same" if same_side else "alternate",
                         "--start-mode", "zones" if start_zones else "random",
                         "--streak", str(streak),
                         "--carry-start", str(carry_start), "--reverse-prob", str(reverse_prob),
                         "--lean-flip-prob", str(lean_flip_prob),
                         "--wind-steps", str(int(wind_steps)), "--reward", reward_mode,
                         "--budget", str(budget), "--laps", str(int(laps))]
                if reward_weights:
                    argv += ["--reward-weights", json.dumps(reward_weights)]
            if start_pos is not None:
                argv += ["--start-pos", *(str(v) for v in start_pos)]
            if target_pos is not None:
                argv += ["--target-pos", *(str(v) for v in target_pos)]

        log_path = run_dir / "train.log"
        kwargs = _popen_kwargs(log_path)
        proc = subprocess.Popen(argv, **kwargs)
        kwargs["stdout"].close()  # the child has its own duplicated handle now
        self._training_procs[run_id] = proc
        self._launch_meta[run_id] = {"n_envs": n_envs,
                                     "vec": "subproc" if (vec == "subproc" or (vec == "auto" and n_envs > 1))
                                     else "dummy"}
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
                    _terminate_tree(pid)
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
            _terminate_tree(proc.pid)        # its env workers too, not just the trainer
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
            tags = _read_tags(d)
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
                "latest_success_rate": status_doc.get("latest_success_rate"),
                "created_at": meta.get("created_at"),
                "n_envs": meta.get("n_envs"),
                "vec": meta.get("vec"),
                "init_parent": meta.get("init_parent"),
                "label": tags.get("label"),
                "starred": bool(tags.get("starred")),
                "best_success_rate": self._best_success(d),
                "best_mean_passes": status_doc.get("best_mean_passes"),
                "comparable": (meta.get("env") == "net" and meta.get("obs_dim") in NET_OBS_DIMS)
                              or meta.get("env") == "onboard",
            })
        current_run_id = runs[-1]["run_id"] if runs else None
        best_run_ids = {env: _pick_best(runs, env) for env in ("net", "onboard")}
        best_run_id = best_run_ids["net"]
        return {"runs": runs, "current_run_id": current_run_id, "best_run_id": best_run_id,
                "best_run_ids": best_run_ids,
                "capacity": self.capacity()}

    def get_run(self, run_id: str):
        d = RUNS_DIR / run_id
        meta = _read_json(d / "meta.json")
        if meta is None:
            return None
        live_status = _read_json(d / "live_status.json")
        curve = _read_curve(d / "evaluations.npz")
        passes = _read_json(d / "passes.json")
        return {"meta": meta, "live_status": live_status, "curve": curve,
                "passes": passes if isinstance(passes, list) else None,
                "label": _read_tags(d).get("label"),
                "starred": bool(_read_tags(d).get("starred")),
                "status": self._status(run_id, meta, live_status or {})}

    def get_config(self, run_id: str):
        """Everything the Config card shows for a run (None if there is no such run)."""
        d = RUNS_DIR / run_id
        meta = _read_json(d / "meta.json") if d.is_dir() else None
        if meta is None:
            return None
        onboard = meta.get("env") == "onboard"
        status = self._status(run_id, meta, _read_json(d / "live_status.json") or {})
        arch = run_config.architecture(d)
        values = run_config.current_values(meta, arch) if onboard else {}
        fields = [dict(f, value=values.get(f["key"])) for f in run_config.field_schema()] if onboard else []
        rw = run_config.current_reward(meta) if onboard else {}
        reward = [dict(r, value=rw.get(r["key"])) for r in run_config.reward_schema()] if onboard else []
        bc = meta.get("bc") or {}
        info = [(k, meta.get(k)) for k in ("run_id", "algo", "env", "task", "env_variant", "obs_dim", "ctrl_hz",
                                          "yaw_convention", "init_parent", "init_kind", "init_timesteps",
                                          "created_at", "finished_at", "final_status", "final_timesteps",
                                          "vec", "pid", "clearml_task_id")]
        info.append(("status", status))
        if bc:
            info.append(("demo samples", ", ".join(f"{k} {v}" for k, v in (bc.get("samples") or {}).items())))
        layers = {k[len("layers."):]: v for k, v in arch.items() if k.startswith("layers.")}
        architecture = {"obs_dim": arch.get("obs_dim"), "act_dim": arch.get("act_dim"),
                        "activation": arch.get("activation"), "policy_kwargs": arch.get("policy_kwargs"),
                        "layers": layers, "log_std": arch.get("log_std_values"),
                        "saved": {k[4:]: v for k, v in arch.items() if k.startswith("ppo.")},
                        "model_file": arch.get("model_file"), "error": arch.get("architecture_error")}
        out = {"run_id": run_id, "status": status, "env": meta.get("env"), "editable": onboard,
               "fields": fields, "reward": reward, "info": [{"k": k, "v": v} for k, v in info if v is not None],
               "architecture": architecture if arch else None,
               "demo_paths": bc.get("demo_paths"), "demo_names": bc.get("demos"),
               "reward_overrides": meta.get("reward_overrides") or {}}
        if onboard:
            req = dict(values, vec=meta.get("vec") or "auto", reward_weights=rw,
                       onboard_laps=values.get("onboard_laps") or 0)
            out["command"] = self._preview(req)["command"]
        return out

    def _preview(self, req: dict, run_id: str = "<new-run-id>") -> dict:
        from schemas import StartRunRequest
        p = StartRunRequest(algo="ppo", env="onboard").model_dump()      # the dashboard's defaults ...
        p.update({k: v for k, v in req.items() if v is not None or k in run_config.FIELD_BY_KEY})
        for k, f in run_config.FIELD_BY_KEY.items():     # ... except a value the run never recorded
            if p.get(k) is None and f.get("null_ok") is not True and f["type"] != "bool":
                p[k] = StartRunRequest(algo="ppo", env="onboard").model_dump().get(k)
        demos = self._demo_paths(p.get("demo_ids"))
        init_dir = self._init_dir(p.get("init_run"), p.get("init_clone"))
        bad = sorted(set(p.get("reward_weights") or {}) - set(run_config.REWARD_DEFAULTS))
        if bad:
            raise ValueError(f"unknown reward weight(s) {bad}; names: {sorted(run_config.REWARD_DEFAULTS)}")
        argv = run_config.onboard_argv(PYTHON_EXE, run_id, f"results/hoop/runs/{run_id}", p,
                                       init_dir=init_dir, demos=demos)
        return {"command": run_config.command_line(argv), "argv": [str(a) for a in argv]}

    def preview_command(self, req: dict) -> dict:
        """The command line a StartRunRequest would run, without starting anything."""
        if req.get("env") != "onboard":
            raise ValueError("the command preview is for onboard-camera runs")
        return self._preview(req)

    def _best_success(self, run_dir: Path):
        """Highest eval success rate the run reached (evaluations.npz 'successes'), or None.
        Cached by the file's mtime: the run list is polled every 2.5 s."""
        path = run_dir / "evaluations.npz"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        hit = self._success_cache.get(run_dir.name)
        if hit and hit[0] == mtime:
            return hit[1]
        best = None
        try:
            import numpy as np
            with np.load(path) as data:
                if "successes" in data.files and data["successes"].size:
                    best = round(float(np.asarray(data["successes"], dtype=float).mean(axis=1).max()), 4)
        except Exception:
            return hit[1] if hit else None         # mid-write: keep the last good value
        self._success_cache[run_dir.name] = (mtime, best)
        return best

    def _run_dir(self, run_id: str) -> Path:
        d = RUNS_DIR / run_id
        if "/" in run_id or "\\" in run_id or not (d / "meta.json").is_file():
            raise ValueError(f"unknown run '{run_id}'")
        return d

    def set_starred(self, run_id: str, starred: bool) -> bool:
        """Mark a run as a keeper (label.json, next to its name)."""
        d = self._run_dir(run_id)
        tags = _read_tags(d)
        tags["starred"] = bool(starred)
        _write_tags(d, tags)
        return bool(starred)

    def set_label(self, run_id: str, label: str) -> str:
        """A run's display name. Kept in label.json, not meta.json, which the trainer rewrites
        when the run ends; the folder (the run id) is never renamed."""
        d = self._run_dir(run_id)
        label = " ".join(label.split())[:60]
        tags = _read_tags(d)
        tags["label"] = label
        _write_tags(d, tags)
        return label

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
        _refuse_onboard(run_dir)
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
        _refuse_onboard(run_dir)
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


def _terminate_tree(pid: int):
    """Terminate a trainer and every process it started (its SubprocVecEnv workers)."""
    try:
        parent = psutil.Process(pid)
        for c in parent.children(recursive=True):
            try:
                c.terminate()
            except psutil.Error:
                pass
        parent.terminate()
    except psutil.Error:
        pass


def _refuse_onboard(run_dir: Path):
    meta = _read_json(run_dir / "meta.json") or {}
    if meta.get("env") == "onboard":
        raise ValueError("this is an onboard-camera run: replays and showcases are PyBullet-only. "
                         "Score it with hardware/onboard/eval_policy.py --model "
                         f"<run>/best_model.zip, or fly it from the Drone tab")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_tags(run_dir: Path) -> dict:
    """The run's own name and star (label.json; meta.json belongs to the trainer)."""
    doc = _read_json(run_dir / "label.json")
    return doc if isinstance(doc, dict) else {}


def _write_tags(run_dir: Path, tags: dict):
    tags = {k: v for k, v in tags.items() if v}       # drop an empty name / an unset star
    path = run_dir / "label.json"
    if not tags:
        if path.is_file():
            path.unlink()
        return
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(tags))
    os.replace(tmp, path)


def _pick_best(runs, env="net"):
    """The run to highlight for one environment ("net" = PyBullet, "onboard" = camera): among its
    comparable runs, the most passes per evaluation episode, then eval success rate, then eval
    reward. Passes are what the task is about (and what a streak bonus rewards); reward is not
    comparable across reward modes (a budget-mode 2 can beat a shaped-mode 13). For "net", falls
    back to the highest reward when no run has been evaluated on today's env."""
    def num(v):
        return -1e18 if v is None else float(v)
    mine = [r for r in runs if r.get("env", "net") == env]
    cands = [r for r in mine if r["comparable"]
             and (r["best_mean_passes"] is not None or r["best_success_rate"] is not None)]
    if cands:
        # older runs did not log mean passes: with one pass per episode it equals the success rate
        best = max(cands, key=lambda r: (num(r["best_mean_passes"] if r["best_mean_passes"] is not None
                                             else r["best_success_rate"]),
                                         num(r["best_success_rate"]),
                                         num(r["best_eval_reward"])))
        return best["run_id"]
    if env != "net":
        return None
    scored = [r for r in runs if r["best_eval_reward"] is not None and r.get("env") != "onboard"]
    return max(scored, key=lambda r: r["best_eval_reward"])["run_id"] if scored else None


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
