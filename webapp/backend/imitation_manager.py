"""
Imitation learning from the dashboard: list demonstrations and clones, and start the
scripts that make demos, train a clone and evaluate one, as jobs (jobs.py).

Reads only files: the demo .npz's small `meta` entry (numpy, no torch), and each clone's
meta.json and evals/*.json. Training and evaluation run in subprocesses, so the backend
still never imports stable_baselines3 or torch.

Demo ids are paths relative to hardware/flights with forward slashes, e.g.
"tello_20261009-101500/demo.npz" (a recorded flight) or "demos/onboard_sim.npz".
"""
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import FLIGHTS_DIR, HARDWARE_DIR, RUNS_DIR
from jobs import PYTHON_EXE, job_manager, read_json

MAKE_DEMOS = HARDWARE_DIR / "onboard" / "make_demos.py"
MAKE_TEACHER = HARDWARE_DIR / "onboard" / "make_teacher_demos.py"
BC_TRAIN = HARDWARE_DIR / "imitation" / "bc_train.py"
EVAL_POLICY = HARDWARE_DIR / "onboard" / "eval_policy.py"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _check_name(name: str) -> str:
    if not NAME_RE.match(name or ""):
        raise ValueError("names may use letters, digits, - and _ (up to 40)")
    return name


class ImitationManager:
    def __init__(self, flights_dir: Path = FLIGHTS_DIR, jobs=job_manager):
        self.flights = flights_dir
        self.demos_dir = flights_dir / "demos"
        self.bc_dir = flights_dir / "bc"
        self.jobs = jobs

    # ------------------------------------------------------------------ demos

    def _demo_path(self, demo_id: str) -> Path:
        p = (self.flights / demo_id).resolve()
        root = self.flights.resolve()
        if root not in p.parents or p.suffix != ".npz":
            raise ValueError(f"not a demo id: '{demo_id}'")
        return p

    def _demo_info(self, p: Path) -> dict:
        rel = p.relative_to(self.flights).as_posix()
        info = {"id": rel, "file": p.name, "size_kb": round(p.stat().st_size / 1024),
                "real": p.parent.name.startswith("tello_"),
                "label": ("pass" if p.name == "demo.npz" else
                          "miss" if p.name == "demo_failed.npz" else None)}
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                n = int(meta.get("n_samples") or z["episode"].shape[0])
                episodes = int(z["episode"].max()) + 1 if n else 0
        except Exception as e:                     # a corrupt or foreign file
            info.update(error=str(e)[:200])
            return info
        flight_meta = (read_json(p.parent / "meta.json") or {}) if info["real"] else {}
        info.update(samples=n, episodes=episodes, task=meta.get("task"),
                    source=meta.get("source"), saved=meta.get("saved"),
                    dry_run=meta.get("dry_run", flight_meta.get("dry_run")),
                    mode=flight_meta.get("mode"),
                    seconds=round(n / 15.0, 1))
        return info

    def list_demos(self) -> List[dict]:
        paths = []
        if self.flights.is_dir():
            for d in sorted(self.flights.glob("tello_*"), reverse=True):
                paths += [d / n for n in ("demo.npz", "demo_failed.npz") if (d / n).is_file()]
        if self.demos_dir.is_dir():
            paths += sorted(self.demos_dir.glob("*.npz"), reverse=True)
        return [self._demo_info(p) for p in paths]

    def label_demo(self, demo_id: str, passed: bool) -> str:
        """Re-label a recorded flight's demo by renaming it, which is what the training
        glob (tello_*/demo.npz) goes by."""
        p = self._demo_path(demo_id)
        if not p.parent.name.startswith("tello_") or not p.is_file():
            raise ValueError("only a recorded flight's demo can be re-labelled")
        target = p.with_name("demo.npz" if passed else "demo_failed.npz")
        if target == p:
            return demo_id
        if target.exists():
            raise ValueError(f"{target.name} already exists in that flight")
        p.rename(target)
        return target.relative_to(self.flights).as_posix()

    def generate_demos(self, name: str, episodes=60, profile="typical", seed=0,
                       render=False, dr=0.0, sway_deg=5.0, hard_start_p=0.0) -> str:
        out = self.demos_dir / f"{_check_name(name)}.npz"
        if out.exists():
            raise ValueError(f"demos/{out.name} already exists; pick another name")
        if not 1 <= int(episodes) <= 2000:
            raise ValueError("episodes must be 1-2000")
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        argv = [PYTHON_EXE, MAKE_DEMOS, "--episodes", int(episodes), "--profile", profile,
                "--seed", int(seed), "--out", out]
        if render:
            argv.append("--render")
        if dr > 0 or hard_start_p > 0:
            # robust demos = today's training task: prop-wash ring, and the expert flying laps
            # (through, clear, turn round, find it again), so a clone learns the after-pass part
            argv += ["--dr", float(dr), "--sway-deg", float(sway_deg),
                     "--hard-start-p", float(hard_start_p), "--wash-deg", 15.0, "--laps", 0]
        robust = " robust" if dr > 0 or hard_start_p > 0 else ""
        return self.jobs.start("demos", argv, f"Simulated{robust} demos '{name}' ({episodes} episodes)",
                               outputs={"demo": out.relative_to(self.flights).as_posix()})

    def generate_teacher_demos(self, run_id: str, name: str, episodes=150, profile="typical",
                               seed=0) -> str:
        """Fly the onboard simulator with a PyBullet (net-env) run as the teacher; its passed
        episodes become onboard-task demos (hardware/onboard/make_teacher_demos.py)."""
        run_dir = (RUNS_DIR / run_id).resolve()
        if run_dir.parent != RUNS_DIR.resolve() or not (run_dir / "meta.json").is_file():
            raise ValueError(f"unknown run '{run_id}'")
        meta = read_json(run_dir / "meta.json") or {}
        if meta.get("env") != "net":
            raise ValueError("teacher demos need a PyBullet net-env run "
                             f"('{run_id}' is env '{meta.get('env', 'legacy')}')")
        if meta.get("algo") == "dqn":
            raise ValueError("a DQN run cannot be a teacher (discrete moves); pick PPO/A2C/SAC/TD3")
        out = self.demos_dir / f"{_check_name(name)}.npz"
        if out.exists():
            raise ValueError(f"demos/{out.name} already exists; pick another name")
        if not 1 <= int(episodes) <= 2000:
            raise ValueError("episodes must be 1-2000")
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        argv = [PYTHON_EXE, MAKE_TEACHER, "--run", run_dir, "--episodes", int(episodes),
                "--profile", profile, "--seed", int(seed), "--out", out]
        return self.jobs.start("teacher", argv,
                               f"Teacher demos '{name}' from {run_id} ({episodes} episodes)",
                               outputs={"demo": out.relative_to(self.flights).as_posix()})

    # ------------------------------------------------------------------ clones

    def _model_dir(self, name: str) -> Path:
        return self.bc_dir / _check_name(name)

    @staticmethod
    def _evals(d: Path) -> list:
        evals = []
        if (d / "evals").is_dir():
            for e in sorted((d / "evals").glob("*.json"), reverse=True):
                r = read_json(e)
                if r:
                    evals.append({k: r.get(k) for k in (
                        "passed", "struck", "missed", "timeout", "episodes",
                        "render", "profile", "saved")})
        return evals

    def list_models(self) -> List[dict]:
        """Behaviour clones, then onboard-task RL runs (name "rl:<run_id>", their best model):
        both fly the same task, so the Drone and Imitation tabs treat them alike."""
        out = []
        if self.bc_dir.is_dir():
            for d in sorted(self.bc_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                zp = d / "bc_model.zip"
                if not d.is_dir() or not zp.is_file():
                    continue
                meta = read_json(d / "meta.json") or {}
                out.append({
                    "name": d.name, "kind": "clone", "model": str(zp),
                    "task": meta.get("task"), "demos": meta.get("demos"),
                    "samples": meta.get("n_samples"), "saved": meta.get("saved"),
                    "val_mse": meta.get("val_dir_loss"), "val_cos": meta.get("val_mean_cos"),
                    "train_eval": meta.get("sim_eval"), "evals": self._evals(d),
                    "flyable": meta.get("task") in (None, "onboard_hoop"),
                })
        if RUNS_DIR.is_dir():
            for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                meta = read_json(d / "meta.json") if d.is_dir() else None
                zp = d / "best_model.zip"
                if not meta or meta.get("env") != "onboard" or not zp.is_file():
                    continue
                status = read_json(d / "live_status.json") or {}
                bc = meta.get("bc") or {}
                out.append({
                    "name": f"rl:{d.name}", "kind": "rl", "model": str(zp),
                    "task": meta.get("task", "onboard_hoop"), "demos": bc.get("demos"),
                    "samples": None, "saved": meta.get("finished_at") or status.get("updated_at"),
                    "val_mse": None, "val_cos": None,
                    "train_eval": ({"pass_rate": status.get("latest_success_rate")}
                                   if status.get("latest_success_rate") is not None else None),
                    "evals": self._evals(d), "flyable": True,
                })
        return out

    def train(self, name: str, demo_ids: List[str], epochs=200, lr=3e-3, use="intent",
              eval_episodes=20, seed=0) -> str:
        out = self._model_dir(name)
        if out.exists():
            raise ValueError(f"a clone called '{name}' already exists; pick another name")
        if not demo_ids:
            raise ValueError("pick at least one demonstration")
        paths = [self._demo_path(i) for i in demo_ids]
        missing = [i for i, p in zip(demo_ids, paths) if not p.is_file()]
        if missing:
            raise ValueError(f"missing demo files: {missing}")
        if use not in ("intent", "executed"):
            raise ValueError("use must be 'intent' or 'executed'")
        if not 1 <= int(epochs) <= 5000:
            raise ValueError("epochs must be 1-5000")
        argv = [PYTHON_EXE, BC_TRAIN, "--demos", *paths, "--out", out,
                "--epochs", int(epochs), "--lr", float(lr), "--use", use,
                "--eval-episodes", int(eval_episodes), "--seed", int(seed)]
        return self.jobs.start("train", argv, f"Train clone '{name}' on {len(paths)} demo file(s)",
                               outputs={"model": name})

    def evaluate(self, name: str, episodes=12, render=False, seed=100) -> str:
        if name.startswith("rl:"):                  # an onboard RL run's best model
            run_id = _check_name(name[3:])
            zp = RUNS_DIR / run_id / "best_model.zip"
            meta = read_json(RUNS_DIR / run_id / "meta.json") or {}
            if meta.get("env") != "onboard" or not zp.is_file():
                raise ValueError(f"no onboard run '{run_id}' with a best model")
        else:
            zp = self._model_dir(name) / "bc_model.zip"
        if not zp.is_file():
            raise ValueError(f"no clone called '{name}'")
        if not 1 <= int(episodes) <= 200:
            raise ValueError("episodes must be 1-200")
        argv = [PYTHON_EXE, EVAL_POLICY, "--model", zp, "--episodes", int(episodes),
                "--seed", int(seed)]
        if render:
            argv.append("--render")
        return self.jobs.start("eval", argv,
                               f"Evaluate '{name}' ({episodes} episodes"
                               f"{', real detector' if render else ''})",
                               outputs={"model": name})


imitation_manager = ImitationManager()
