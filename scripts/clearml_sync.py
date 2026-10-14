"""
ClearML sync for every run this project makes: PyBullet training, onboard training, behaviour
clones and real / dry-run Tello flights. One helper, so a task looks the same whether it was
logged live or backfilled later by clearml_upload_runs.py.

    run = clearml_sync.start("onboard-rl", run_dir, meta)       # at run start (live tasks)
    callback_list.append(run.callback(eval_cb))                  # SB3: eval + loss scalars
    run.finish("stopped")                                        # in the trainer's finally

Projects in ClearML: Tello/onboard-rl, Tello/pybullet, Tello/clones, Tello/flights.

Never breaks a run. It is a no-op, with one printed line saying why, when:
  * clearml is not installed, or no credentials are configured (run `clearml-init` once);
  * CLEARML_OFF=1 is set, or the trainer was started with --no-clearml;
  * it runs under pytest (the test suites call the trainers' main()).
Any ClearML error after that is printed and swallowed: training and flying carry on.

Flights are uploaded only AFTER landing (from the files fly_hoop.py already wrote), so no
network traffic or background thread ever competes with the 15 Hz flight loop.
"""
import json
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = "Tello"
ARTIFACT_FILES = ("best_model.zip", "latest_model.zip", "final_model.zip", "bc_model.zip",
                  "meta.json", "evaluations.npz", "passes.json", "live_status.json",
                  "train.log", "train.err.log", "camera_latest.jpg")
FLIGHT_FILES = ("meta.json", "summary.json", "events.jsonl", "ticks.jsonl", "telemetry.csv",
                "demo.npz", "demo_failed.npz")
TICK_SERIES = ("range_m", "cx", "cy", "axis_ratio", "scale", "age", "travel_m", "h", "bat")
# A flight's tick log is 15 Hz; every 3rd tick (5 Hz) is plenty for plots and is a third of the
# report_scalar calls, which were most of a flight upload's time.
FLIGHT_TICK_EVERY = 3
STATUS_FINISH = {"completed": "completed", "stopped": "stopped", "crashed": "failed",
                 "failed": "failed", "landed": "completed", "landed_on_request": "stopped",
                 "refused": "failed", "interrupted": "stopped"}

_warned = {"done": False}


def _say(msg):
    print(f"[clearml] {msg}")


def disabled_reason():
    """Why syncing is off, or None when it is on."""
    if os.environ.get("CLEARML_OFF", "").strip() not in ("", "0"):
        return "CLEARML_OFF is set"
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return "running under pytest"
    try:
        import clearml  # noqa: F401
    except ImportError:
        return "clearml is not installed (pip install clearml)"
    has_env = os.environ.get("CLEARML_API_ACCESS_KEY") and os.environ.get("CLEARML_API_SECRET_KEY")
    cfg = os.environ.get("CLEARML_CONFIG_FILE") or str(Path.home() / "clearml.conf")
    if not has_env and not Path(cfg).is_file():
        return "no ClearML credentials (run `clearml-init` once)"
    return None


def enabled(no_clearml_flag=False) -> bool:
    reason = "--no-clearml" if no_clearml_flag else disabled_reason()
    if reason and not _warned["done"]:
        _say(f"sync off: {reason}")
        _warned["done"] = True
    return reason is None


# --- what goes into a task ----------------------------------------------------------------

def flatten(d, prefix="", out=None):
    """{'a': {'b': 1}} -> {'a.b': 1}; lists become JSON strings so ClearML shows them whole."""
    out = {} if out is None else out
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flatten(v, key + ".", out)
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def model_architecture(zip_path) -> dict:
    """Read a saved SB3 model zip WITHOUT building an env: algorithm hyperparameters from its
    `data` JSON, and the real layer shapes from policy.pth. Activation is SB3's default for
    MlpPolicy (Tanh) unless policy_kwargs says otherwise."""
    out = {}
    try:
        with zipfile.ZipFile(zip_path) as z:
            data = json.loads(z.read("data"))
            for k in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef",
                      "vf_coef", "max_grad_norm", "num_timesteps", "n_envs"):
                if k in data and not isinstance(data[k], dict):
                    out[f"ppo.{k}"] = data[k]
            for k in ("clip_range", "learning_rate"):
                v = data.get(k)
                out[f"ppo.{k}"] = v if not isinstance(v, dict) else "schedule (see meta)"
            out["obs_dim"] = (data.get("observation_space") or {}).get("_shape")
            out["act_dim"] = (data.get("action_space") or {}).get("_shape")
            pk = {k: v for k, v in (data.get("policy_kwargs") or {}).items() if k != ":type:"}
            out["policy_kwargs"] = json.dumps(pk, default=str) if pk else "{} (SB3 defaults)"
            out["activation"] = str(pk.get("activation_fn", "Tanh (SB3 MlpPolicy default)"))
            if "policy.pth" in z.namelist():
                import io
                import torch
                sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
                for name, t in sd.items():
                    if name.endswith(".weight") or name == "log_std":
                        out[f"layers.{name}"] = "x".join(str(s) for s in t.shape)
                if "log_std" in sd:
                    out["log_std_values"] = json.dumps([round(float(x), 3) for x in sd["log_std"]])
    except Exception as e:      # an unreadable zip must not stop a sync
        out["architecture_error"] = str(e)
    return out


def _tags(kind, meta):
    tags = [kind]
    for key in ("algo", "env", "task", "mode", "profile"):
        if meta.get(key):
            tags.append(f"{key}:{meta[key]}")
    if meta.get("obs_dim"):
        tags.append(f"obs:{meta['obs_dim']}")
    if meta.get("init_parent"):
        tags.append(f"parent:{meta['init_parent']}")
    if meta.get("dry_run") is not None:
        tags.append("dry_run" if meta["dry_run"] else "real_flight")
    if meta.get("model"):
        tags.append(f"model:{Path(str(meta['model'])).parent.name}")
    robust = meta.get("robust") or {}
    if robust.get("dr"):
        tags.append(f"dr:{robust['dr']}")
    if meta.get("laps") is not None:
        tags.append(f"laps:{meta['laps']}")
    return tags


def find_task_id(project, name):
    """The id of an existing task with this name in this project, or None."""
    from clearml import Task
    try:
        ids = Task.query_tasks(project_name=project, task_name=f"^{name}$")
        return ids[0] if ids else None
    except Exception:
        return None


# --- a run being synced -------------------------------------------------------------------

class ClearMLRun:
    """Wraps one ClearML task. Every method swallows errors (printing them once each)."""

    def __init__(self, task, kind, run_dir, meta, live):
        self.task, self.kind, self.run_dir, self.meta, self.live = task, kind, Path(run_dir), meta, live
        self.logger = task.get_logger() if task else None

    @property
    def active(self):
        return self.task is not None

    def _safe(self, what, fn, *a, **kw):
        if not self.active:
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:
            _say(f"{what} failed (run continues): {e}")
            return None

    def set_meta(self, meta):
        """Hyperparameters tab: meta.json flattened into sections, plus the whole file as a
        configuration object."""
        self.meta = meta

        def _do():
            flat = flatten(meta)
            params = {}
            for k, v in flat.items():
                section = k.split(".", 1)[0] if "." in k else "run"
                params[f"{section}/{k}"] = "" if v is None else v
            self.task.set_parameters(params)
            self.task.connect_configuration(meta, name="meta.json")
        self._safe("set_meta", _do)

    def set_architecture(self, zip_path):
        def _do():
            arch = model_architecture(zip_path)
            self.task.set_parameters({f"architecture/{k}": "" if v is None else v for k, v in arch.items()})
        if zip_path and Path(zip_path).is_file():
            self._safe("architecture", _do)

    def scalar(self, title, series, value, step):
        if value is None:
            return
        self._safe("scalar", self.logger.report_scalar, title, series, float(value), int(step))

    def upload(self, path, name=None, wait=True):
        path = Path(path)
        if path.is_file():
            self._safe(f"upload {path.name}", self.task.upload_artifact, name or path.name,
                       artifact_object=str(path), wait_on_upload=wait)

    def callback(self, eval_cb=None):
        """An SB3 callback for live training: hooks the run's SB3 logger (every value it dumps:
        train/*, rollout/*, eval/*, bc/*, time/*) and reports passes per evaluation."""
        return _SB3Callback(self, eval_cb)

    def report_files(self):
        """Scalars from the files a run leaves behind (used by the backfill, and at finish for
        anything the live callback could not see)."""
        d = self.run_dir
        ev = d / "evaluations.npz"
        if ev.is_file():
            def _ev():
                import numpy as np
                e = np.load(ev)
                steps = e["timesteps"]
                res = e["results"]
                lens = e["ep_lengths"] if "ep_lengths" in e.files else None
                succ = e["successes"] if "successes" in e.files else None
                for i, t in enumerate(steps):
                    self.logger.report_scalar("eval", "mean_reward", float(res[i].mean()), int(t))
                    self.logger.report_scalar("eval", "std_reward", float(res[i].std()), int(t))
                    if lens is not None:
                        self.logger.report_scalar("eval", "mean_ep_length", float(lens[i].mean()), int(t))
                    if succ is not None and np.size(succ[i]):
                        self.logger.report_scalar("eval", "success_rate", float(np.mean(succ[i])), int(t))
            self._safe("evaluations.npz", _ev)
        pj = d / "passes.json"
        if pj.is_file():
            def _pj():
                for row in json.loads(pj.read_text()):
                    t = int(row["t"])
                    self.logger.report_scalar("passes", "mean_per_episode", float(row["mean"]), t)
                    self.logger.report_scalar("passes", "max_in_one_episode", float(row["max"]), t)
                    for z, v in (row.get("zones") or {}).items():
                        self.logger.report_scalar("passes_by_zone", str(z), float(v), t)
            self._safe("passes.json", _pj)

    def report_flight(self):
        """A flight's tick log as scalar series against time (seconds), plus its outcome."""
        d = self.run_dir
        ticks = d / "ticks.jsonl"
        if ticks.is_file():
            def _ticks():
                for line in ticks.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "t" not in r:
                        continue
                    it = int(round(float(r["t"]) * 15))          # tick index at 15 Hz
                    if it % FLIGHT_TICK_EVERY:
                        continue
                    for k in TICK_SERIES:
                        v = r.get(k)
                        if isinstance(v, (int, float)) and v is not None:
                            self.logger.report_scalar("ring" if k in ("range_m", "cx", "cy", "axis_ratio",
                                                                      "scale", "age") else "drone",
                                                      k, float(v), it)
                    self.logger.report_scalar("ring", "found", 1.0 if r.get("found") else 0.0, it)
                    for i, ch in enumerate(("lr", "fb", "ud", "yaw")):
                        act, rc = r.get("action"), r.get("rc")
                        if isinstance(act, list) and len(act) > i:
                            self.logger.report_scalar("policy_action", ch, float(act[i]), it)
                        if isinstance(rc, list) and len(rc) > i:
                            self.logger.report_scalar("rc_sent", ch, float(rc[i]), it)
            self._safe("ticks.jsonl", _ticks)
        sj = d / "summary.json"
        if sj.is_file():
            self._safe("summary", lambda: self.task.connect_configuration(
                json.loads(sj.read_text(encoding="utf-8")), name="summary.json"))

    def finish(self, status=None, extra_files=()):
        """Upload the run's files and mark the task. Safe to call twice."""
        if not self.active:
            return
        meta_path = self.run_dir / "meta.json"
        if meta_path.is_file():
            try:
                self.set_meta(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        for name in (FLIGHT_FILES if self.kind == "flights" else ARTIFACT_FILES):
            self.upload(self.run_dir / name)
        for f in extra_files:
            self.upload(f)
        model = next((self.run_dir / n for n in ("best_model.zip", "bc_model.zip", "final_model.zip")
                      if (self.run_dir / n).is_file()), None)
        self.set_architecture(model)
        if self.kind == "flights":
            self.report_flight()
        elif not self.live:
            self.report_files()
        status = status or self.meta.get("final_status") or "completed"
        mark = STATUS_FINISH.get(status, "completed")

        def _mark():
            self.task.add_tags([f"status:{status}"])
            self.logger.flush()
            if mark == "failed":
                self.task.mark_failed(status_reason=status, force=True)
            elif mark == "stopped":
                self.task.mark_stopped(force=True, status_message=status)
            else:
                self.task.mark_completed(force=True)
        self._safe("finish", _mark)
        if self.live:
            self._safe("close", self.task.close)
        self.task = None


def _write_task_id(run_dir, task_id):
    """Remember the task in meta.json so the backfill never uploads this run twice."""
    p = Path(run_dir) / "meta.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        m["clearml_task_id"] = task_id
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, indent=2), encoding="utf-8")
        os.replace(tmp, p)
    except (OSError, json.JSONDecodeError):
        pass


def _parent_id(project, meta):
    parent = meta.get("init_parent")
    return find_task_id(project, parent) if parent else None


def start(kind, run_dir, meta, no_clearml=False) -> ClearMLRun:
    """Live sync for a run starting now (training scripts). Returns an inactive ClearMLRun when
    sync is off, so callers never need an `if`."""
    run_dir = Path(run_dir)
    if not enabled(no_clearml):
        return ClearMLRun(None, kind, run_dir, meta, live=True)
    project = f"{PROJECT_ROOT}/{kind}"
    name = meta.get("run_id") or meta.get("flight_id") or run_dir.name
    try:
        from clearml import Task
        task = Task.init(project_name=project, task_name=name, tags=_tags(kind, meta),
                         reuse_last_task_id=False, continue_last_task=False,
                         auto_connect_frameworks={"pytorch": False, "tensorboard": False,
                                                  "matplotlib": False},
                         auto_resource_monitoring=True)
        run = ClearMLRun(task, kind, run_dir, meta, live=True)
        run.set_meta(meta)
        pid = _parent_id(project, meta)
        if pid:
            run._safe("parent", task.set_parent, pid)
        meta["clearml_task_id"] = task.id        # the trainer rewrites meta.json from this dict
        _write_task_id(run_dir, task.id)
        _say(f"syncing to {project}/{name}: {task.get_output_log_web_page()}")
        return run
    except Exception as e:
        _say(f"could not start a task (run continues without ClearML): {e}")
        return ClearMLRun(None, kind, run_dir, meta, live=True)


def record(kind, run_dir, meta=None, status=None, skip_existing=True) -> str:
    """Upload a FINISHED run or flight as one task (flights after landing, and the backfill).
    Returns 'uploaded', 'skipped' or 'off'."""
    run_dir = Path(run_dir)
    if not enabled():
        return "off"
    if meta is None:
        try:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    project = f"{PROJECT_ROOT}/{kind}"
    name = meta.get("run_id") or meta.get("flight_id") or run_dir.name
    if skip_existing and (meta.get("clearml_task_id") or find_task_id(project, name)):
        return "skipped"
    try:
        from clearml import Task
        task = Task.create(project_name=project, task_name=name,
                           task_type=Task.TaskTypes.training if kind != "flights" else Task.TaskTypes.inference)
        task.add_tags(_tags(kind, meta))
        task.mark_started(force=True)
        run = ClearMLRun(task, kind, run_dir, meta, live=False)
        run.set_meta(meta)
        pid = _parent_id(project, meta)
        if pid:
            run._safe("parent", task.set_parent, pid)
        if kind == "flights" and meta.get("model"):
            for proj in ("onboard-rl", "clones"):
                mid = find_task_id(f"{PROJECT_ROOT}/{proj}", Path(str(meta["model"])).parent.name)
                if mid:
                    run._safe("model link", task.set_parameters, {"run/model_task_id": mid})
                    break
        tid = task.id
        run.finish(status)
        _write_task_id(run_dir, tid)
        return "uploaded"
    except Exception as e:
        _say(f"upload of {name} failed: {e}")
        return "failed"


# --- SB3 hook -----------------------------------------------------------------------------

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter
except ImportError:          # flights and the backfill do not need SB3
    BaseCallback = object
    KVWriter = object


class _ClearMLWriter(KVWriter):
    """An extra output for SB3's logger: whatever SB3 prints in its table goes to ClearML too."""

    def __init__(self, run):
        self.run = run

    def write(self, key_values, key_excluded, step=0):
        for key, value in key_values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                title, _, series = key.partition("/")
                self.run.scalar(title or "train", series or key, value, step)

    def close(self):
        pass


class _SB3Callback(BaseCallback):
    def __init__(self, run, eval_cb=None):
        super().__init__()
        self.run, self.eval_cb = run, eval_cb
        self._seen_eval = None

    def _on_training_start(self):
        if not self.run.active:
            return
        fmts = self.model.logger.output_formats
        if not any(isinstance(f, _ClearMLWriter) for f in fmts):
            fmts.append(_ClearMLWriter(self.run))

    def _on_step(self) -> bool:
        cb = self.eval_cb
        if self.run.active and cb is not None and getattr(cb, "latest_mean_passes", None) is not None:
            key = (len(getattr(cb, "_history", [])), cb.latest_mean_passes)
            if key != self._seen_eval:
                self._seen_eval = key
                t = self.num_timesteps
                self.run.scalar("passes", "mean_per_episode", cb.latest_mean_passes, t)
                self.run.scalar("passes", "max_in_one_episode", cb.max_passes, t)
                for z, v in (cb.zone_passes or {}).items():
                    self.run.scalar("passes_by_zone", str(z), v, t)
        return True


def main(argv=None):
    """python scripts/clearml_sync.py record flights <dir> [--status S]: upload one finished run or
    flight (fly_hoop.py starts this as a detached process after landing, so the flight process
    can exit at once instead of waiting ~35 s for the upload)."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("kind")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--status", default=None)
    a = ap.parse_args(argv)
    marker = a.run_dir / "clearml_uploading.flag"
    try:
        marker.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass
    try:
        out = record(a.kind, a.run_dir, status=a.status)
        _say(f"{a.run_dir.name}: {out}")
    finally:
        try:
            marker.unlink()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
