"""
Three Stable-Baselines3 callbacks, combined via CallbackList in
train_hoop_multi.py, that let the dashboard watch a training run without the
backend ever importing stable_baselines3/pybullet/torch itself. Each one
writes small files under the run directory that the backend tails or reads.

TelemetryCallback   position/orientation/target, for the live 3D view.
DashboardEvalCallback  current/best eval reward, for the live chart + status pill.
CameraCallback      onboard camera frame, only while explicitly toggled on.

Observation layout this relies on (same for every algorithm and for both the
continuous VEL env and the HoopFindDiscreteVel-wrapped one, since the wrapper
only changes the action space, not the observation): obs[0:3] is the drone's
position, obs[3:6] is roll/pitch/yaw, and the LAST 3 entries are the vector
from the drone to the hoop (HoopFindAviary._computeObs appends that after the
parent's kinematic + action-history observation), so
target_pos = obs[0:3] + obs[-3:]. Verified against scripts/hoop_env.py.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_capture import get_onboard_frame  # noqa: E402


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class TelemetryCallback(BaseCallback):
    """Throttled drone position/orientation/target stream for the live 3D view."""

    def __init__(self, run_dir: Path, env_index: int = 0, hz: float = 12.0,
                 max_lines: int = 1800, rotate_every: int = 150, verbose=0):
        super().__init__(verbose)
        self.telemetry_path = Path(run_dir) / "telemetry.jsonl"
        self.env_index = env_index
        self.min_interval = 1.0 / hz
        self.max_lines = max_lines
        self.rotate_every = rotate_every
        self._last_write = 0.0
        self._append_count = 0

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._last_write < self.min_interval:
            return True
        self._last_write = now
        new_obs = self.locals.get("new_obs")
        if new_obs is None or len(new_obs) <= self.env_index:
            return True
        obs_i = np.asarray(new_obs[self.env_index]).reshape(-1)
        pos = obs_i[0:3]
        rpy = obs_i[3:6]
        rel = obs_i[-3:]
        record = {
            "t": int(self.num_timesteps),
            "ts": now,
            "pos": pos.tolist(),
            "rpy": rpy.tolist(),
            "target": (pos + rel).tolist(),
        }
        # Best-effort, like CameraCallback: OneDrive sync (the run dir lives
        # under a OneDrive-synced path) or the dashboard backend polling the
        # same file can hold a transient lock, which must never take down
        # training over what's just a dropped dashboard frame.
        try:
            with open(self.telemetry_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._append_count += 1
        except OSError as e:
            if self.verbose:
                print(f"[TelemetryCallback] write failed: {e}")
            return True
        if self._append_count % self.rotate_every == 0:
            self._rotate()
        return True

    def _rotate(self):
        try:
            lines = self.telemetry_path.read_text().splitlines()
        except FileNotFoundError:
            return
        except OSError as e:
            if self.verbose:
                print(f"[TelemetryCallback] rotate read failed: {e}")
            return
        if len(lines) > self.max_lines:
            try:
                _atomic_write_text(self.telemetry_path, "\n".join(lines[-self.max_lines:]) + "\n")
            except OSError as e:
                if self.verbose:
                    print(f"[TelemetryCallback] rotate write failed: {e}")


class DashboardEvalCallback(EvalCallback):
    """EvalCallback that also writes a live_status.json snapshot on every eval.

    Subclasses EvalCallback (rather than a second independent callback)
    because it already exposes self.best_mean_reward / self.last_mean_reward
    as public attributes, updated every eval_freq calls. This is the one
    place the backend reads "is training still going, what's the current/best
    number", so it never has to parse stdout or race-read evaluations.npz
    mid-write.
    """

    def __init__(self, *args, run_dir: Path, total_timesteps: int, t0: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_status_path = Path(run_dir) / "live_status.json"
        self.total_timesteps = total_timesteps
        self.t0 = t0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            status = {
                "status": "training",
                "current_timestep": int(self.num_timesteps),
                "total_timesteps": self.total_timesteps,
                "latest_eval_reward": (None if self.last_mean_reward == -np.inf
                                        else float(self.last_mean_reward)),
                "best_eval_reward": (None if self.best_mean_reward == -np.inf
                                      else float(self.best_mean_reward)),
                "elapsed_sec": time.time() - self.t0,
                "updated_at": datetime.now().isoformat(),
            }
            try:
                _atomic_write_text(self.live_status_path, json.dumps(status))
            except OSError as e:
                if self.verbose:
                    print(f"[DashboardEvalCallback] status write failed: {e}")
        return continue_training


class CameraCallback(BaseCallback):
    """Writes the drone's onboard camera view to camera_latest.jpg, but only
    while camera_enabled.flag exists under the run directory. Does nothing at
    all (no render call, no file write) when the flag is absent, which is the
    default, so training pays zero extra cost unless the camera was actually
    turned on. Checked (and possibly toggled) live via
    POST /api/runs/{run_id}/camera, no restart of training needed.
    """

    def __init__(self, run_dir: Path, env_index: int = 0, hz: float = 2.5, verbose=0):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.flag_path = self.run_dir / "camera_enabled.flag"
        self.frame_path = self.run_dir / "camera_latest.jpg"
        self.env_index = env_index
        self.min_interval = 1.0 / hz
        self._last_write = 0.0

    def _on_step(self) -> bool:
        if not self.flag_path.exists():
            return True
        now = time.time()
        if now - self._last_write < self.min_interval:
            return True
        self._last_write = now
        # Capture, encode, and write are all best-effort: a camera failure
        # must never take down the training run, it should just mean no
        # fresh frame this tick.
        try:
            # self.training_env is a DummyVecEnv; get_attr("unwrapped", ...)
            # walks each sub-env's wrapper stack (Monitor, and for DQN also
            # HoopFindDiscreteVel) down to the raw HoopFindAviary instance,
            # which is what _getDroneImages lives on.
            env = self.training_env.get_attr("unwrapped", indices=[self.env_index])[0]
            frame = get_onboard_frame(env, nth_drone=0)
            from PIL import Image
            import io
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=70)
            _atomic_write_bytes(self.frame_path, buf.getvalue())
        except Exception as e:
            if self.verbose:
                print(f"[CameraCallback] frame capture failed: {e}")
        return True
