"""
Three Stable-Baselines3 callbacks, combined via CallbackList in
train_hoop_multi.py, that let the dashboard watch a training run without the
backend ever importing stable_baselines3/pybullet/torch itself. Each one
writes small files under the run directory that the backend tails or reads.

TelemetryCallback   position/orientation/target, for the live 3D view.
DashboardEvalCallback  current/best eval reward, for the live chart + status pill.
CameraCallback      onboard camera frame, only while explicitly toggled on.
PaceCallback        slows training to real time while realtime.flag exists (watching).

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

    def __init__(self, run_dir: Path, env_index: int = 0, hz: float = 15.0,
                 max_lines: int = 1800, rotate_every: int = 150, state_method: str = None,
                 ctrl_hz: float = None, verbose=0):
        """state_method: for envs whose observation is not HoopNetAviary's layout (the onboard
        camera task), the name of an env method returning {pos, rpy, target, hoop_normal,
        hoop_radius}; it is called through env_method, so it works in subprocess envs.
        ctrl_hz: the env's control rate, sent along so the dashboard's clock is right."""
        super().__init__(verbose)
        self.state_method = state_method
        self.ctrl_hz = ctrl_hz
        self.telemetry_path = Path(run_dir) / "telemetry.jsonl"
        self.env_index = env_index
        self.min_interval = 1.0 / hz
        self.max_lines = max_lines
        self.rotate_every = rotate_every
        self._last_write = 0.0
        self._append_count = 0
        self._ep = 0              # episode counter for the watched env
        self._ep_step = 0
        self._net_hit = self._rim_hit = False     # contact since the last written record

    def _on_step(self) -> bool:
        now = time.time()
        infos = self.locals.get("infos") or [{}]
        dones = self.locals.get("dones")
        info = infos[self.env_index] if len(infos) > self.env_index else {}
        done = bool(dones[self.env_index]) if dones is not None and len(dones) > self.env_index else False
        self._ep_step += 1
        self._net_hit |= bool(info.get("net_contact"))
        self._rim_hit |= bool(info.get("rim_contact"))
        # The last step of an episode is always written (so the dashboard sees passes / ends
        # even when the 12 Hz throttle would have skipped it); other steps are throttled.
        if not done and now - self._last_write < self.min_interval:
            return True
        self._last_write = now
        new_obs = self.locals.get("new_obs")
        if new_obs is None or len(new_obs) <= self.env_index:
            return True
        obs_i = np.asarray(new_obs[self.env_index]).reshape(-1)
        if done and "terminal_observation" in info:
            # SB3 auto-resets: new_obs is already the next episode's first obs
            obs_i = np.asarray(info["terminal_observation"]).reshape(-1)
        state = None
        if self.state_method:
            # (at an episode end the env has already reset: this is the new episode's start,
            # which only shows for one frame; the done/success flags below are the old episode's)
            try:
                state = self.training_env.env_method(self.state_method, indices=[self.env_index])[0]
            except Exception:
                return True
            if not state:
                return True
            pos, rpy, target = state["pos"], state["rpy"], state["target"]
        else:
            pos = obs_i[0:3]
            rpy = obs_i[3:6]
            target = (pos + obs_i[-3:]).tolist()
            pos, rpy = pos.tolist(), rpy.tolist()
        record = {
            "t": int(self.num_timesteps),
            "ts": now,
            "pos": pos,
            "rpy": rpy,
            "target": target,
            "ep": self._ep,
            "step": self._ep_step,
            "net_contact": self._net_hit,
            "rim_contact": self._rim_hit,
        }
        self._net_hit = self._rim_hit = False
        if "passes" in info:
            record["passes"] = int(info["passes"])
        if "score" in info:
            record["score"] = round(float(info["score"]), 4)
        if "reversed" in info:
            record["reversed"] = bool(info["reversed"])
        zone = _start_zone(info)
        if zone:
            record["start_zone"] = zone
        # this episode's ring swing (HoopNetAviary): the dashboard HUD shows it, since the swing is
        # wind ramp x per-episode gust x the "Max swing" setting, often well under the setting
        for k in ("wind_scale", "swing_deg", "swing_period_s"):
            if k in info:
                record[k] = round(float(info[k]), 3)
        if done:
            record["done"] = True
            record["success"] = bool(info.get("is_success", False))
            self._ep += 1
            self._ep_step = 0
        if self.ctrl_hz:
            record["ctrl_hz"] = self.ctrl_hz
        if state is not None:
            record["hoop_normal"] = state.get("hoop_normal")
            record["hoop_radius"] = state.get("hoop_radius")
        elif obs_i.size == 84:     # + streak_obs: [..., radius, armed, streak, relative vector]
            record["hoop_normal"] = obs_i[-12:-9].tolist()
            record["hoop_radius"] = float(obs_i[-6])
        elif obs_i.size == 83:     # HoopNetAviary(armed_obs=True): [..., radius, armed, relative vector]
            record["hoop_normal"] = obs_i[-11:-8].tolist()
            record["hoop_radius"] = float(obs_i[-5])
        elif obs_i.size == 82:     # HoopNetAviary: [..., normal, ring vel, ring radius, relative vector]
            record["hoop_normal"] = obs_i[-10:-7].tolist()
            record["hoop_radius"] = float(obs_i[-4])
        elif obs_i.size == 78:   # older HoopNetAviary: ring normal just before the relative vector
            record["hoop_normal"] = obs_i[-6:-3].tolist()
            try:
                record["hoop_radius"] = float(
                    self.training_env.get_attr("inner_radius", indices=[self.env_index])[0])
            except Exception:
                pass
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

    best_model.zip follows PASSES (2026-10-11 night): the highest mean passes per evaluation
    episode, ties broken by mean reward. Reward is not comparable across reward changes and a
    streak bonus makes it noisy; passes are what the task is about. Envs that report no passes
    (the legacy point-reaching env) fall back to reward, as before. Every evaluation is also
    appended to passes.json (mean / max passes, and per start zone when the env reports one).
    """

    def __init__(self, *args, run_dir: Path, total_timesteps: int, t0: float, **kwargs):
        # this class saves best_model.zip itself (by passes); SB3's own save is by reward
        self._best_path = kwargs.pop("best_model_save_path", None)
        kwargs["best_model_save_path"] = None
        super().__init__(*args, **kwargs)
        self.run_dir = Path(run_dir)
        self.live_status_path = Path(run_dir) / "live_status.json"
        self.total_timesteps = total_timesteps
        self.t0 = t0
        self._passes_buffer = []          # passes per episode in the current evaluation
        self.latest_mean_passes = None
        self.best_mean_passes = None
        self.max_passes = None            # most passes in any one evaluation episode so far
        self._zone_buffer = []            # (start zone, passes) per episode in the current evaluation
        self.zone_passes = None           # {zone: mean passes} of the latest evaluation
        self._best_key = None             # (mean passes, mean reward) of best_model.zip
        self._history = []                # passes.json

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            info = locals_["info"]
            passes = info.get("passes")
            if passes is not None:
                self._passes_buffer.append(int(passes))
                zone = _start_zone(info)
                if zone:
                    self._zone_buffer.append((zone, int(passes)))

    def _on_step(self) -> bool:
        evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluating:
            self._passes_buffer = []
            self._zone_buffer = []
        continue_training = super()._on_step()
        if evaluating and self._passes_buffer:
            self.latest_mean_passes = float(np.mean(self._passes_buffer))
            self.best_mean_passes = max(self.best_mean_passes or 0.0, self.latest_mean_passes)
            self.max_passes = max(self.max_passes or 0, max(self._passes_buffer))
            if self._zone_buffer:
                zones = {}
                for z, n in self._zone_buffer:
                    zones.setdefault(z, []).append(n)
                self.zone_passes = {str(z): float(np.mean(v)) for z, v in sorted(zones.items())}
            if self.verbose:
                print(f"Passes per episode: mean {self.latest_mean_passes:.2f}, "
                      f"this eval {self._passes_buffer}, best mean {self.best_mean_passes:.2f}, "
                      f"max ever {self.max_passes}"
                      + (f", per zone {self.zone_passes}" if self._zone_buffer else ""))
            self._history.append({"t": int(self.num_timesteps), "mean": self.latest_mean_passes,
                                  "max": int(max(self._passes_buffer)),
                                  "zones": self.zone_passes if self._zone_buffer else None})
            try:
                _atomic_write_text(self.run_dir / "passes.json", json.dumps(self._history))
            except OSError as e:
                if self.verbose:
                    print(f"[DashboardEvalCallback] passes.json write failed: {e}")
        if evaluating:
            self._maybe_save_best()
            # latest_model.zip = the policy as of this eval, so the dashboard's "watch current
            # policy" always has something to load (best_model.zip only changes on a new best).
            try:
                tmp = self.run_dir / "latest_model.tmp.zip"
                self.model.save(str(tmp))
                os.replace(tmp, self.run_dir / "latest_model.zip")
            except OSError as e:
                if self.verbose:
                    print(f"[DashboardEvalCallback] latest model save failed: {e}")
            status = {
                "status": "training",
                "current_timestep": int(self.num_timesteps),
                "total_timesteps": self.total_timesteps,
                "latest_eval_reward": (None if self.last_mean_reward == -np.inf
                                        else float(self.last_mean_reward)),
                "best_eval_reward": (None if self.best_mean_reward == -np.inf
                                      else float(self.best_mean_reward)),
                "latest_success_rate": (float(np.mean(self._is_success_buffer))
                                         if len(self._is_success_buffer) else None),
                "latest_mean_passes": self.latest_mean_passes,
                "best_mean_passes": self.best_mean_passes,
                "max_passes": self.max_passes,
                "zone_passes": self.zone_passes,
                "best_by": "passes" if self._passes_buffer else "reward",
                "elapsed_sec": time.time() - self.t0,
                "updated_at": datetime.now().isoformat(),
            }
            try:
                _atomic_write_text(self.live_status_path, json.dumps(status))
            except OSError as e:
                if self.verbose:
                    print(f"[DashboardEvalCallback] status write failed: {e}")
        return continue_training

    def _maybe_save_best(self):
        """best_model.zip on a new best (mean passes, mean reward), or (mean reward) without passes."""
        if self._best_path is None or self.last_mean_reward == -np.inf:
            return
        reward = float(self.last_mean_reward)
        key = (self.latest_mean_passes, reward) if self._passes_buffer else (reward,)
        if self._best_key is not None and len(self._best_key) == len(key) and key <= self._best_key:
            return
        self._best_key = key
        try:
            os.makedirs(self._best_path, exist_ok=True)
            self.model.save(os.path.join(self._best_path, "best_model"))
            if self.verbose:
                print("New best model (" + (f"{key[0]:.2f} passes per episode, " if len(key) == 2 else "")
                      + f"reward {reward:.2f})")
        except OSError as e:
            if self.verbose:
                print(f"[DashboardEvalCallback] best model save failed: {e}")


def _start_zone(info) -> int:
    """The start zone (1-9) an episode began in: HoopNetAviary's start_zone, or the onboard env's
    start_type 'zone_<k>'. 0 when the env does not use zone starts."""
    z = info.get("start_zone")
    if z:
        return int(z)
    st = info.get("start_type")
    if isinstance(st, str) and st.startswith("zone_"):
        try:
            return int(st[5:])
        except ValueError:
            return 0
    return 0


class CameraCallback(BaseCallback):
    """Writes the drone's onboard camera view to camera_latest.jpg, but only
    while camera_enabled.flag exists under the run directory. Does nothing at
    all (no render call, no file write) when the flag is absent, which is the
    default, so training pays zero extra cost unless the camera was actually
    turned on. Checked (and possibly toggled) live via
    POST /api/runs/{run_id}/camera, no restart of training needed.
    """

    def __init__(self, run_dir: Path, env_index: int = 0, hz: float = 6.0, verbose=0):
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
            # env_method runs onboard_frame() where the env lives, which works for both a
            # DummyVecEnv and a SubprocVecEnv (whose PyBullet env cannot be pickled across);
            # env_method also looks through the wrappers (Monitor, DQN's discrete wrapper).
            try:
                frame = self.training_env.env_method("onboard_frame", (200, 150),
                                                     indices=[self.env_index])[0]
            except AttributeError:           # an env class without onboard_frame()
                env = self.training_env.get_attr("unwrapped", indices=[self.env_index])[0]
                frame = get_onboard_frame(env, nth_drone=0, res=(200, 150))
            from PIL import Image
            import io
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=70)
            _atomic_write_bytes(self.frame_path, buf.getvalue())
        except Exception as e:
            if self.verbose:
                print(f"[CameraCallback] frame capture failed: {e}")
        return True


class PaceCallback(BaseCallback):
    """While realtime.flag exists in the run directory, sleep so that each vec-env step takes at
    least one control period (1/30 s): the watched env then flies at real speed, so the dashboard's
    far view and drone POV look like a real flight. The flag is toggled live from the dashboard
    (POST /api/runs/{run_id}/realtime); without it this does nothing and training runs at full speed.
    Evaluation episodes are not paced (EvalCallback steps its own env)."""

    def __init__(self, run_dir: Path, ctrl_dt: float = 1 / 30, check_every: float = 0.5,
                 clock=time.perf_counter, sleep=time.sleep, verbose=0):
        super().__init__(verbose)
        self.flag_path = Path(run_dir) / "realtime.flag"
        self.ctrl_dt = ctrl_dt
        self.check_every = check_every
        self._clock, self._sleep = clock, sleep
        self._active = False
        self._last_check = -1e9
        self._last_step = None

    def _on_step(self) -> bool:
        now = self._clock()
        if now - self._last_check >= self.check_every:
            self._last_check = now
            self._active = self.flag_path.exists()
        if not self._active:
            self._last_step = None
            return True
        if self._last_step is not None:
            wait = self._last_step + self.ctrl_dt - now
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_step = now
        return True
