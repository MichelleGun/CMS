"""
Multi-algorithm training for the hoop-finding task, driven by the web
dashboard. Writes into a run-scoped directory (results/hoop/runs/<run_id>/)
so multiple runs, across algorithms, do not overwrite each other, and the
dashboard's file-based "database" (see webapp/backend/run_manager.py) has
somewhere to read from.

For manual/CLI use without the dashboard, scripts/train_hoop_find.py (PPO
only, writes to results/hoop/ directly) still works exactly as before, this
script does not replace it.

Usage:
    conda activate drones
    python train_hoop_multi.py --algo ppo --run-id ppo_20260930-143210 \
        --out ../results/hoop/runs/ppo_20260930-143210
    python train_hoop_multi.py --algo dqn --seed 3 --timesteps 500000 \
        --run-id dqn_test --out ../results/hoop/runs/dqn_test
    python train_hoop_multi.py --algo sac --start-pos 0.3 -0.2 0.1 \
        --target-pos -0.4 0.5 0.9 --run-id sac_fixed --out ../results/hoop/runs/sac_fixed
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from gym_pybullet_drones.utils.enums import ObservationType, ActionType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hoop_env import HoopFindAviary, ROOM  # noqa: E402
from hoop_net_env import HoopNetAviary  # noqa: E402
from discrete_action_wrapper import HoopFindDiscreteVel  # noqa: E402
from telemetry_callback import (TelemetryCallback, DashboardEvalCallback, CameraCallback,  # noqa: E402
                                PaceCallback)

OBS = ObservationType.KIN

# --algo -> (SB3 class, needs the discrete-action wrapper)
ALGO_CLASSES = {
    "ppo": (PPO, False),
    "a2c": (A2C, False),
    "sac": (SAC, False),
    "td3": (TD3, False),
    "dqn": (DQN, True),
}

# Windows raises SIGBREAK on Ctrl+Break / a process-group break signal; POSIX
# uses SIGTERM. Either one sets this flag, which a callback checks each step
# so model.learn() can stop cleanly instead of being killed mid-write.
_stop_requested = {"flag": False}


def _request_stop(signum, frame):
    _stop_requested["flag"] = True


UNTIL_STOPPED_BUDGET = 10 ** 12      # "run until Stop": a budget no run will ever reach


class RadiusCurriculumCallback(BaseCallback):
    """Widens the ring opening early in training (scale `start` -> 1.0 over the first `horizon`
    timesteps) so the policy first learns to fly through a forgiving ring."""
    def __init__(self, horizon, start=1.5, every=5000):
        super().__init__()
        self.horizon, self.start, self.every = horizon, start, every
        self._last = None

    def _scale(self):
        k = min(1.0, self.num_timesteps / max(1.0, self.horizon))
        return round(self.start + (1.0 - self.start) * k, 3)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every < self.training_env.num_envs:
            s = self._scale()
            if s != self._last:
                self.training_env.env_method("set_hoop_radius_scale", s)
                self._last = s
        return True


class StopRequestCallback(BaseCallback):
    """Returns False (stop training) once a SIGTERM/SIGBREAK has set the
    module-level stop flag, so model.learn() exits its loop cleanly instead
    of the process being killed mid-write."""
    def _on_step(self) -> bool:
        return not _stop_requested["flag"]


def _keep_awake(on: bool):
    """Windows: stop the PC from going to sleep while an overnight run trains (the screen may
    still turn off). Released when the run ends. No-op elsewhere."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception as e:
        print(f"[WARN] could not change the sleep setting: {e}")


def _atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _make_env_kwargs(args):
    kwargs = dict(obs=OBS, act=ActionType.VEL, room=ROOM)
    if args.env == "net":
        kwargs["ring_jitter"] = args.ring_jitter
        kwargs["hoop_radius_scale"] = args.radius_start
        kwargs["sway_deg"] = args.sway_deg
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.start_pos is not None:
        kwargs["start_pos"] = args.start_pos
    if args.target_pos is not None:
        kwargs["target_pos"] = args.target_pos
    return kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(ALGO_CLASSES), default="ppo")
    ap.add_argument("--env", choices=["net", "legacy"], default="net",
                     help="net = 2x2x2 m net + physical ring, floor starts, pass-through success "
                          "(hoop_net_env.py); legacy = the original point-reaching HoopFindAviary")
    ap.add_argument("--ring-jitter", type=float, default=1.0,
                     help="net env: 0 = fixed ring, 1 = full per-episode jitter of pose/facing/size")
    ap.add_argument("--radius-start", type=float, default=1.5,
                     help="net env: opening scale at the start of training, shrinking to 1.0 over "
                          "the first half (1.0 disables the curriculum)")
    ap.add_argument("--sway-deg", type=float, default=20.0,
                     help="net env: largest swing of the ring about its rope knot at full gust (deg)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--target-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--until-stopped", action="store_true",
                     help="ignore --timesteps and train until stopped (dashboard Stop / Ctrl+Break); "
                          "keeps the PC awake meanwhile")
    ap.add_argument("--curriculum-steps", type=int, default=None,
                     help="net env: timesteps over which the wider ring shrinks to real size "
                          "(default: half of --timesteps, or 500k with --until-stopped)")
    ap.add_argument("--n-envs", type=int, default=4,
                     help="parallel env copies. Note: stable_baselines3's make_vec_env "
                          "defaults to DummyVecEnv, which steps these sequentially in "
                          "this one process, not across separate CPU cores.")
    ap.add_argument("--eval-freq", type=int, default=2000)
    ap.add_argument("--n-eval-episodes", type=int, default=10)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)

    budget = UNTIL_STOPPED_BUDGET if args.until_stopped else args.timesteps
    curriculum_steps = args.curriculum_steps or (500_000 if args.until_stopped else args.timesteps // 2)

    algo_cls, needs_discrete = ALGO_CLASSES[args.algo]
    env_variant = "discrete_vel" if needs_discrete else "continuous_vel"

    meta = {
        "run_id": args.run_id,
        "algo": args.algo,
        "env_variant": env_variant,
        "env": args.env,
        "seed": args.seed,
        "start_override": args.start_pos,
        "target_override": args.target_pos,
        "timesteps_requested": None if args.until_stopped else args.timesteps,
        "until_stopped": bool(args.until_stopped),
        "curriculum_steps": curriculum_steps if args.env == "net" else None,
        "sway_deg": args.sway_deg if args.env == "net" else None,
        "n_envs": args.n_envs,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "pid": os.getpid(),
        "created_at": datetime.now().isoformat(),
        "finished_at": None,
        "final_status": None,
        "final_timesteps": None,
    }
    _atomic_write_json(run_dir / "meta.json", meta)

    env_cls = HoopNetAviary if args.env == "net" else HoopFindAviary
    env_kwargs = _make_env_kwargs(args)
    wrapper_class = HoopFindDiscreteVel if needs_discrete else None

    train_env = make_vec_env(env_cls, env_kwargs=env_kwargs, n_envs=args.n_envs,
                              seed=(args.seed if args.seed is not None else 0),
                              wrapper_class=wrapper_class)
    eval_env_kwargs = dict(env_kwargs)
    if args.env == "net":
        eval_env_kwargs["hoop_radius_scale"] = 1.0     # always evaluate on the real-size ring
    eval_env_kwargs["seed"] = (args.seed + 1) if args.seed is not None else 1
    eval_env = make_vec_env(env_cls, env_kwargs=eval_env_kwargs, n_envs=1,
                             seed=eval_env_kwargs["seed"], wrapper_class=wrapper_class)

    print(f"[INFO] algo:             {args.algo} ({env_variant})")
    print("[INFO] action space:     ", train_env.action_space)
    print("[INFO] observation space:", train_env.observation_space)

    # The policy is a tiny MLP: extra torch threads don't speed it up, they just spin and fight
    # with everything else on the machine (the simulator, the dashboard's 3D view), which can slow
    # training several-fold. One thread is as fast here and far more robust.
    import torch
    torch.set_num_threads(1)
    model = algo_cls("MlpPolicy", train_env, verbose=1)

    t0 = time.time()
    telemetry_cb = TelemetryCallback(run_dir=run_dir, env_index=0)
    camera_cb = CameraCallback(run_dir=run_dir, env_index=0)
    eval_cb = DashboardEvalCallback(
        eval_env,
        run_dir=run_dir,
        total_timesteps=None if args.until_stopped else args.timesteps,
        t0=t0,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks = [telemetry_cb, camera_cb, eval_cb, PaceCallback(run_dir=run_dir), StopRequestCallback()]
    if args.env == "net" and args.radius_start != 1.0:
        callbacks.append(RadiusCurriculumCallback(curriculum_steps, start=args.radius_start))
    callback = CallbackList(callbacks)

    final_status = "crashed"
    final_timesteps = 0
    if args.until_stopped:
        _keep_awake(True)
        print("[INFO] training until stopped (dashboard Stop, or Ctrl+Break); PC kept awake")
    try:
        model.learn(total_timesteps=budget, callback=callback, log_interval=10)
        final_timesteps = model.num_timesteps
        final_status = "stopped" if _stop_requested["flag"] else "completed"
    except KeyboardInterrupt:
        final_status = "stopped"
        final_timesteps = model.num_timesteps
    finally:
        if args.until_stopped:
            _keep_awake(False)
        try:
            model.save(str(run_dir / "final_model.zip"))
        except Exception as e:
            print(f"[WARN] could not save final_model.zip: {e}")
        meta["finished_at"] = datetime.now().isoformat()
        meta["final_status"] = final_status
        meta["final_timesteps"] = final_timesteps
        _atomic_write_json(run_dir / "meta.json", meta)
        try:
            train_env.close()
            eval_env.close()
        except Exception:
            pass

    print(f"\n[DONE] {args.run_id}: {final_status} at {final_timesteps} steps "
          f"in {time.time() - t0:.0f}s")
    print(f"[DONE] best model -> {run_dir}/best_model.zip")


if __name__ == "__main__":
    main()
