"""
Reinforcement learning on the onboard-camera task (OnboardHoopEnv), started from a
behaviour clone and kept close to the demonstrations while it improves.

    # 1. a clone of the human flights + the PyBullet policy's teacher demos (human weighted x3)
    python hardware/imitation/bc_train.py --demos hardware/flights/tello_*/demo.npz \
        hardware/flights/demos/teacher_*.npz --out hardware/flights/bc/mixed --eval-episodes 20
    # 2. fine-tune it with PPO, the same demos pulling it back towards the human style
    python hardware/onboard/train_onboard.py --init hardware/flights/bc/mixed \
        --demos hardware/flights/tello_*/demo.npz hardware/flights/demos/teacher_*.npz \
        --timesteps 500000 --run-id onboard_mixed --out results/hoop/runs/onboard_mixed
    # continue that run later (its latest -> best -> final model, steps carry on)
    python hardware/onboard/train_onboard.py --init results/hoop/runs/onboard_mixed --demos ... \
        --run-id onboard_mixed2 --out results/hoop/runs/onboard_mixed2

Why the demo term. PPO started from a clone tends to forget it: the first updates are
driven by a value function that knows nothing yet, and a few bad updates undo what the clone
learned. Two things stop that here:
  * --critic-warmup-steps: the actor is frozen while the critic learns the clone's values.
  * BC regularisation: after each PPO update, the same number of gradient steps on
    bc_coef * weighted MSE(policy mean, demo action) over minibatches of the demos (human
    samples weighted --source-weight human=3 by default). bc_coef decays linearly from
    --bc-coef to --bc-coef-min over --bc-decay-steps, so RL is free to improve on the demos
    later, but never drifts all the way from them.

The run directory follows train_hoop_multi.py's contract (meta.json, live_status.json,
evaluations.npz, latest/best/final_model.zip, stop by CTRL_BREAK/SIGTERM), so the dashboard
lists, charts and stops it like any other run. meta.json carries task "onboard_hoop", so
hardware/onboard/policy.py (and fly_hoop.py --mode policy) loads best_model.zip as is.
"""
import argparse
import glob
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HW_ROOT = Path(__file__).resolve().parents[1]
CMS = HW_ROOT.parent
for p in (HW_ROOT, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# flake8: noqa: E402
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from imitation.record_demo import load_demos, parse_source_weights, sample_weights
from onboard import spec
from onboard.gym_env import OnboardHoopEnv
from telemetry_callback import DashboardEvalCallback
from vec_util import VEC_CHOICES, load_for_resume, make_train_vec_env, resolve_model_path, resolve_vec

UNTIL_STOPPED_BUDGET = 10 ** 12
EVAL_SEED_BASE = 900_000_000
DEFAULT_SOURCE_WEIGHTS = {"human": 3.0}
BC_ATTRS = ("_bc_obs", "_bc_act", "_bc_w", "_bc_rng")

_stop_requested = {"flag": False}


def _request_stop(signum, frame):
    _stop_requested["flag"] = True


class StopRequestCallback(BaseCallback):
    def _on_step(self) -> bool:
        return not _stop_requested["flag"]


class EvalSeedReset(BaseCallback):
    """Put just before the eval callback: rewinds the eval env's episode seed so every
    evaluation scores the SAME layouts. Differences between evaluations are then the policy,
    not the luck of the draw, which is what picking best_model.zip depends on."""
    def __init__(self, eval_env, eval_freq, seed):
        super().__init__()
        self.eval_env, self.eval_freq, self.seed = eval_env, eval_freq, seed

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.env_method("set_episode_seed", self.seed)
        return True


def _atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _keep_awake(on: bool):
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception as e:
        print(f"[WARN] could not change the sleep setting: {e}")


class BCRegularizedPPO(PPO):
    """PPO plus a decaying behaviour-cloning term and an optional frozen-actor warm-up.

    Saves as a plain PPO zip (the demo arrays are excluded), so PPO.load and policy.py
    read it like any other model."""

    bc_coef = 0.0
    bc_coef_min = 0.0
    bc_decay_steps = 1
    bc_batch = 256
    bc_start_steps = 0
    critic_warmup_steps = 0
    _bc_obs = _bc_act = _bc_w = _bc_rng = None

    def configure(self, demo_obs=None, demo_act=None, demo_w=None, bc_coef=1.0, bc_coef_min=0.1,
                  bc_decay_steps=300_000, bc_batch=256, critic_warmup_steps=0, seed=0):
        import torch
        self.bc_coef, self.bc_coef_min = float(bc_coef), float(bc_coef_min)
        self.bc_decay_steps = max(1, int(bc_decay_steps))
        self.bc_batch = int(bc_batch)
        self.bc_start_steps = int(self.num_timesteps)          # decay counts from THIS run's start
        self.critic_warmup_steps = int(critic_warmup_steps)
        if demo_obs is not None and len(demo_obs):
            self._bc_obs = torch.as_tensor(np.asarray(demo_obs, np.float32))
            self._bc_act = torch.as_tensor(np.clip(np.asarray(demo_act, np.float32), -1, 1))
            self._bc_w = torch.as_tensor(np.ones(len(demo_obs), np.float32) if demo_w is None
                                         else np.asarray(demo_w, np.float32))
        else:
            self._bc_obs = self._bc_act = self._bc_w = None
        self._bc_rng = np.random.default_rng(seed)
        return self

    def _excluded_save_params(self):
        return super()._excluded_save_params() + list(BC_ATTRS)

    def current_bc_coef(self) -> float:
        k = min(1.0, (self.num_timesteps - self.bc_start_steps) / self.bc_decay_steps)
        return self.bc_coef + (self.bc_coef_min - self.bc_coef) * k

    def _actor_params(self):
        pol = self.policy
        return (list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
                + [pol.log_std])

    def _warming_up(self) -> bool:
        return self.num_timesteps - self.bc_start_steps < self.critic_warmup_steps

    def train(self) -> None:
        import torch
        warm = self._warming_up()
        for prm in self._actor_params():
            prm.requires_grad_(not warm)
        super().train()                    # the ordinary PPO update (critic only while warming up)
        for prm in self._actor_params():
            prm.requires_grad_(True)
        self.logger.record("bc/critic_warmup", int(warm))
        if warm or self._bc_obs is None:
            return

        coef = self.current_bc_coef()
        if coef <= 0:
            return
        # As many BC steps as PPO just took, so the two terms stay comparable in strength.
        n_steps = self.n_epochs * max(1, (self.n_steps * self.n_envs) // self.batch_size)
        n = len(self._bc_obs)
        self.policy.set_training_mode(True)
        losses = []
        for _ in range(n_steps):
            ids = torch.as_tensor(self._bc_rng.integers(0, n, size=min(self.bc_batch, n)))
            mean = self.policy.get_distribution(self._bc_obs[ids]).distribution.mean
            w = self._bc_w[ids]
            per = ((mean - self._bc_act[ids]) ** 2).mean(dim=1)
            loss = coef * (per * w).sum() / w.sum().clamp_min(1e-8)
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item() / coef)
        self.logger.record("bc/coef", coef)
        self.logger.record("bc/mse", float(np.mean(losses)))


def _load_demo_set(patterns, weights_arg):
    files = sorted({f for pat in patterns for f in glob.glob(str(pat))})
    if not files:
        sys.exit(f"[ERROR] no demo files match {patterns}")
    d = load_demos(files)
    tasks = {m.get("task") for m in d["meta"]}
    if tasks != {spec.TASK_NAME}:
        sys.exit(f"[ERROR] demos must all be '{spec.TASK_NAME}' (onboard camera); got {tasks}")
    if d["obs"].shape[1] != spec.OBS_DIM:
        sys.exit(f"[ERROR] demo observations are {d['obs'].shape[1]} wide, expected {spec.OBS_DIM}")
    sw = dict(DEFAULT_SOURCE_WEIGHTS, **parse_source_weights(weights_arg))
    w = sample_weights(d["source"], sw)
    counts = {s: int((d["source"] == s).sum()) for s in sorted(set(d["source"].tolist()))}
    return files, d["obs"], d["intent"], w, counts, {k: v for k, v in sw.items() if k in counts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, default=None,
                    help="start from a behaviour clone (hardware/flights/bc/<name>) or continue an "
                         "earlier onboard run (its directory or a .zip). Omit to start from scratch.")
    ap.add_argument("--demos", nargs="*", default=[],
                    help="demo .npz files (globs allowed) for the BC term; none = plain PPO")
    ap.add_argument("--source-weight", nargs="*", default=None, metavar="SOURCE=W",
                    help="weight per demo source, e.g. human=3 sim_teacher=1 (default human=3)")
    ap.add_argument("--bc-coef", type=float, default=1.0)
    ap.add_argument("--bc-coef-min", type=float, default=0.1)
    ap.add_argument("--bc-decay-steps", type=int, default=300_000)
    ap.add_argument("--critic-warmup-steps", type=int, default=None,
                    help="freeze the actor this many steps while the critic learns "
                         "(default 20000 from a clone, 0 otherwise)")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default 1e-4 from a clone or an earlier run, 3e-4 from scratch)")
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh"], default="typical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--until-stopped", action="store_true")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--vec", choices=VEC_CHOICES, default="auto")
    ap.add_argument("--eval-freq", type=int, default=2500, help="per env (x n_envs timesteps)")
    ap.add_argument("--n-eval-episodes", type=int, default=20)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)

    vec = resolve_vec(args.vec, args.n_envs)
    init_path = resolve_model_path(args.init) if args.init else None
    init_meta = {}
    if init_path is not None and (init_path.parent / "meta.json").is_file():
        init_meta = json.loads((init_path.parent / "meta.json").read_text())
        if init_meta.get("task") not in (None, spec.TASK_NAME):
            sys.exit(f"[ERROR] {init_path} is a '{init_meta.get('task')}' model, not '{spec.TASK_NAME}'. "
                     "PyBullet runs cannot be loaded here; turn one into demos with "
                     "make_teacher_demos.py instead.")
        if init_meta.get("algo") not in (None, "ppo"):
            sys.exit(f"[ERROR] {init_path} is a {init_meta['algo']} model; this trainer is PPO")
    from_clone = init_meta.get("source") == "bc"
    lr = args.lr or (1e-4 if init_path is not None else 3e-4)
    warmup = args.critic_warmup_steps if args.critic_warmup_steps is not None else (
        20_000 if from_clone else 0)

    demo_files, d_obs, d_act, d_w, counts, sw = ([], None, None, None, {}, {})
    if args.demos:
        demo_files, d_obs, d_act, d_w, counts, sw = _load_demo_set(args.demos, args.source_weight)
        print(f"[INFO] BC demos: {len(d_obs)} samples {counts}, weights {sw}")

    meta = {
        "run_id": args.run_id, "algo": "ppo", "env": "onboard", "task": spec.TASK_NAME,
        "env_variant": "onboard_rc", "profile": args.profile, "seed": args.seed,
        "timesteps_requested": None if args.until_stopped else args.timesteps,
        "until_stopped": bool(args.until_stopped),
        "n_envs": args.n_envs, "vec": vec,
        "eval_freq": args.eval_freq, "eval_every_timesteps": args.eval_freq * args.n_envs,
        "n_eval_episodes": args.n_eval_episodes,
        "init_from": str(init_path) if init_path else None,
        "init_parent": init_meta.get("run_id") or (init_path.parent.name if init_path else None),
        "init_kind": ("clone" if from_clone else "run") if init_path else None,
        "init_timesteps": None,
        "learning_rate": lr, "critic_warmup_steps": warmup,
        "bc": ({"demos": [Path(f).name for f in demo_files], "samples": counts, "source_weights": sw,
                "coef": args.bc_coef, "coef_min": args.bc_coef_min,
                "decay_steps": args.bc_decay_steps} if demo_files else None),
        "pid": os.getpid(), "created_at": datetime.now().isoformat(),
        "finished_at": None, "final_status": None, "final_timesteps": None,
    }
    _atomic_write_json(run_dir / "meta.json", meta)

    env_kwargs = dict(profile=args.profile, seed=args.seed * 100_000)
    train_env = make_train_vec_env(OnboardHoopEnv, env_kwargs, args.n_envs, seed=args.seed, vec=vec,
                                   rank_seed_kwarg="seed")
    # Evaluation layouts come from a separate stream that training never draws from;
    # EvalSeedReset rewinds it before every evaluation.
    eval_env = make_train_vec_env(OnboardHoopEnv, dict(profile=args.profile, seed=EVAL_SEED_BASE),
                                  1, seed=args.seed + 1, vec="dummy")

    import torch
    torch.set_num_threads(1)
    if init_path is not None:
        try:
            model, _ = load_for_resume(BCRegularizedPPO, init_path, train_env,
                                       learning_rate=lr, lr_schedule=lambda _: lr)
        except ValueError as e:
            train_env.close()
            eval_env.close()
            sys.exit(f"[ERROR] {e}")
        model.verbose = 1
        if from_clone:
            model.num_timesteps = 0         # a clone has no RL steps of its own
        meta["init_timesteps"] = int(model.num_timesteps)
        if not args.until_stopped:
            meta["timesteps_requested"] = int(model.num_timesteps) + args.timesteps
        _atomic_write_json(run_dir / "meta.json", meta)
        print(f"[INFO] starting from {init_path} ({'clone' if from_clone else 'run'}, "
              f"{model.num_timesteps} steps), lr {lr}")
    else:
        model = BCRegularizedPPO("MlpPolicy", train_env, learning_rate=lr, seed=args.seed,
                                 device="cpu", verbose=1)
    model.configure(d_obs, d_act, d_w, bc_coef=args.bc_coef, bc_coef_min=args.bc_coef_min,
                    bc_decay_steps=args.bc_decay_steps, critic_warmup_steps=warmup, seed=args.seed)
    start_steps = int(model.num_timesteps)
    print(f"[INFO] envs {args.n_envs} ({vec}); actor frozen for the first {warmup} steps; "
          f"BC coef {args.bc_coef if demo_files else 0} -> {args.bc_coef_min if demo_files else 0}")

    t0 = time.time()
    eval_cb = DashboardEvalCallback(
        eval_env, run_dir=run_dir,
        total_timesteps=None if args.until_stopped else start_steps + args.timesteps, t0=t0,
        best_model_save_path=str(run_dir), log_path=str(run_dir), eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes, deterministic=True, render=False)
    callback = CallbackList([EvalSeedReset(eval_env, args.eval_freq, EVAL_SEED_BASE), eval_cb,
                             StopRequestCallback()])

    budget = UNTIL_STOPPED_BUDGET if args.until_stopped else args.timesteps
    final_status, final_timesteps = "crashed", 0
    if args.until_stopped:
        _keep_awake(True)
    try:
        model.learn(total_timesteps=budget, callback=callback, log_interval=10,
                    reset_num_timesteps=init_path is None)
        final_timesteps = model.num_timesteps
        final_status = "stopped" if _stop_requested["flag"] else "completed"
    except KeyboardInterrupt:
        final_status, final_timesteps = "stopped", model.num_timesteps
    finally:
        if args.until_stopped:
            _keep_awake(False)
        try:
            model.save(str(run_dir / "final_model.zip"))
        except Exception as e:
            print(f"[WARN] could not save final_model.zip: {e}")
        meta.update(finished_at=datetime.now().isoformat(), final_status=final_status,
                    final_timesteps=final_timesteps)
        _atomic_write_json(run_dir / "meta.json", meta)
        for e in (train_env, eval_env):
            try:
                e.close()
            except Exception:
                pass
    print(f"\n[DONE] {args.run_id}: {final_status} at {final_timesteps} steps in {time.time() - t0:.0f}s")
    print(f"[DONE] best model -> {run_dir}/best_model.zip")


if __name__ == "__main__":
    main()
