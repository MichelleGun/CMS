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
    # continue an earlier run's policy (its latest -> best -> final model) in a new run dir
    python train_hoop_multi.py --algo ppo --init ../results/hoop/runs/ppo_20261010-023538 \
        --run-id ppo_more --out ../results/hoop/runs/ppo_more
    # MDP v4 (2026-10-11 night): budget + shaping, hover starts, prop-wash ring, "armed" in the obs
    # (an 82-D parent is widened to 83 inputs automatically)
    python train_hoop_multi.py --algo ppo --until-stopped --n-envs 6 --reward budget --budget 10 \
        --reward-weights '{"budget_pass": 5}' --budget-shape 2 --start-z 0.8 --sway-deg 5 \
        --wash-deg 15 --init ../results/hoop/runs/ppo_20261011-212417 \
        --run-id ppo_v4 --out ../results/hoop/runs/ppo_v4
    # MDP v5: same-side laps (round the outside), streak bonus, starts from all 9 zones; these are
    # the defaults now (--lap-dir same --streak 0.5 --streak-cap 3 --start-mode zones), obs 84
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

from gym_pybullet_drones.utils.enums import ObservationType, ActionType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hoop_env import HoopFindAviary, ROOM  # noqa: E402
from hoop_net_env import HoopNetAviary, merge_reward_weights, LAP_DIRS, START_MODES  # noqa: E402
from discrete_action_wrapper import HoopFindDiscreteVel  # noqa: E402
import clearml_sync  # noqa: E402
import train_speedups as speed  # noqa: E402
from telemetry_callback import (TelemetryCallback, DashboardEvalCallback, CameraCallback,  # noqa: E402
                                PaceCallback)
from vec_util import (VEC_CHOICES, load_for_resume, make_train_vec_env, resolve_model_path,  # noqa: E402
                      resolve_vec)

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

    def _on_training_start(self):
        # A continued run starts mid-curriculum: set the right size now, not at the next boundary.
        self._last = self._scale()
        self.training_env.env_method("set_hoop_radius_scale", self._last)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every < self.training_env.num_envs:
            s = self._scale()
            if s != self._last:
                self.training_env.env_method("set_hoop_radius_scale", s)
                self._last = s
        return True


class WindCurriculumCallback(BaseCallback):
    """Ramps the ring's live wind from calm (0) to full (1) over the first `horizon` timesteps, so
    the policy learns to fly through a still ring before it has to time a swinging one."""
    def __init__(self, horizon, every=5000):
        super().__init__()
        self.horizon, self.every = horizon, every
        self._last = None

    def _scale(self):
        return round(min(1.0, self.num_timesteps / max(1.0, self.horizon)), 3)

    def _on_training_start(self):
        self._last = self._scale()
        self.training_env.env_method("set_wind", self._last)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every < self.training_env.num_envs:
            s = self._scale()
            if s != self._last:
                self.training_env.env_method("set_wind", s)
                self._last = s
        return True


def linear_schedule(start, end):
    """SB3 learning-rate schedule: `start` at the beginning of learn() -> `end` at its end."""
    def f(progress_remaining):
        return end + (start - end) * progress_remaining
    return f


# PPO settings for the net env (fresh runs). The defaults (constant lr 3e-4, 10 epochs, no KL cap,
# gamma 0.99) made ppo_20261010-093427 overwrite its policy every update (approx_kl ~0.9,
# clip_fraction ~0.8, action std collapsed to 0.06) and stall at ~6% success for 10M steps.
# gamma 0.995 at 30 Hz looks ~6.7 s ahead (0.99: ~3.3 s), far enough to see the pass from takeoff.
PPO_NET_KWARGS = dict(n_steps=2048, batch_size=256, n_epochs=5, target_kl=0.03, ent_coef=0.003,
                      gamma=0.995, gae_lambda=0.95,
                      policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
PPO_NET_LR = (3e-4, 5e-5)          # linear over --timesteps (constant 1e-4 with --until-stopped)


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
        kwargs["carry_start"] = args.carry_start
        kwargs["reverse_prob"] = args.reverse_prob
        kwargs["lean_flip_prob"] = args.lean_flip_prob
        kwargs["laps"] = args.laps
        kwargs["episode_sec"] = args.episode_sec
        kwargs["net_terminate"] = args.net_terminate
        kwargs["reward_mode"] = args.reward
        kwargs["budget"] = args.budget
        kwargs["reward_weights"] = args.reward_weights
        kwargs["wash_deg"] = args.wash_deg
        kwargs["start_z"] = args.start_z if args.start_z > 0 else None
        kwargs["armed_obs"] = args.armed_obs
        kwargs["streak_obs"] = args.armed_obs and args.streak_obs
        kwargs["lap_dir"] = args.lap_dir
        kwargs["start_mode"] = args.start_mode
        kwargs["action_repeat"] = args.action_repeat
        kwargs["act_hist"] = args.act_hist
        kwargs["post_pass_p"] = args.post_pass_p
        if args.wind_steps > 0:
            kwargs["wind"] = 0.0       # the wind curriculum ramps it up from calm
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
    ap.add_argument("--sway-deg", type=float, default=5.0,
                     help="net env: largest swing of the ring about its rope knot at full gust (deg). "
                          "The real noodle hardly swings on its own (default 5, was 20)")
    ap.add_argument("--wash-deg", type=float, default=15.0,
                     help="net env: largest tilt of the ring when the drone's prop wash pushes it away "
                          "(deg, 0 = off). The real noodle tilts up about its rope as the drone nears")
    ap.add_argument("--start-z", type=float, default=0.8,
                     help="net env: start hovering at this height (m, +-0.1), where the real Tello hands "
                          "over after its automatic takeoff; 0 = start on the floor (the old behaviour)")
    ap.add_argument("--armed-obs", action=argparse.BooleanOptionalAction, default=True,
                     help="net env: put the lap 'armed' flag in the observation (83-D). Without it, "
                          "'just passed, not re-armed' and 'lined up, go' look the same to the policy")
    ap.add_argument("--streak-obs", action=argparse.BooleanOptionalAction, default=True,
                     help="net env: put the pass count of the episode in the observation (84-D), so "
                          "the critic can see what the streak bonus will pay")
    ap.add_argument("--lap-dir", choices=LAP_DIRS, default="same",
                     help="net env, laps: 'same' = every pass the same way, flying round the outside "
                          "of the ring back to the entry side; 'alternate' = back-and-forth (v4)")
    ap.add_argument("--streak", type=float, default=0.5,
                     help="net env: pass k of an episode pays x min(1 + STREAK x (k - 1), --streak-cap) "
                          "(0 = every pass the same)")
    ap.add_argument("--streak-cap", type=float, default=3.0)
    ap.add_argument("--start-mode", choices=START_MODES, default="zones",
                     help="net env: 'zones' = each episode starts in the next of 9 zones (the 8 outer "
                          "cells of a 3x3 grid and under the hoop), round-robin; 'random' = anywhere / "
                          "carried over from the last episode (v4)")
    ap.add_argument("--budget-shape", type=float, default=None,
                     help="net env, --reward budget: weight of the potential-based shaping (MDP v4; "
                          "default 2, 0 = plain v3). Overrides budget_shape in --reward-weights")
    ap.add_argument("--carry-start", type=float, default=0.7,
                     help="net env: chance an episode starts where the previous one ended "
                          "(else anywhere on the floor)")
    ap.add_argument("--reverse-prob", type=float, default=0.5,
                     help="net env: chance an episode must fly through the ring back to front")
    ap.add_argument("--lean-flip-prob", type=float, default=0.3,
                     help="net env: chance the ring leans the other way (wind holding it over)")
    ap.add_argument("--laps", type=int, default=1,
                     help="net env: passes per episode, the direction flipping after each "
                          "(0 = endless back-and-forth until the net / time limit)")
    ap.add_argument("--episode-sec", type=float, default=None,
                     help="net env: episode time limit (s; default 20, or 120 with --reward budget, "
                          "where it is only a safety cap)")
    ap.add_argument("--reward", choices=["shaped", "budget"], default="shaped",
                     help="net env: shaped = MDP v2 (shaping + pass bonus, one pass per episode by "
                          "default); budget = MDP v3 (-0.005/step, +1 per pass, -0.05 rim, endless "
                          "passes; the episode ends when its total reaches -BUDGET or at the net)")
    ap.add_argument("--budget", type=float, default=3.0,
                     help="net env, --reward budget: the episode ends when its total reward falls to "
                          "-BUDGET (3 = 20 s for the first pass; each pass adds 6.7 s)")
    ap.add_argument("--reward-weights", type=json.loads, default=None, metavar="JSON",
                     help="net env: override reward weights, e.g. '{\"pass_bonus\": 30}' "
                          "(names: hoop_net_env.DEFAULT_REWARD_WEIGHTS)")
    ap.add_argument("--net-terminate", action=argparse.BooleanOptionalAction, default=True,
                     help="net env: touching the net ends the episode (--no-net-terminate: bounce)")
    ap.add_argument("--wind-steps", type=int, default=1_500_000,
                     help="net env: timesteps over which the ring's wind ramps from calm to full "
                          "(0 = full wind from the start; evaluation always uses full wind)")
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
    ap.add_argument("--n-envs", type=int, default=4, help="parallel env copies")
    ap.add_argument("--vec", choices=VEC_CHOICES, default="auto",
                     help="subproc = each env copy in its own process (uses n_envs cores); "
                          "dummy = all copies stepped in turn in this process (one core); "
                          "auto = subproc whenever n_envs > 1. Same learning either way, "
                          "only the wall-clock speed differs.")
    ap.add_argument("--init", type=Path, default=None,
                     help="continue from a saved policy: a .zip, or a run directory "
                          "(latest -> best -> final model). Its step count carries on.")
    ap.add_argument("--eval-freq", type=int, default=2000)
    ap.add_argument("--n-eval-episodes", type=int, default=18,
                     help="18 = every start zone twice with --start-mode zones")
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-clearml", action="store_true",
                    help="don't sync this run to ClearML (also: CLEARML_OFF=1)")
    s = ap.add_argument_group("Training speed and learning (TRAINING_SPEEDUPS.md)")
    s.add_argument("--n-eval-envs", type=int, default=None,
                   help="evaluation copies run in parallel (default: --n-envs up to 4, 1 with --vec "
                        "dummy). Net env: every evaluation flies the same fixed layouts")
    s.add_argument("--checkpoint-min", type=float, default=15.0,
                   help="save latest_model.zip + checkpoints/ every this many minutes (0 = off)")
    s.add_argument("--checkpoint-keep", type=int, default=5)
    s.add_argument("--min-std-warn", type=float, default=0.05,
                   help="warn once when the action noise falls below this (0 = off)")
    s.add_argument("--action-repeat", type=int, default=1,
                   help="net env: hold each action for N control steps (2 = decide at 15 Hz; the PID "
                        "still runs at 30 Hz). Fresh runs get gamma 0.995^N (same look-ahead in seconds)")
    s.add_argument("--act-hist", type=int, default=None,
                   help="net env: keep only the last K actions in the observation (default all 15; "
                        "4 cuts the input from 84 to 40 numbers). Fresh runs only")
    s.add_argument("--post-pass-p", type=float, default=0.0,
                   help="net env, laps: share of training episodes that start just past the ring, one "
                        "pass counted, not re-armed (practise the turn-around). Evaluation unchanged")
    s.add_argument("--zone-focus", type=float, default=0.0,
                   help="net env, --start-mode zones: training starts favour failing zones "
                        "(weight 1 + this x shortfall; 0 = round-robin)")
    s.add_argument("--budget-start", type=float, default=None,
                   help="net env, --reward budget: training budget starts here and rises to --budget "
                        "over --budget-ramp-steps (more, shorter episodes early). Evaluation: --budget")
    s.add_argument("--budget-ramp-steps", type=int, default=2_000_000)
    s.add_argument("--sde", action="store_true",
                   help="gSDE exploration (noise held ~--sde-sample-freq steps). Fresh runs only")
    s.add_argument("--sde-sample-freq", type=int, default=8)
    s.add_argument("--batch-size", type=int, default=None, help="PPO minibatch (net preset: 256)")
    s.add_argument("--norm-reward", action="store_true",
                   help="normalise training rewards (VecNormalize, reward only)")
    args = ap.parse_args()

    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)

    if args.env == "net":
        if args.reward == "budget":
            args.laps, args.net_terminate = 0, True      # endless passes; the net always ends it
            rw = dict(args.reward_weights or {})
            if args.budget_shape is not None:
                rw["budget_shape"] = args.budget_shape
            elif "budget_shape" not in rw:
                rw["budget_shape"] = 2.0
            args.reward_weights = rw
        rw = dict(args.reward_weights or {})
        rw.setdefault("budget_streak", args.streak)
        rw.setdefault("budget_streak_cap", args.streak_cap)
        if args.lap_dir == "same":
            rw.setdefault("budget_wrong", 0.5)
        args.reward_weights = rw
        if args.episode_sec is None:
            args.episode_sec = 120.0 if args.reward == "budget" else 20.0
    if args.env != "net" and (args.action_repeat != 1 or args.act_hist is not None or args.post_pass_p
                              or args.zone_focus or args.budget_start is not None):
        sys.exit("[ERROR] --action-repeat / --act-hist / --post-pass-p / --zone-focus / --budget-start "
                 "are net-env options")
    if args.init is not None and (args.sde or args.act_hist is not None):
        sys.exit("[ERROR] --sde and --act-hist change the policy's shape: start a fresh run (no --init)")
    budget = UNTIL_STOPPED_BUDGET if args.until_stopped else args.timesteps
    curriculum_steps = args.curriculum_steps or (500_000 if args.until_stopped else args.timesteps // 2)

    algo_cls, needs_discrete = ALGO_CLASSES[args.algo]
    env_variant = "discrete_vel" if needs_discrete else "continuous_vel"
    vec = resolve_vec(args.vec, args.n_envs)

    init_path, init_parent = None, None
    if args.init is not None:
        init_path = resolve_model_path(args.init)
        parent_meta = init_path.parent / "meta.json"
        if parent_meta.is_file():
            pm = json.loads(parent_meta.read_text())
            init_parent = pm.get("run_id", init_path.parent.name)
            if pm.get("algo") and pm["algo"] != args.algo:
                sys.exit(f"[ERROR] --init {init_path} is a {pm['algo']} model; "
                         f"pass --algo {pm['algo']} to continue it")
            if args.env == "net" and int(pm.get("action_repeat") or 1) != args.action_repeat:
                sys.exit(f"[ERROR] --init {init_path} decided every {pm.get('action_repeat') or 1} "
                         f"control step(s); pass --action-repeat {pm.get('action_repeat') or 1} or start fresh")
            if pm.get("act_hist") is not None and args.act_hist is None:
                args.act_hist = int(pm["act_hist"])     # keep the parent's observation layout
            if pm.get("env", "legacy") != args.env and pm.get("source") != "bc":
                sys.exit(f"[ERROR] --init {init_path} was trained on env "
                         f"'{pm.get('env', 'legacy')}', not '{args.env}'")

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
        "carry_start": args.carry_start if args.env == "net" else None,
        "reverse_prob": args.reverse_prob if args.env == "net" else None,
        "lean_flip_prob": args.lean_flip_prob if args.env == "net" else None,
        "laps": args.laps if args.env == "net" else None,
        "episode_sec": args.episode_sec if args.env == "net" else None,
        "net_terminate": args.net_terminate if args.env == "net" else None,
        "wind_steps": args.wind_steps if args.env == "net" else None,
        "reward_mode": args.reward if args.env == "net" else None,
        "budget": args.budget if args.env == "net" and args.reward == "budget" else None,
        "reward_weights": merge_reward_weights(args.reward_weights) if args.env == "net" else None,
        "wash_deg": args.wash_deg if args.env == "net" else None,
        "start_z": (args.start_z if args.start_z > 0 else None) if args.env == "net" else None,
        "armed_obs": bool(args.armed_obs) if args.env == "net" else None,
        "streak_obs": bool(args.armed_obs and args.streak_obs) if args.env == "net" else None,
        "lap_dir": args.lap_dir if args.env == "net" else None,
        "start_mode": args.start_mode if args.env == "net" else None,
        "action_repeat": args.action_repeat if args.env == "net" else None,
        "act_hist": args.act_hist if args.env == "net" else None,
        "post_pass_p": args.post_pass_p if args.env == "net" else None,
        "budget_start": args.budget_start if args.env == "net" and args.reward == "budget" else None,
        "budget_ramp_steps": (args.budget_ramp_steps if args.env == "net" and args.reward == "budget"
                              and args.budget_start is not None else None),
        "speedups": speed.speedup_meta(n_eval_envs=None, checkpoint_min=args.checkpoint_min,
                                       zone_focus=args.zone_focus, norm_reward=bool(args.norm_reward),
                                       sde=bool(args.sde), batch_size=args.batch_size),
        "obs_dim": None,
        "hyperparams": None,
        "n_envs": args.n_envs,
        "vec": vec,
        "init_from": str(init_path) if init_path else None,
        "init_parent": init_parent,
        "init_timesteps": None,
        "eval_freq": args.eval_freq,
        "eval_every_timesteps": args.eval_freq * args.n_envs,   # SB3 counts eval_freq per vec step
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

    train_env = make_train_vec_env(env_cls, env_kwargs, args.n_envs,
                                   seed=(args.seed if args.seed is not None else 0),
                                   wrapper_class=wrapper_class, vec=vec)
    eval_env_kwargs = dict(env_kwargs)
    if args.env == "net":
        eval_env_kwargs["hoop_radius_scale"] = 1.0     # always evaluate on the real-size ring
        eval_env_kwargs.pop("wind", None)              # ... in full wind
    eval_env_kwargs.pop("post_pass_p", None)           # evaluation: ordinary starts only
    eval_env_kwargs["seed"] = (args.seed + 1) if args.seed is not None else 1
    n_eval_envs = args.n_eval_envs or speed.default_eval_envs(args.n_envs, vec)
    meta["speedups"]["n_eval_envs"] = n_eval_envs
    eval_env = make_train_vec_env(env_cls, eval_env_kwargs, n_eval_envs, seed=eval_env_kwargs["seed"],
                                  wrapper_class=wrapper_class, vec="subproc" if n_eval_envs > 1 else "dummy")
    if args.norm_reward:
        train_env, eval_env = speed.wrap_reward_norm(train_env, eval_env, 0.995 ** args.action_repeat)

    print(f"[INFO] algo:             {args.algo} ({env_variant})")
    print(f"[INFO] envs:             {args.n_envs} ({vec})")
    print("[INFO] action space:     ", train_env.action_space)
    print("[INFO] observation space:", train_env.observation_space)

    # The policy is a tiny MLP: extra torch threads don't speed it up, they just spin and fight
    # with everything else on the machine (the simulator, the dashboard's 3D view), which can slow
    # training several-fold. One thread is as fast here and far more robust.
    import torch
    torch.set_num_threads(1)
    if init_path is not None:
        try:
            # an older parent without the "armed" flag / pass count is widened (zero weights for the
            # new inputs, which all sit just before the 3-number vector to the ring)
            widen = -3 if args.env == "net" and args.armed_obs else None
            over = {"batch_size": args.batch_size} if args.batch_size else {}
            model, _ = load_for_resume(algo_cls, init_path, train_env, widen_at=widen, **over)
        except ValueError as e:
            train_env.close()
            eval_env.close()
            sys.exit(f"[ERROR] {e}")
        model.verbose = 1
        meta["init_timesteps"] = int(model.num_timesteps)
        if not args.until_stopped:          # the dashboard's progress bar counts the parent's steps too
            meta["timesteps_requested"] = int(model.num_timesteps) + args.timesteps
        _atomic_write_json(run_dir / "meta.json", meta)
        print(f"[INFO] continuing from {init_path} at {model.num_timesteps} steps")
        if args.env == "net":
            print(f"[INFO] reward: {args.reward} (if the parent used another reward, the critic "
                  "re-fits to the new scale over the first updates)")
    else:
        hp = {}
        if args.algo == "ppo" and args.env == "net":
            hp = dict(PPO_NET_KWARGS)
            if args.env == "net" and args.action_repeat > 1:
                hp["gamma"] = round(PPO_NET_KWARGS["gamma"] ** args.action_repeat, 6)
            if args.batch_size:
                hp["batch_size"] = args.batch_size
            if args.sde:
                hp.update(use_sde=True, sde_sample_freq=args.sde_sample_freq)
            if args.until_stopped:
                hp["learning_rate"] = 1e-4
                meta["hyperparams"] = dict(hp)
            else:
                hp["learning_rate"] = linear_schedule(*PPO_NET_LR)
                meta["hyperparams"] = dict(hp, learning_rate=f"linear {PPO_NET_LR[0]} -> {PPO_NET_LR[1]}")
        model = algo_cls("MlpPolicy", train_env, verbose=1, **hp)
    meta["obs_dim"] = int(train_env.observation_space.shape[-1])
    _atomic_write_json(run_dir / "meta.json", meta)
    start_steps = int(model.num_timesteps)
    sync = clearml_sync.start("pybullet", run_dir, meta, no_clearml=args.no_clearml)

    t0 = time.time()
    telemetry_cb = TelemetryCallback(run_dir=run_dir, env_index=0)
    camera_cb = CameraCallback(run_dir=run_dir, env_index=0)
    eval_cb = DashboardEvalCallback(
        eval_env,
        run_dir=run_dir,
        total_timesteps=None if args.until_stopped else start_steps + args.timesteps,
        t0=t0,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    pace = PaceCallback(run_dir=run_dir, ctrl_dt=args.action_repeat / 30.0)
    callbacks = [telemetry_cb, camera_cb]
    if args.env == "net":                # fixed evaluation layouts (seed + zone), any number of copies
        callbacks.append(speed.EvalLayoutReset(eval_env, args.eval_freq, 900_000_000, args.n_eval_episodes))
    callbacks += [eval_cb, sync.callback(eval_cb), pace, StopRequestCallback(),
                  speed.ExplorationWatch(args.min_std_warn, run_dir)]
    if args.checkpoint_min > 0:
        callbacks.append(speed.TimedCheckpoint(run_dir, args.checkpoint_min, args.checkpoint_keep))
    if args.env == "net" and args.zone_focus > 0 and args.start_mode == "zones":
        callbacks.append(speed.ZoneFocus(args.zone_focus))
    if args.env == "net" and args.reward == "budget" and args.budget_start is not None:
        callbacks.append(speed.BudgetRamp(args.budget_start, args.budget, args.budget_ramp_steps))
    if args.env == "net" and args.radius_start != 1.0:
        callbacks.append(RadiusCurriculumCallback(curriculum_steps, start=args.radius_start))
    if args.env == "net" and args.wind_steps > 0:
        callbacks.append(WindCurriculumCallback(args.wind_steps))
    callback = CallbackList(callbacks)

    final_status = "crashed"
    final_timesteps = 0
    if args.until_stopped:
        _keep_awake(True)
        print("[INFO] training until stopped (dashboard Stop, or Ctrl+Break); PC kept awake")
    try:
        # reset_num_timesteps=False: a continued run's step count (and with it the curriculum
        # and the eval curve's x axis) carries on from the parent's; `budget` more steps are trained.
        model.learn(total_timesteps=budget, callback=callback, log_interval=10,
                    reset_num_timesteps=init_path is None)
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
        sync.finish(final_status)
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
