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
    # v4 (2026-10-11 night): laps, prop-wash ring, little swing, lap expert demos; a 33-input
    # parent is widened to the 34-input observation (the `pushed` field) automatically
    python hardware/onboard/make_demos.py --episodes 300 --dr 1 --sway-deg 5 --wash-deg 15 \
        --hard-start-p 0.3 --laps 0 --out hardware/flights/demos/onboard_sim_laps.npz
    python hardware/onboard/train_onboard.py --until-stopped --init results/hoop/runs/onboard_laps \
        --demos hardware/flights/demos/onboard_sim_laps.npz hardware/flights/tello_*/demo.npz \
        --dr 1 --sway-deg 5 --wash-deg 15 --hard-start-p 0.5 --laps 0 \
        --run-id onboard_v4 --out results/hoop/runs/onboard_v4

v6 (2026-10-14 overnight): --history-lags (the actor sees older observations too), --asym-critic
(the value network also gets the true state; asymmetric actor-critic, Pinto et al. 2018) and
--net-arch. See hardware/onboard/history.py and aac_policy.py. --w-dwell (default 0.05) costs
loitering in the opening; --sway-deg now defaults to 2 (the real noodle barely swings when the
drone goes straight through).

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

import clearml_sync
from imitation.record_demo import load_demos, parse_source_weights, sample_weights
from onboard import spec
from onboard.gym_env import REWARD_WEIGHTS, OnboardHoopEnv
from telemetry_callback import (CameraCallback, DashboardEvalCallback, PaceCallback, TelemetryCallback,
                                replace_retry)
from vec_util import VEC_CHOICES, load_for_resume, make_train_vec_env, resolve_model_path, resolve_vec
import train_speedups as speed

UNTIL_STOPPED_BUDGET = 10 ** 12
EVAL_SEED_BASE = 900_000_000
DEFAULT_SOURCE_WEIGHTS = {"human": 3.0}
BC_ATTRS = ("_bc_obs", "_bc_act", "_bc_w", "_bc_rng")

_stop_requested = {"flag": False}


def _request_stop(signum, frame):
    _stop_requested["flag"] = True


class StopRequestCallback(BaseCallback):
    """Stops on SIGTERM/CTRL_BREAK, or when <run_dir>/stop.flag appears (how w10_race.py ends a
    run that has fallen clearly behind: learn() returns, final_model and meta are still written)."""
    def __init__(self, run_dir=None, every=1000):
        super().__init__()
        self.flag = Path(run_dir) / "stop.flag" if run_dir is not None else None
        self.every = every

    def _on_step(self) -> bool:
        if self.flag is not None and self.n_calls % self.every == 0 and self.flag.exists():
            _stop_requested["flag"] = True
        return not _stop_requested["flag"]


class RampCallback(BaseCallback):
    """Robustness curriculum: ramps the training envs' `ramp` (how much of the domain
    randomisation and of the hard starts applies) from 0 to 1 over `horizon` timesteps, so the
    policy first learns the measured front approach. Like train_hoop_multi's wind ramp, but
    counted from THIS run's start, so a run continued from an easier one ramps up too."""
    def __init__(self, horizon, every=5000):
        super().__init__()
        self.horizon, self.every = horizon, every
        self._last = None
        self._start = 0

    def _scale(self):
        return round(min(1.0, (self.num_timesteps - self._start) / max(1.0, self.horizon)), 3)

    def _on_training_start(self):
        self._start = self.num_timesteps
        self._last = self._scale()
        self.training_env.env_method("set_ramp", self._last)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every < self.training_env.num_envs:
            s = self._scale()
            if s != self._last:
                self.training_env.env_method("set_ramp", s)
                self._last = s
        return True


def mirror_signs(width: int = spec.OBS_DIM):
    """(obs_sign, act_sign): multiplying by these reflects a sample left <-> right (the ring
    on the other side, the drone strafing and turning the other way). Lateral terms flip:
    the ring's image x (now and last seen), sideways speed vgy, roll, yaw rate, and the LR
    and YAW actions, both the commanded one and those in the action history; with the ring
    tracker, also the ring's estimated sideways position."""
    obs = np.ones(width, np.float32)
    if width == spec.TRACKER_OBS_DIM:
        obs[spec.TRACKER_SLICES["ring_est"].start + 1] = -1     # right
    sl = spec.OBS_SLICES
    obs[sl["centre"].start] = -1          # cx
    obs[sl["last_centre"].start] = -1     # last cx
    obs[sl["vel"].start + 1] = -1         # vgy
    obs[sl["att"].start + 1] = -1         # roll
    obs[sl["att"].start + 2] = -1         # yaw rate
    obs[sl["pass_heading"].start + 1] = -1    # sin of the heading since the last pass
    a0 = sl["actions"].start
    for k in range(spec.ACT_BUFFER):
        obs[a0 + k * spec.ACT_DIM + spec.ACT_LR] = -1
        obs[a0 + k * spec.ACT_DIM + spec.ACT_YAW] = -1
    act = np.ones(spec.ACT_DIM, np.float32)
    act[spec.ACT_LR] = act[spec.ACT_YAW] = -1
    return obs, act


def noise_columns():
    """(ring_cols, body_cols): observation columns that get small noise in BC augmentation.
    Ring columns only where the ring is actually found (noise on a 'not seen' zero would invent
    a sighting); body columns (velocity, attitude) always."""
    sl = spec.OBS_SLICES
    ring = [i for name in ("centre", "scale", "range") for i in range(sl[name].start, sl[name].stop)]
    body = [i for name in ("vel", "att") for i in range(sl[name].start, sl[name].stop)]
    return ring, body


BC_NOISE_STD = 0.02


def _widen_index(old_width: int) -> int:
    """Where an older policy's observation gets the newer fields: 33 wide (before `pushed`) at the
    end of `last_centre`; 34 wide (before the heading memory) just after `pushed`."""
    if old_width == spec.OBS_DIM:              # 37 -> 41: the ring tracker's fields go at the end
        return spec.OBS_DIM
    return spec.layout_slices(old_width)["last_centre"].stop + (1 if old_width == 34 else 0)


class EvalSeedReset(speed.EvalLayoutReset):
    """Put just before the eval callback: rewinds the eval env's episode seeds AND start zones so
    every evaluation scores the SAME layouts, on one eval copy or several (train_speedups.eval_plan).
    Differences between evaluations are then the policy, not the luck of the draw, which is what
    picking best_model.zip depends on. (Before 2026-10-15 only the seed was rewound, so the zones
    shifted by one between evaluations.)"""
    def __init__(self, eval_env, eval_freq, seed, n_episodes=18):
        super().__init__(eval_env, eval_freq, seed, n_episodes)


def _atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    replace_retry(tmp, path)            # OneDrive can hold the file for a moment


def _policy_spec(args, actor_dim):
    """(policy class or name, policy_kwargs) for --asym-critic / --net-arch."""
    pk = {}
    if getattr(args, "net_arch", None):
        pk["net_arch"] = [int(x) for x in args.net_arch.split(",")]
    if getattr(args, "asym_critic", False):
        from onboard.aac_policy import MaskedActorCriticPolicy
        pk["actor_dim"] = int(actor_dim)
        return MaskedActorCriticPolicy, pk
    return "MlpPolicy", pk


def _new_model(train_env, lr, args, ppo_kw, actor_dim=None) -> "BCRegularizedPPO":
    """A fresh BCRegularizedPPO with the --gamma ... --sde settings (SB3 defaults otherwise)."""
    kw = dict(ppo_kw)
    policy, pk = _policy_spec(args, actor_dim or int(train_env.observation_space.shape[0]))
    if args.sde:
        # gSDE's noise is latent features x a learned matrix, so its log_std is not the action std;
        # -2 gives roughly the gentle noise a clone starts with
        kw.update(use_sde=True, sde_sample_freq=args.sde_sample_freq)
        pk["log_std_init"] = -2.0 if args.log_std_reset is None else args.log_std_reset
    if pk:
        kw["policy_kwargs"] = pk
    return BCRegularizedPPO(policy, train_env, learning_rate=lr, seed=args.seed,
                            device="cpu", verbose=1, **kw)


def _transplant(parent, model):
    """Copy every tensor of `parent`'s policy whose name and shape match into `model`'s (a new
    policy class or noise model on the same layers). -> (copied, fresh) name lists."""
    src = parent.policy.state_dict()
    dst = model.policy.state_dict()
    copied = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    model.policy.load_state_dict({**dst, **copied})
    model.num_timesteps, model._n_updates = parent.num_timesteps, parent._n_updates
    return sorted(copied), sorted(set(dst) - set(copied))


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
    bc_augment = False
    _bc_obs = _bc_act = _bc_w = _bc_rng = None

    def configure(self, demo_obs=None, demo_act=None, demo_w=None, bc_coef=1.0, bc_coef_min=0.1,
                  bc_decay_steps=300_000, bc_batch=256, critic_warmup_steps=0, seed=0,
                  bc_augment=False):
        import torch
        self.bc_augment = bool(bc_augment)
        obs_sign, act_sign = mirror_signs(int(self.observation_space.shape[0]))
        ring_cols, body_cols = noise_columns()
        self._aug = dict(obs_sign=torch.as_tensor(obs_sign), act_sign=torch.as_tensor(act_sign),
                         ring=torch.as_tensor(ring_cols), body=torch.as_tensor(body_cols),
                         found=spec.OBS_SLICES["found"].start)
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
        return super()._excluded_save_params() + list(BC_ATTRS) + ["_aug"]

    def augment(self, obs, act):
        """Half the batch mirrored left <-> right, plus small observation noise: the few demos
        there are cover twice the situations and don't pin the policy to exact readings."""
        import torch
        a = self._aug
        flip = torch.as_tensor(self._bc_rng.random(len(obs)) < 0.5)
        obs = torch.where(flip[:, None], obs * a["obs_sign"], obs)
        act = torch.where(flip[:, None], act * a["act_sign"], act)
        gen = torch.Generator().manual_seed(int(self._bc_rng.integers(2 ** 31)))
        noise = torch.randn(obs.shape, generator=gen) * BC_NOISE_STD
        seen = (obs[:, a["found"]] > 0.5).float()[:, None]
        obs = obs.clone()
        obs[:, a["ring"]] += noise[:, a["ring"]] * seen
        obs[:, a["body"]] += noise[:, a["body"]]
        return obs, act

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
            obs_b, act_b = self._bc_obs[ids], self._bc_act[ids]
            if self.bc_augment:
                obs_b, act_b = self.augment(obs_b, act_b)
            mean = self.policy.get_distribution(obs_b).distribution.mean
            w = self._bc_w[ids]
            per = ((mean - act_b) ** 2).mean(dim=1)
            loss = coef * (per * w).sum() / w.sum().clamp_min(1e-8)
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(loss.item() / coef)
        self.logger.record("bc/coef", coef)
        self.logger.record("bc/mse", float(np.mean(losses)))


def _rel_demo_paths(files):
    """Demo files relative to hardware/flights (what the dashboard calls a demo id), so a later
    run can be started with the same demos; a file outside that folder keeps its full path."""
    root = Path(__file__).resolve().parents[1] / "flights"
    out = []
    for f in files:
        p = Path(f).resolve()
        try:
            out.append(p.relative_to(root).as_posix())
        except ValueError:
            out.append(str(p))
    return out


def _load_demo_set(patterns, weights_arg, ring_tracker=False, full_width=None):
    files = sorted({f for pat in patterns for f in glob.glob(str(pat))})
    if not files:
        sys.exit(f"[ERROR] no demo files match {patterns}")
    d = load_demos(files)
    tasks = {m.get("task") for m in d["meta"]}
    if tasks != {spec.TASK_NAME}:
        sys.exit(f"[ERROR] demos must all be '{spec.TASK_NAME}' (onboard camera); got {tasks}")
    if full_width is not None and d["obs"].shape[1] == full_width:
        pass                                   # recorded from this very env (DAgger with history)
    elif ring_tracker and full_width is None and d["obs"].shape[1] == spec.TRACKER_OBS_DIM:
        pass                                   # recorded from the tracker env itself (loop_labels.py):
        #                                        its tracker columns are live, not replayed
    elif d["obs"].shape[1] != spec.OBS_DIM:
        sys.exit(f"[ERROR] demo observations are {d['obs'].shape[1]} wide, expected {spec.OBS_DIM}"
                 + (f" or {full_width}" if full_width else ""))
    elif full_width is not None:
        sys.exit("[ERROR] with --history-lags / --asym-critic the BC demos must be recorded from the "
                 "same env (hardware/privileged/dagger_hist.py), not plain 37-wide flights")
    elif ring_tracker:                         # replayed from the demo's own columns (obs.add_tracker)
        from onboard.obs import add_tracker
        d["obs"] = add_tracker(d["obs"], d["episode"], d["t"])
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
    ap.add_argument("--reward-weights", default=None, metavar="JSON",
                    help="override reward values by name, as a JSON object, e.g. "
                         "'{\"r_passed\": 25, \"w_rim\": 0.5}' (names: onboard/gym_env.REWARD_WEIGHTS); "
                         "streak / streak_cap have their own flags")
    ap.add_argument("--ring-tracker", action="store_true",
                    help="add the 4 ring-tracker inputs (obs.RingTracker: where the ring is while it is "
                         "out of view); a 37-input parent is widened with zero weights on them")
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh", "measured"], default="typical",
                    help="'measured' = gains fitted to the real flight logs (onboard/sim.py)")
    ap.add_argument("--dr", type=float, default=0.0,
                    help="domain randomisation strength 0..1 (drone dynamics and calibration, ring "
                         "placement, a swinging ring, detector noise/latency); see onboard/sim.py")
    ap.add_argument("--sway-deg", type=float, default=2.0,
                    help="largest swing of the ring on its rope at full wind (deg), with --dr > 0. "
                         "The real noodle hardly swings when the drone goes straight through "
                         "(default 2; 5 before 2026-10-14, 25 before that)")
    ap.add_argument("--w-dwell", type=float, default=0.05,
                    help="cost per tick of loitering in the opening without pushing forward "
                         "(gym_env W_DWELL; 0 = off, as before 2026-10-14)")
    ap.add_argument("--history-lags", default=None,
                    help="actor memory: also feed the observations this many ticks ago, e.g. "
                         "'0,1,2,4,8,16' (onboard/history.py). Default: none")
    ap.add_argument("--asym-critic", action="store_true",
                    help="asymmetric actor-critic: the value network also sees the true state "
                         "(onboard/aac_policy.py); the real drone pads those inputs with zeros")
    ap.add_argument("--net-arch", default=None,
                    help="hidden layers of actor and critic, e.g. '256,256' (default SB3's 64,64)")
    ap.add_argument("--wash-deg", type=float, default=15.0,
                    help="largest tilt of the ring when the drone's prop wash pushes it away (deg, "
                         "0 = off): the real noodle tilts up about its rope as the drone nears")
    ap.add_argument("--stress-p", type=float, default=0.0,
                    help="training only: share of episodes with a slower, driftier drone (command delay "
                         "+0.06 s, drift x2; onboard/sim.py STRESS_*), for the real drone's corner clipping")
    ap.add_argument("--oblique-p", type=float, default=0.0,
                    help="training only: share of episodes that start 0.5-1.1 m from the ring, 35-80 deg "
                         "off its axis, ring in view at a slant (as from a corner of the real cage)")
    ap.add_argument("--near-pass-p", type=float, default=0.0,
                    help="training only: share of episodes that start lined up just in front of the "
                         "ring, so most of the episode is the loop after the pass (curriculum for "
                         "--laps 0). Evaluation keeps the normal starts")
    ap.add_argument("--hard-start-p", type=float, default=0.0,
                    help="share of episodes starting behind / beside / below the ring or anywhere, "
                         "often facing away (the ring must be searched for)")
    ap.add_argument("--dr-steps", type=int, default=500_000,
                    help="timesteps over which --dr and --hard-start-p ramp up from 0 (0 = full "
                         "from the start); evaluation is always at full strength")
    ap.add_argument("--laps", type=int, default=0,
                    help="passes per episode: 0 = endless back-and-forth (fly through, clear the ring, "
                         "turn round, come back through), 1 = the episode ends at the first crossing")
    ap.add_argument("--lap-dir", choices=["alternate", "same"], default="same",
                    help="laps: 'same' = every pass the same way, round the outside of the ring back to "
                         "the entry side; 'alternate' = back-and-forth")
    ap.add_argument("--streak", type=float, default=0.5,
                    help="pass k of an episode pays R_PASSED x min(1 + STREAK x (k - 1), --streak-cap)")
    ap.add_argument("--streak-cap", type=float, default=3.0)
    ap.add_argument("--start-mode", choices=["ring", "zones"], default="zones",
                    help="'zones' = starts cycle through 9 zones of the cage (8 outer cells of a 3x3 "
                         "grid and under the hoop), facing anywhere; 'ring' = front runway + "
                         "--hard-start-p")
    ap.add_argument("--bc-augment", action=argparse.BooleanOptionalAction, default=True,
                    help="mirror half of each demo batch left/right and add small observation noise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--until-stopped", action="store_true")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--vec", choices=VEC_CHOICES, default="auto")
    ap.add_argument("--eval-freq", type=int, default=2500, help="per env (x n_envs timesteps)")
    ap.add_argument("--n-eval-episodes", type=int, default=18,
                    help="18 = every start zone twice with --start-mode zones")
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-clearml", action="store_true",
                    help="don't sync this run to ClearML (also: CLEARML_OFF=1)")
    # PPO settings (Week 10 experiments). Left out = SB3's default from scratch, or the parent's
    # value when continuing a run, i.e. exactly what every run before these flags used.
    s = ap.add_argument_group("Training speed (TRAINING_SPEEDUPS.md)")
    s.add_argument("--n-eval-envs", type=int, default=None,
                   help="evaluation copies run in parallel (default: --n-envs, at most 4, or 1 with "
                        "--vec dummy). Same layouts as 1 copy (scores equal up to float rounding); each "
                        "copy is a process (~0.3-0.5 GB)")
    s.add_argument("--checkpoint-min", type=float, default=15.0,
                   help="save latest_model.zip + checkpoints/ every this many minutes (0 = off)")
    s.add_argument("--checkpoint-keep", type=int, default=5, help="checkpoints/ keeps the newest N")
    s.add_argument("--min-std-warn", type=float, default=0.05,
                   help="warn once when the action noise falls below this (0 = off)")
    s.add_argument("--zone-focus", type=float, default=0.0,
                   help="training starts favour the zones that fail: weight 1 + this x shortfall "
                        "(0 = round-robin, 2 = the worst zone up to 3x as often). Evaluation unchanged")
    s.add_argument("--stall-sec", type=float, default=0.0,
                   help="cut a training episode after this many seconds without a pass (0 = off; "
                        "try 30). Evaluation unchanged")
    s.add_argument("--norm-reward", action="store_true",
                   help="normalise training rewards (VecNormalize, reward only): steadier critic; "
                        "nothing changes for replays or the drone")

    g = ap.add_argument_group("PPO settings (default: SB3 default / the parent's value)")
    g.add_argument("--gamma", type=float, default=None,
                   help="discount; 0.99 = ~6.7 s horizon at 15 Hz, 0.995 = ~13 s (one same-side lap)")
    g.add_argument("--gae-lambda", type=float, default=None)
    g.add_argument("--ent-coef", type=float, default=None,
                   help="entropy bonus that keeps exploration noise from collapsing (0.0 so far)")
    g.add_argument("--target-kl", type=float, default=None,
                   help="stop an update's epochs early once the mean KL to the rollout policy passes "
                        "this (trust-region brake, e.g. 0.02)")
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--n-epochs", type=int, default=None)
    g.add_argument("--sde", action="store_true",
                   help="gSDE: smooth, state-dependent exploration noise instead of independent noise "
                        "every tick. With --init the parent's layers are copied in; only the noise "
                        "parameters start fresh (at --log-std-reset, default -2)")
    g.add_argument("--sde-sample-freq", type=int, default=4,
                   help="with --sde: draw a new noise matrix every this many ticks")
    g.add_argument("--log-interval", type=int, default=10,
                   help="write the train/* rollout/* table (stdout, progress.csv, ClearML) every N updates")
    g.add_argument("--log-std-reset", type=float, default=None,
                   help="set the exploration log_std to this before training (e.g. -1 = std 0.37) "
                        "to undo a collapsed one")
    args = ap.parse_args(argv)
    ppo_kw = {k: v for k, v in (("gamma", args.gamma), ("gae_lambda", args.gae_lambda),
                                ("ent_coef", args.ent_coef), ("target_kl", args.target_kl),
                                ("batch_size", args.batch_size), ("n_epochs", args.n_epochs))
              if v is not None}

    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)

    vec = resolve_vec(args.vec, args.n_envs)
    n_eval_envs = args.n_eval_envs or speed.default_eval_envs(args.n_envs, vec)
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

    reward_over = {}
    if args.reward_weights:
        try:
            reward_over = json.loads(args.reward_weights)
            assert isinstance(reward_over, dict)
        except (ValueError, AssertionError):
            sys.exit("[ERROR] --reward-weights must be a JSON object, e.g. '{\"r_passed\": 25}'")
        bad = sorted(set(reward_over) - set(REWARD_WEIGHTS))
        if bad:
            sys.exit(f"[ERROR] unknown reward weight(s) {bad}; names: {sorted(REWARD_WEIGHTS)}")
        reward_over = {k: v for k, v in reward_over.items() if k not in ("streak", "streak_cap")}

    from onboard.history import PRIV_DIM, parse_lags, stacked_width
    lags = parse_lags(args.history_lags)
    frame_w = spec.TRACKER_OBS_DIM if args.ring_tracker else spec.OBS_DIM
    actor_dim = stacked_width(frame_w, lags)
    full_w = stacked_width(frame_w, lags, args.asym_critic)
    v6 = full_w != frame_w
    if v6 and args.bc_augment:
        print("[INFO] --bc-augment is off with history / privileged inputs (its mirror map is per frame)")
        args.bc_augment = False
    if args.w_dwell:
        reward_over = dict(reward_over, w_dwell=args.w_dwell)

    demo_files, d_obs, d_act, d_w, counts, sw = ([], None, None, None, {}, {})
    if args.demos:
        demo_files, d_obs, d_act, d_w, counts, sw = _load_demo_set(args.demos, args.source_weight,
                                                                       args.ring_tracker,
                                                                       full_w if v6 else None)
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
        "reward": dict(REWARD_WEIGHTS, **reward_over, streak=args.streak, streak_cap=args.streak_cap),
        "reward_overrides": reward_over,       # what this run changed from the defaults
        "laps": args.laps, "lap_dir": args.lap_dir,
        "streak": args.streak, "streak_cap": args.streak_cap,
        "robust": {"dr": args.dr, "sway_deg": args.sway_deg, "hard_start_p": args.hard_start_p,
                   "wash_deg": args.wash_deg, "start_mode": args.start_mode,
                   "dr_steps": args.dr_steps, "bc_augment": bool(args.bc_augment),
                   "near_pass_p": args.near_pass_p, "oblique_p": args.oblique_p,
                   "stress_p": args.stress_p},
        "obs_dim": full_w,
        "frame_dim": frame_w,            # one frame: what ObsBuilder builds on the drone
        "ring_tracker": bool(args.ring_tracker),
        "obs_history": list(lags) if lags else None,
        "asym_critic": bool(args.asym_critic), "actor_dim": actor_dim,
        "priv_dim": PRIV_DIM if args.asym_critic else 0,
        "net_arch": args.net_arch,
        "ctrl_hz": spec.CTRL_HZ,
        "yaw_convention": spec.YAW_CONVENTION,
        "bc": ({"demos": [Path(f).name for f in demo_files], "demo_paths": _rel_demo_paths(demo_files),
                "samples": counts, "source_weights": sw,
                "coef": args.bc_coef, "coef_min": args.bc_coef_min,
                "decay_steps": args.bc_decay_steps} if demo_files else None),
        "speedups": speed.speedup_meta(
            n_eval_envs=n_eval_envs, checkpoint_min=args.checkpoint_min, zone_focus=args.zone_focus,
            stall_sec=args.stall_sec, norm_reward=bool(args.norm_reward)),
        "pid": os.getpid(), "created_at": datetime.now().isoformat(),
        "finished_at": None, "final_status": None, "final_timesteps": None,
    }
    _atomic_write_json(run_dir / "meta.json", meta)

    robust = dict(dr=args.dr, sway_deg=args.sway_deg, hard_start_p=args.hard_start_p, laps=args.laps,
                  wash_deg=args.wash_deg, lap_dir=args.lap_dir, start_mode=args.start_mode,
                  streak=args.streak, streak_cap=args.streak_cap, ring_tracker=args.ring_tracker,
                  reward_weights=reward_over or None, history_lags=lags, privileged=args.asym_critic)
    ramped = args.dr_steps > 0 and (args.dr > 0 or args.hard_start_p > 0)
    env_kwargs = dict(profile=args.profile, seed=args.seed * 100_000, **robust,
                      ramp=0.0 if ramped else 1.0, near_pass_p=args.near_pass_p,
                      oblique_p=args.oblique_p, stall_sec=args.stall_sec, stress_p=args.stress_p)
    train_env = make_train_vec_env(OnboardHoopEnv, env_kwargs, args.n_envs, seed=args.seed, vec=vec,
                                   rank_seed_kwarg="seed")
    # Evaluation layouts come from a separate stream that training never draws from;
    # EvalSeedReset rewinds it before every evaluation.
    # ... always at full robustness strength: its success rate is the robust one
    # Several copies fly the same layouts one copy would (EvalSeedReset hands each its block).
    eval_env = make_train_vec_env(OnboardHoopEnv, dict(profile=args.profile, seed=EVAL_SEED_BASE,
                                                       **robust, ramp=1.0),
                                  n_eval_envs, seed=args.seed + 1,
                                  vec="subproc" if n_eval_envs > 1 else "dummy")
    if args.norm_reward:
        train_env, eval_env = speed.wrap_reward_norm(train_env, eval_env, args.gamma or 0.99)

    import torch
    torch.set_num_threads(1)
    if init_path is not None:
        try:
            # a parent from before the newer observation fields (33 / 34 inputs) is widened, with
            # zero weights on the new inputs, so it starts out flying exactly as it did
            def widen_at(old_w):
                # a plain one-frame parent: its inputs are the current frame, which comes first
                return old_w if (v6 and old_w == frame_w) else _widen_index(old_w)
            model, _ = load_for_resume(BCRegularizedPPO, init_path, train_env,
                                       widen_at=widen_at,
                                       learning_rate=lr, lr_schedule=lambda _: lr, **ppo_kw)
        except ValueError as e:
            train_env.close()
            eval_env.close()
            sys.exit(f"[ERROR] {e}")
        if spec.old_yaw_convention(init_meta):
            # trained before the clockwise yaw convention: convert, so it starts out flying
            # exactly as it did (onboard/spec.py, YAW_CONVENTION)
            from onboard.policy import convert_old_yaw
            if convert_old_yaw(model):
                print("[INFO] parent converted to the clockwise yaw convention")
            meta["init_yaw_converted"] = True
        want_cls = _policy_spec(args, actor_dim)[0]
        same_cls = ((want_cls == "MlpPolicy" and type(model.policy).__name__ == "ActorCriticPolicy")
                    or (want_cls != "MlpPolicy" and isinstance(model.policy, want_cls)))
        if args.sde or not same_cls or args.net_arch:
            # gSDE has different noise parameters, an asymmetric critic a different policy class, so
            # the parent cannot be loaded as is: build a fresh model and copy every layer whose
            # shape matches (all of them for the same --net-arch; none of the noise for gSDE).
            parent = model
            model = _new_model(train_env, lr, args, ppo_kw, actor_dim)
            copied, fresh = _transplant(parent, model)
            print(f"[INFO] new policy ({type(model.policy).__name__}): copied {len(copied)} tensors "
                  f"from the parent (fresh: {fresh})")
            meta["init_transplant"] = {"copied": len(copied), "fresh": fresh}
            del parent
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
        model = _new_model(train_env, lr, args, ppo_kw, actor_dim)
    if args.log_std_reset is not None and not args.sde:
        with torch.no_grad():
            model.policy.log_std.fill_(args.log_std_reset)
        print(f"[INFO] exploration log_std reset to {args.log_std_reset} (std {np.exp(args.log_std_reset):.2f})")
    meta["ppo"] = {"gamma": model.gamma, "gae_lambda": model.gae_lambda, "ent_coef": model.ent_coef,
                   "target_kl": model.target_kl, "batch_size": model.batch_size,
                   "n_epochs": model.n_epochs, "n_steps": model.n_steps, "use_sde": bool(model.use_sde),
                   "sde_sample_freq": model.sde_sample_freq if model.use_sde else None,
                   "log_std_reset": args.log_std_reset}
    _atomic_write_json(run_dir / "meta.json", meta)
    # progress.csv in the run folder: every train/* rollout/* bc/* value SB3 prints, so runs can be
    # compared offline (tutorial/week10_compare.ipynb) as well as in ClearML
    from stable_baselines3.common.logger import configure as sb3_configure
    model.set_logger(sb3_configure(str(run_dir), ["stdout", "csv"]))
    model.configure(d_obs, d_act, d_w, bc_coef=args.bc_coef, bc_coef_min=args.bc_coef_min,
                    bc_decay_steps=args.bc_decay_steps, critic_warmup_steps=warmup, seed=args.seed,
                    bc_augment=args.bc_augment)
    start_steps = int(model.num_timesteps)
    sync = clearml_sync.start("onboard-rl", run_dir, meta, no_clearml=args.no_clearml)
    print(f"[INFO] evaluation on {n_eval_envs} cop{'y' if n_eval_envs == 1 else 'ies'}; "
          f"checkpoints every {args.checkpoint_min:g} min" + (f"; zone focus {args.zone_focus:g}" if args.zone_focus else "")
          + (f"; stall cut {args.stall_sec:g} s" if args.stall_sec else "") + ("; reward normalised" if args.norm_reward else ""))
    print(f"[INFO] envs {args.n_envs} ({vec}); actor frozen for the first {warmup} steps; "
          f"BC coef {args.bc_coef if demo_files else 0} -> {args.bc_coef_min if demo_files else 0}")

    t0 = time.time()
    eval_cb = DashboardEvalCallback(
        eval_env, run_dir=run_dir,
        total_timesteps=None if args.until_stopped else start_steps + args.timesteps, t0=t0,
        best_model_save_path=str(run_dir), log_path=str(run_dir), eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes, deterministic=True, render=False)
    # the dashboard's live view: the watched env's pose + ring, its camera view, and the
    # "real time" toggle (the same callbacks as the PyBullet trainer, fed through env methods)
    watch = [TelemetryCallback(run_dir=run_dir, env_index=0, state_method="telemetry",
                               ctrl_hz=spec.CTRL_HZ),
             CameraCallback(run_dir=run_dir, env_index=0),
             PaceCallback(run_dir=run_dir, ctrl_dt=spec.CTRL_DT)]
    if ramped:
        watch.append(RampCallback(args.dr_steps))
    extra = [speed.ExplorationWatch(args.min_std_warn, run_dir)]
    if args.checkpoint_min > 0:
        extra.append(speed.TimedCheckpoint(run_dir, args.checkpoint_min, args.checkpoint_keep))
    if args.zone_focus > 0 and args.start_mode == "zones":
        extra.append(speed.ZoneFocus(args.zone_focus))
    callback = CallbackList([*watch, *extra,
                             EvalSeedReset(eval_env, args.eval_freq, EVAL_SEED_BASE, args.n_eval_episodes),
                             eval_cb, sync.callback(eval_cb), StopRequestCallback(run_dir)])

    budget = UNTIL_STOPPED_BUDGET if args.until_stopped else args.timesteps
    final_status, final_timesteps = "crashed", 0
    if args.until_stopped:
        _keep_awake(True)
    try:
        model.learn(total_timesteps=budget, callback=callback, log_interval=args.log_interval,
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
        sync.finish(final_status)
        for e in (train_env, eval_env):
            try:
                e.close()
            except Exception:
                pass
    print(f"\n[DONE] {args.run_id}: {final_status} at {final_timesteps} steps in {time.time() - t0:.0f}s")
    print(f"[DONE] best model -> {run_dir}/best_model.zip")


if __name__ == "__main__":
    main()
