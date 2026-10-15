"""
Training speed-ups shared by both trainers (scripts/train_hoop_multi.py for PyBullet and
hardware/onboard/train_onboard.py for the onboard-camera task). See TRAINING_SPEEDUPS.md.

    eval_plan / EvalLayoutReset   evaluation on several env copies at once, on the same episodes
                                  (seed and start zone) as one copy, rewound before every evaluation
    TimedCheckpoint               latest_model.zip + checkpoints/ on a wall-clock timer, so a crash
                                  loses minutes, not everything since the last evaluation
    ExplorationWatch              logs the policy's action noise; warns once if it collapses
    ZoneFocus / zone_weights      training starts drawn more often from the zones it fails in
    BudgetRamp                    PyBullet budget mode: a small budget first, raised over training
    wrap_reward_norm              VecNormalize on the reward only (nothing to carry to the drone)

Measured on the w10_n9_O_oblique setup (2026-10-15): one 36-episode evaluation took 33 s on one
core, while the 20k training steps between evaluations took ~22 s with 2 envs, so evaluation was
~60 % of the wall clock.

Same layouts, not bit-identical scores: with 2+ copies the policy is computed for a batch of
observations, which rounds differently (~1e-7) from one observation at a time, and a 90 s closed-loop
flight amplifies that. On the same 36 layouts the best O model scored 1.25 passes/ep with 1 copy and
1.08 with 2, 3 or 4 (which agree exactly with each other). That spread is the evaluation's own noise,
which is why rescore_checkpoints.py re-scores candidates on more episodes.
"""
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

N_ZONES = 9


# ------------------------------------------------------------------------------------------------
# evaluation on several copies, identical to one copy

def default_eval_envs(n_envs: int, vec: str = "subproc", cap: int = 4) -> int:
    """How many evaluation copies when the flag is left at auto. With subprocess training envs: as
    many as training envs (their cores sit idle while an evaluation runs), at most `cap` (each copy
    is a process, ~0.3-0.5 GB). With --vec dummy (chosen to save memory / cores): 1, as before."""
    if vec == "dummy":
        return 1
    return max(1, min(int(n_envs), int(cap)))


def eval_plan(n_episodes: int, n_envs: int):
    """[(first episode index, episodes)] per eval copy, in the order SB3's evaluate_policy hands
    them out (copy i runs (n_episodes + i) // n_envs episodes). Giving copy i the episode seeds and
    zones starting at its first index makes the union of all copies exactly the episodes 0 .. n-1
    one copy would have flown."""
    n_envs = max(1, int(n_envs))
    counts = [(int(n_episodes) + i) // n_envs for i in range(n_envs)]
    plan, first = [], 0
    for c in counts:
        plan.append((first, c))
        first += c
    return plan


def _base_venv(venv):
    while hasattr(venv, "venv"):
        venv = venv.venv
    return venv


def reset_eval_layouts(eval_env, seed: int, n_episodes: int):
    """Rewind every eval copy to its block of the fixed evaluation layouts (seed + episode index,
    start zone = episode index mod 9). Envs without the methods are left alone."""
    plan = eval_plan(n_episodes, eval_env.num_envs)
    base = _base_venv(eval_env)
    if hasattr(base, "_seeds"):              # make_vec_env's one-off seeds would override ours
        base._seeds = [None] * base.num_envs
    for i, (first, _count) in enumerate(plan):
        for method, value in (("set_episode_seed", seed + first), ("set_zone_index", first)):
            try:
                eval_env.env_method(method, value, indices=[i])
            except AttributeError:
                pass


class EvalLayoutReset(BaseCallback):
    """Put just before the eval callback (same eval_freq): every evaluation flies the same
    layouts, whatever the number of eval copies, so differences are the policy, not the draw."""

    def __init__(self, eval_env, eval_freq: int, seed: int, n_episodes: int):
        super().__init__()
        self.eval_env, self.eval_freq = eval_env, int(eval_freq)
        self.seed, self.n_episodes = int(seed), int(n_episodes)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            reset_eval_layouts(self.eval_env, self.seed, self.n_episodes)
        return True


# ------------------------------------------------------------------------------------------------
# checkpoints on a timer

def _replace(tmp: Path, path: Path):
    try:
        from telemetry_callback import replace_retry     # OneDrive-safe (retries a locked target)
    except ImportError:                                  # pragma: no cover
        import os
        replace_retry = os.replace
    replace_retry(tmp, path)


class TimedCheckpoint(BaseCallback):
    """Every `every_min` minutes of wall clock: save the policy to latest_model.zip (what
    --init / --resume-crashed continue from) and to checkpoints/step_<steps>.zip, keeping the
    newest `keep`. Also on the way out, so even Ctrl+C keeps the last minutes."""

    def __init__(self, run_dir, every_min: float = 15.0, keep: int = 5):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.every_s = float(every_min) * 60.0
        self.keep = int(keep)
        self._last = None

    def _on_training_start(self):
        self._last = time.time()

    def _on_step(self) -> bool:
        if self.every_s > 0 and time.time() - self._last >= self.every_s:
            self.save()
        return True

    def _on_training_end(self):
        if self.every_s > 0:
            self.save()

    def save(self):
        self._last = time.time()
        ck = self.run_dir / "checkpoints"
        try:
            ck.mkdir(parents=True, exist_ok=True)
            name = f"step_{int(self.num_timesteps):012d}.zip"
            tmp = ck / ("tmp_" + name)
            self.model.save(str(tmp))
            _replace(tmp, ck / name)
            meta = self.run_dir / "meta.json"       # the flight loader (policy.load_policy) reads
            if meta.is_file():                      # <stem>.meta.json beside a model
                (ck / name.replace(".zip", ".meta.json")).write_text(meta.read_text())
            latest_tmp = self.run_dir / "latest_model.tmp.zip"
            self.model.save(str(latest_tmp))
            _replace(latest_tmp, self.run_dir / "latest_model.zip")
            old = sorted(ck.glob("step_*.zip"))[:-self.keep] if self.keep > 0 else []
            for p in old:
                p.unlink(missing_ok=True)
                p.with_name(p.name.replace(".zip", ".meta.json")).unlink(missing_ok=True)
        except OSError as e:
            print(f"[TimedCheckpoint] save failed (training continues): {e}")


# ------------------------------------------------------------------------------------------------
# exploration watch

class ExplorationWatch(BaseCallback):
    """Logs explore/std (the policy's action noise) every rollout, and warns once if it falls
    below `min_std`: exploration has collapsed and more steps will mostly repeat the same moves.
    (gSDE's noise scale also depends on the state, so its std is logged but never warned about.)"""

    def __init__(self, min_std: float = 0.05, run_dir=None):
        super().__init__()
        self.min_std = float(min_std)
        self.run_dir = Path(run_dir) if run_dir else None
        self.warned = False

    def current_std(self):
        log_std = getattr(self.model.policy, "log_std", None)
        if log_std is None:
            return None
        return float(np.exp(log_std.detach().cpu().numpy()).mean())

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        std = self.current_std()
        if std is None:
            return
        self.logger.record("explore/std", std)
        if self.min_std > 0 and not self.model.use_sde and std < self.min_std and not self.warned:
            self.warned = True
            msg = (f"[WARN] exploration collapsed: action std {std:.3f} < {self.min_std} at "
                   f"{self.num_timesteps} steps. More steps will mostly repeat the same moves; "
                   f"consider --log-std-reset -1 or a higher --ent-coef.")
            print(msg)
            if self.run_dir is not None:
                try:
                    (self.run_dir / "exploration_warning.txt").write_text(msg + "\n")
                except OSError:
                    pass


# ------------------------------------------------------------------------------------------------
# failure-weighted start zones

def zone_weights(means: dict, strength: float) -> list:
    """Probabilities of the 9 start zones from the recent mean passes per zone (keys 1..9).
    Each zone gets 1 + strength x (1 - mean / best mean): the best zone weight 1, a zone with no
    passes 1 + strength, a zone with no data yet 1 + strength (so it gets tried). Every zone keeps
    a share, so a solved zone is not forgotten."""
    known = [float(v) for v in means.values() if v is not None]
    best = max(known) if known else 0.0
    w = []
    for z in range(1, N_ZONES + 1):
        m = means.get(z)
        if m is None or best <= 0:
            w.append(1.0 + strength if m is None else 1.0)
        else:
            w.append(1.0 + strength * (1.0 - min(float(m), best) / best))
    if best <= 0 and all(means.get(z) is not None for z in range(1, N_ZONES + 1)):
        w = [1.0] * N_ZONES                  # nothing passes anywhere yet: plain round-robin share
    w = np.asarray(w, dtype=float)
    return (w / w.sum()).tolist()


def _start_zone(info) -> int:
    try:
        from telemetry_callback import _start_zone as sz
        return sz(info)
    except ImportError:                                  # pragma: no cover
        return int(info.get("start_zone") or 0)


class ZoneFocus(BaseCallback):
    """Training only: every `every` finished episodes, re-weight the start zones by how badly each
    has gone over its last `window` episodes (zone_weights) and hand the weights to every training
    env (set_zone_weights). Evaluation keeps its fixed round of all 9 zones."""

    def __init__(self, strength: float = 2.0, window: int = 30, every: int = 45):
        super().__init__()
        self.strength, self.every = float(strength), int(every)
        self.hist = {z: deque(maxlen=int(window)) for z in range(1, N_ZONES + 1)}
        self._since = 0
        self.weights = None

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
            z = _start_zone(info)
            if z in self.hist:
                self.hist[z].append(float(info.get("passes", 0)))
                self._since += 1
        if self._since >= self.every:
            self._since = 0
            means = {z: (float(np.mean(h)) if len(h) else None) for z, h in self.hist.items()}
            self.weights = zone_weights(means, self.strength)
            self.training_env.env_method("set_zone_weights", self.weights)
            for z, w in enumerate(self.weights, 1):
                self.logger.record(f"zones/weight_{z}", w)
        return True


# ------------------------------------------------------------------------------------------------
# PyBullet budget ramp

class BudgetRamp(BaseCallback):
    """PyBullet --reward budget: the training envs' budget goes from `start` to `end` over
    `steps` timesteps (counted from this run's own start). A parking policy burns the whole budget
    each episode, so a small budget early means several times more episodes and fresh starts from
    the same steps; evaluation always uses the full budget."""

    def __init__(self, start: float, end: float, steps: int, every: int = 5000):
        super().__init__()
        self.start, self.end, self.steps, self.every = float(start), float(end), int(steps), int(every)
        self._t0 = None
        self._last = None

    def value(self, t: int) -> float:
        k = min(1.0, max(0.0, t / max(1, self.steps)))
        return round(self.start + (self.end - self.start) * k, 3)

    def _on_training_start(self):
        self._t0 = int(self.num_timesteps)
        self._apply(self.value(0))

    def _apply(self, b):
        self.training_env.env_method("set_budget", b)
        self._last = b
        self.logger.record("curriculum/budget", b)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every < self.training_env.num_envs:
            b = self.value(self.num_timesteps - self._t0)
            if b != self._last:
                self._apply(b)
        return True


# ------------------------------------------------------------------------------------------------
# reward normalisation

def wrap_reward_norm(train_env, eval_env, gamma: float):
    """Scale training rewards by a running estimate of the return's spread (VecNormalize with
    norm_obs=False). Only the critic's targets change; the observations, the saved policy and what
    flies on the drone do not, so nothing has to be carried to replays or the Tello. The eval env
    gets a pass-through VecNormalize so SB3 can sync the two and still report raw rewards."""
    from stable_baselines3.common.vec_env import VecNormalize
    train = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=float(gamma))
    ev = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    return train, ev


def speedup_meta(**kw) -> dict:
    """What a run used, for meta.json (JSON-safe)."""
    return json.loads(json.dumps(kw, default=str))
