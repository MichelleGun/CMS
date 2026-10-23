"""
Hierarchical RL for the loop task: SKILLS (low level, 15 Hz) and a MANAGER (high level) that picks
which skill flies next. This is the options framework (Sutton, Precup & Singh 1999, "Between MDPs and
semi-MDPs: a framework for temporal abstraction in reinforcement learning"):

  * an option = a skill policy run for up to OPTION_TICKS ticks (8 = 0.53 s), or until a pass;
  * the manager is a policy over options: its MDP is a semi-MDP whose step is one option and whose
    reward is the sum of the environment's rewards while the option ran (so its discount per
    decision is about gamma_tick ** 8);
  * the skills are frozen while the manager learns.

Why: the task has phases (find and enter the hoop; go round and find it again), and one flat policy
traded them off: training for loops made slanted (corner) entries worse (67 % -> 57 %), and a corner
reward made loops worse. Skills can each be trained for their own phase (week10_experiments SE
"enter", SL "loop"); the manager learns when to use which.

Two managers:
  * PhaseSwitch: the baseline. The enter skill until the first pass, then the loop skill. "A pass has
    happened" comes from the observation's `streak` field (onboard/obs.py PassDetector: telemetry
    only), so the rule works on the real drone too;
  * learned: an SB3 PPO policy over the skills (train_hrl.py).

Both fly through policy.load_policy -> HierarchicalPolicy, so fly_hoop.py, export_policy.py and the
optimiser's referee work unchanged. An option ends early when `streak` changes (a pass), the same rule
in the simulator (OptionEnv) and on the drone (HierarchicalPolicy), so both decide at the same ticks.
"""
from pathlib import Path
from typing import List, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import spec
from .history import DEFAULT_LAGS, PRIV_DIM, HistoryStack

OPTION_TICKS = 8
STREAK_IDX = spec.OBS_SLICES["streak"].start          # in a frame; the newest frame comes first
CMS = Path(__file__).resolve().parents[2]


def env_kwargs(**over) -> dict:
    """The skills' environment (the memory policies': as n8 / C9 and privileged.dagger_hist)."""
    kw = dict(profile="measured", laps=0, lap_dir="same", start_mode="zones", sway_deg=2.0, wash_deg=15.0,
              dr=1.0, ring_tracker=True, history_lags=tuple(DEFAULT_LAGS), privileged=True)
    kw.update(over)
    return kw


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else CMS / p


def load_skill(path):
    """A skill model (a memory policy: BCRegularizedPPO with the masked asymmetric policy)."""
    from .aac_policy import MaskedActorCriticPolicy
    from stable_baselines3 import PPO
    p = resolve(path)
    if p.is_dir():
        p = next((p / n for n in ("best_model.zip", "final_model.zip", "latest_model.zip") if (p / n).is_file()), p)
    return PPO.load(str(p), device="cpu", custom_objects={"policy_class": MaskedActorCriticPolicy})


def skill_act(model, full_obs) -> np.ndarray:
    a, _ = model.predict(np.asarray(full_obs, np.float32), deterministic=True)
    return np.clip(np.asarray(a, dtype=float).reshape(-1)[:spec.ACT_DIM], -1.0, 1.0)


def passed_count(frame) -> float:
    return float(np.asarray(frame).reshape(-1)[STREAK_IDX])


def phase_choice(frame, enter=0, loop=1) -> int:
    """The baseline manager: the enter skill until the first pass, then the loop skill."""
    return loop if passed_count(frame) > 0 else enter


def one_hot(i: Optional[int], k: int) -> np.ndarray:
    v = np.zeros(k, np.float32)
    if i is not None:
        v[int(i)] = 1.0
    return v


class OptionEnv(gym.Env):
    """The manager's semi-MDP: action = which skill flies next; that skill flies up to OPTION_TICKS
    ticks of the laps task (or until a pass); reward = the sum of the env's rewards over them.
    Observation = the actor's inputs (newest frame + history, no privileged inputs) + a one-hot of
    the skill that flew last. Skills are frozen (loaded once per process, lazily)."""

    metadata = {"render_modes": []}

    def __init__(self, skills: Sequence[str], option_ticks: int = OPTION_TICKS, seed: int = 0,
                 env_kw: Optional[dict] = None, phase: bool = False):
        from .gym_env import OnboardHoopEnv
        self.skill_paths = [str(s) for s in skills]
        self.k = len(self.skill_paths)
        self.option_ticks = int(option_ticks)
        self.phase = bool(phase)
        self.env = OnboardHoopEnv(seed=seed, **(env_kw or env_kwargs()))
        self.frame_dim = self.env.obs_dim
        self.actor_dim = self.env.actor_dim
        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.actor_dim + self.k,), dtype=np.float32)
        self._skills = None
        self._full = None
        self._prev = None
        self.usage = np.zeros((2, self.k), np.int64)      # [before / after the first pass, skill] ticks

    def skills(self):
        if self._skills is None:
            self._skills = [load_skill(p) for p in self.skill_paths]
        return self._skills

    def _mobs(self):
        return np.concatenate([self._full[:self.actor_dim], one_hot(self._prev, self.k)]).astype(np.float32)

    def set_episode_seed(self, seed):
        self.env.set_episode_seed(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._full, info = self.env.reset(seed=seed)
        self._prev = None
        return self._mobs(), info

    def step(self, action):
        a = phase_choice(self._full[:self.frame_dim]) if self.phase else int(action)
        skill = self.skills()[a]
        total, ticks, info = 0.0, 0, {}
        term = trunc = False
        streak0 = passed_count(self._full[:self.frame_dim])
        for _ in range(self.option_ticks):
            self.usage[int(streak0 > 0), a] += 1
            self._full, r, term, trunc, info = self.env.step(skill_act(skill, self._full))
            total += float(r)
            ticks += 1
            if term or trunc or passed_count(self._full[:self.frame_dim]) != streak0:
                break                                   # a pass (or the end): the manager decides again
        self._prev = a
        info = dict(info, option=a, option_ticks=ticks)
        return self._mobs(), total, term, trunc, info


class HierarchicalPolicy:
    """A hierarchy as the flight's obs(frame) -> action callable (policy.load_policy returns it for a
    meta.json with "hrl"). One HistoryStack (the skills share their history), the critic-only inputs
    padded with zeros, a decision every OPTION_TICKS ticks or when `streak` changes (a pass), exactly
    as OptionEnv decides in the simulator."""

    def __init__(self, meta: dict, manager=None, deterministic: bool = True):
        h = meta["hrl"]
        self.meta = meta
        self.kind = h.get("kind", "learned")
        self.skill_paths = list(h["skills"])
        self.skills = [load_skill(p) for p in self.skill_paths]
        self.k = len(self.skills)
        self.option_ticks = int(h.get("option_ticks", OPTION_TICKS))
        self.manager = manager
        self.deterministic = deterministic
        if self.kind == "learned" and manager is None:
            raise ValueError("a learned hierarchy needs its manager model")
        lags = h.get("obs_history") or list(DEFAULT_LAGS)
        self.obs_width = int(h.get("frame_dim") or spec.TRACKER_OBS_DIM)
        self.stack = HistoryStack(self.obs_width, lags)
        self.pad = np.zeros(PRIV_DIM, np.float32)
        self.actor_dim = self.obs_width * len(lags)
        self.yaw_converted = False
        self.reset()

    def reset(self):
        self.stack.reset()
        self.current, self.left, self.prev, self.streak = None, 0, None, None
        self.choices = []                                 # (tick, skill) for logs / tests

    def choose(self, full, frame) -> int:
        if self.kind == "phase":
            return phase_choice(frame)
        x = np.concatenate([full[:self.actor_dim], one_hot(self.prev, self.k)]).astype(np.float32)
        a, _ = self.manager.predict(x, deterministic=self.deterministic)
        return int(np.asarray(a).reshape(-1)[0])

    def __call__(self, obs) -> np.ndarray:
        frame = np.asarray(obs, dtype=np.float32).reshape(-1)
        if frame.shape[0] != self.obs_width:
            raise ValueError(f"this hierarchy needs {self.obs_width} observation inputs per frame, got "
                             f"{frame.shape[0]}")
        full = np.concatenate([self.stack.push(frame), self.pad])
        s = passed_count(frame)
        if self.left <= 0 or (self.streak is not None and s != self.streak):
            self.prev = self.current
            self.current = self.choose(full, frame)
            self.left = self.option_ticks
            self.streak = s
            self.choices.append((len(self.choices), self.current))
        self.left -= 1
        return skill_act(self.skills[self.current], full)


def hrl_meta(kind: str, skills: List[str], option_ticks: int = OPTION_TICKS) -> dict:
    """The meta.json a hierarchy carries: what policy.load_policy needs to rebuild it, and the
    skills' training conditions (robust) so export_policy scores it like the flat policies."""
    return dict(task=spec.TASK_NAME, algo="ppo", yaw_convention=spec.YAW_CONVENTION, ring_tracker=True,
                profile="measured", robust=dict(dr=1.0, sway_deg=2.0, wash_deg=15.0),
                hrl=dict(kind=kind, skills=[str(s) for s in skills], option_ticks=int(option_ticks),
                         obs_history=list(DEFAULT_LAGS), frame_dim=spec.TRACKER_OBS_DIM))


def write_phase_bundle(path: Path, skills: List[str]) -> Path:
    """A fixed-switch hierarchy as a file the flight / export can load (no manager model)."""
    import json
    meta = dict(hrl_meta("phase", skills),
                note="baseline: enter skill until the first pass, then the loop skill (hrl.PhaseSwitch)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
