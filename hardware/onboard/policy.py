"""
Load a trained policy for the onboard-camera task and expose it as a plain callable.

The controller slot in fly_hoop.py takes any `obs -> action` callable, so a scripted
Expert, a behaviour clone and an RL policy are interchangeable there. This module is the
adapter for the two learned cases.

The shape check is not a formality. A model trained on the simulator's task has a
75-number observation and an action meaning [dx, dy, dz, speed_fraction]; this task has
spec.OBS_DIM and four rc channels. Loading the wrong one would not raise anywhere -- it
would fly, badly, on four numbers that mean something else. So the spaces are verified
before the model is ever called, and the error says which task the file belongs to.
"""
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import spec

ALGOS = ("ppo", "a2c", "sac", "td3")


def _load_meta(model_path: Path) -> dict:
    for name in ("meta.json", model_path.stem + ".meta.json"):
        p = model_path.parent / name
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
    return {}


def load_policy(model_path, algo: Optional[str] = None, deterministic: bool = True
                ) -> Callable[[np.ndarray], np.ndarray]:
    """-> a callable obs(OBS_DIM,) -> action(4,) in [-1, 1].

    `algo` comes from meta.json when present, else must be given. Raises ValueError if
    the model's spaces do not match this task.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"no model at {model_path}")
    meta = _load_meta(model_path)
    algo = (algo or meta.get("algo") or "ppo").lower()
    if algo not in ALGOS:
        raise ValueError(f"unknown algo '{algo}'; expected one of {ALGOS}")
    task = meta.get("task")
    if task is not None and task != spec.TASK_NAME:
        raise ValueError(
            f"{model_path.name} was trained for task '{task}', not '{spec.TASK_NAME}'. "
            "A simulator-task model cannot fly this one: its action means a direction "
            "plus a speed fraction, not four rc channels.")

    from stable_baselines3 import A2C, PPO, SAC, TD3
    cls = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3}[algo]
    model = cls.load(str(model_path), device="cpu")

    got = tuple(model.observation_space.shape)
    if got != (spec.OBS_DIM,):
        raise ValueError(
            f"{model_path.name} expects observations of shape {got}, but this task "
            f"produces {(spec.OBS_DIM,)}. A shape of (1, 75) means it was trained on the "
            "simulator's HoopFindAviary task; use hardware/control/run_policy.py for those.")
    if tuple(model.action_space.shape) != (spec.ACT_DIM,):
        raise ValueError(f"{model_path.name} has action shape "
                         f"{tuple(model.action_space.shape)}, expected {(spec.ACT_DIM,)}")

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        a, _ = model.predict(np.asarray(obs, dtype=np.float32).reshape(-1),
                             deterministic=deterministic)
        return np.clip(np.asarray(a, dtype=float).reshape(-1)[:spec.ACT_DIM], -1.0, 1.0)

    policy_fn.model = model          # kept for tests and for introspection
    policy_fn.meta = meta
    return policy_fn
