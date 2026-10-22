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


def convert_old_yaw(model) -> bool:
    """Negate the first-layer input weights of the yaw-convention columns (spec.yaw_columns), in
    the actor and the critic, so a model trained before the clockwise convention acts on
    today's observations exactly as it did on its own. Returns True if anything changed."""
    import torch
    width = int(model.observation_space.shape[0])
    if width == spec.TRACKER_OBS_DIM:
        # a 37-input model just widened for the ring tracker: the base columns are where they
        # were, and the tracker's own inputs still have zero weights
        cols = spec.yaw_columns(spec.OBS_DIM)
    elif width not in spec.LAYOUT_FIELDS:
        return False
    else:
        cols = spec.yaw_columns(width)
    ext = getattr(model.policy, "mlp_extractor", None)
    changed = False
    with torch.no_grad():
        for net_name in ("policy_net", "value_net"):
            net = getattr(ext, net_name, None)
            first = next((m for m in net.modules() if isinstance(m, torch.nn.Linear)), None) if net else None
            if first is not None and first.in_features == width:
                first.weight[:, cols] *= -1
                changed = True
    return changed


def load_policy(model_path, algo: Optional[str] = None, deterministic: bool = True
                ) -> Callable[[np.ndarray], np.ndarray]:
    """-> a callable obs(OBS_DIM,) -> action(4,) in [-1, 1].

    `algo` comes from meta.json when present, else must be given. Raises ValueError if
    the model's spaces do not match this task.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"no model at {model_path}")
    if model_path.suffix == ".json":
        # a fixed-switch hierarchy (hrl.write_phase_bundle): skills only, no manager model
        meta = json.loads(model_path.read_text(encoding="utf-8"))
        if not meta.get("hrl"):
            raise ValueError(f"{model_path.name} is not a hierarchy bundle (no 'hrl' in it)")
        from .hrl import HierarchicalPolicy
        return HierarchicalPolicy(meta, None, deterministic)
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
    if meta.get("hrl"):
        # a learned hierarchy (train_hrl.py): this zip is the MANAGER; meta.json names its skills
        from .hrl import HierarchicalPolicy
        return HierarchicalPolicy(meta, cls.load(str(model_path), device="cpu"), deterministic)
    custom = None
    if meta.get("asym_critic"):
        # the policy class is pickled by reference; hand it over explicitly so a model loads
        # whichever way this package was imported (onboard.* / hardware.onboard.*)
        from .aac_policy import MaskedActorCriticPolicy
        custom = {"policy_class": MaskedActorCriticPolicy}
    model = cls.load(str(model_path), device="cpu", custom_objects=custom)

    if meta.get("obs_history") or meta.get("asym_critic"):
        return _history_policy(model, meta, deterministic, model_path)

    got = tuple(model.observation_space.shape)
    # a policy from before the newer fields reads the same observation minus those columns
    legacy = len(got) == 1 and got[0] != spec.OBS_DIM and got[0] in spec.LAYOUT_FIELDS
    keep = spec.legacy_columns(got[0]) if legacy else None
    tracker = got == (spec.TRACKER_OBS_DIM,)
    if got != (spec.OBS_DIM,) and not legacy and not tracker:
        raise ValueError(
            f"{model_path.name} expects observations of shape {got}, but this task "
            f"produces {(spec.OBS_DIM,)}. A shape of (1, 75) means it was trained on the "
            "simulator's HoopFindAviary task; use hardware/control/run_policy.py for those.")
    if tuple(model.action_space.shape) != (spec.ACT_DIM,):
        raise ValueError(f"{model_path.name} has action shape "
                         f"{tuple(model.action_space.shape)}, expected {(spec.ACT_DIM,)}")

    # only once the spaces are known to be this task's (a 75-input sim model has no yaw columns)
    yaw_converted = spec.old_yaw_convention(meta) and convert_old_yaw(model)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape[0] == spec.TRACKER_OBS_DIM and not tracker:
            o = o[:spec.OBS_DIM]               # the tracker's fields are appended at the end
        if tracker and o.shape[0] != spec.TRACKER_OBS_DIM:
            raise ValueError(f"this policy uses the ring tracker: it needs {spec.TRACKER_OBS_DIM} "
                             f"observation inputs, got {o.shape[0]} (build them with "
                             f"ObsBuilder(width=spec.TRACKER_OBS_DIM))")
        if legacy and o.shape[0] == spec.OBS_DIM:
            o = o[keep]
        a, _ = model.predict(o, deterministic=deterministic)
        return np.clip(np.asarray(a, dtype=float).reshape(-1)[:spec.ACT_DIM], -1.0, 1.0)

    policy_fn.model = model          # kept for tests and for introspection
    policy_fn.meta = meta
    policy_fn.yaw_converted = bool(yaw_converted)
    policy_fn.obs_width = spec.TRACKER_OBS_DIM if tracker else spec.OBS_DIM   # what to build
    return policy_fn


class _HistoryPolicy:
    """A v6 model (onboard/history.py) as an obs(frame) -> action callable. Keeps the same
    HistoryStack the simulator used, and pads the critic-only privileged inputs with zeros (the
    actor never reads them: aac_policy.MaskExtractor). Call reset() when the drone takes control
    (fly_hoop does, after the hover): the stack then starts from the current frame."""

    def __init__(self, model, meta, deterministic):
        from .history import PRIV_DIM, HistoryStack, stacked_width
        self.model, self.meta, self.deterministic = model, meta, deterministic
        self.obs_width = int(meta.get("frame_dim") or (spec.TRACKER_OBS_DIM if meta.get("ring_tracker")
                                                       else spec.OBS_DIM))
        lags = meta.get("obs_history") or [0]
        self.stack = HistoryStack(self.obs_width, lags)
        self.pad = np.zeros(PRIV_DIM if meta.get("asym_critic") else 0, np.float32)
        self.width = stacked_width(self.obs_width, lags, bool(meta.get("asym_critic")))
        self.yaw_converted = False
        got = tuple(model.observation_space.shape)
        if got != (self.width,):
            raise ValueError(f"model expects {got} inputs but its meta.json describes {self.width} "
                             f"(frame {self.obs_width}, history {lags}, privileged {len(self.pad)})")
        if tuple(model.action_space.shape) != (spec.ACT_DIM,):
            raise ValueError(f"model has action shape {tuple(model.action_space.shape)}, "
                             f"expected {(spec.ACT_DIM,)}")

    def reset(self):
        self.stack.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape[0] != self.obs_width:
            raise ValueError(f"this policy needs {self.obs_width} observation inputs per frame, got "
                             f"{o.shape[0]} (build them with ObsBuilder(width={self.obs_width}))")
        x = np.concatenate([self.stack.push(o), self.pad])
        a, _ = self.model.predict(x, deterministic=self.deterministic)
        return np.clip(np.asarray(a, dtype=float).reshape(-1)[:spec.ACT_DIM], -1.0, 1.0)


def _history_policy(model, meta, deterministic, model_path):
    try:
        return _HistoryPolicy(model, meta, deterministic)
    except ValueError as e:
        raise ValueError(f"{Path(model_path).name}: {e}") from e
