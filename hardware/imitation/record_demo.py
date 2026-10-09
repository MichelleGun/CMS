"""
Demonstration recording and loading, in the simulator's own formats so one demo file can train
(or fine-tune) the same policy the sim trains:

    obs      (N, D)   float32   D = 75 for HoopFindAviary; onboard/spec.OBS_DIM for the
                                onboard-camera traversal task
    intent   (N, 4)   float32   what the operator / expert asked for
    executed (N, 4)   float32   what was actually commanded after geofence + speed ramp
    t        (N,)               seconds since the first sample
    episode  (N,)       int     episode index (one flight = one episode)
    meta     JSON string        obs_dim, task, obs_shape, hoop, source, profile, ...

BC trains on `intent` by default: the human wanted that motion, even if the fence trimmed it.

The observation width is a parameter, not a constant, because two different tasks record
through this module: the simulator's 75-number state observation (where `intent` means
[dx, dy, dz, speed_fraction]) and the onboard-camera task's feature vector (where it
means four rc channels). `meta["task"]` and `meta["obs_shape"]` tell the trainer which it
is holding, so it can pick the right loss and build a model with the right spaces.
"""
import json
import time
from pathlib import Path
from typing import List

import numpy as np

HW = Path(__file__).resolve().parents[1]
DEMO_DIR = HW / "flights" / "demos"


SIM_OBS_DIM = 75                 # HoopFindAviary; see common/constants.py OBS_DIM
SIM_TASK = "sim_hoop"


class DemoRecorder:
    def __init__(self, meta: dict = None, episode=0, obs_dim=SIM_OBS_DIM,
                 task=SIM_TASK, obs_shape=None):
        """obs_dim/task/obs_shape describe which task is being recorded.

        obs_shape is the shape a model built for this task expects per sample: (1, 75)
        for the simulator (BaseRLAviary returns a 2-D observation) and (OBS_DIM,) for the
        onboard-camera task. It is stored so the trainer does not have to guess.
        """
        self.obs_dim = int(obs_dim)
        self.meta = dict(meta or {})
        self.meta.setdefault("task", task)
        self.meta.setdefault("obs_dim", self.obs_dim)
        self.meta.setdefault("obs_shape", list(obs_shape) if obs_shape is not None
                             else ([1, self.obs_dim] if self.obs_dim == SIM_OBS_DIM
                                   else [self.obs_dim]))
        self.episode = episode
        self._obs, self._intent, self._exec, self._t, self._ep = [], [], [], [], []
        self._t0 = None

    def add(self, obs, intent, executed, t):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape[0] != self.obs_dim or not np.all(np.isfinite(o)):
            return False
        self._t0 = t if self._t0 is None else self._t0
        self._obs.append(o)
        self._intent.append(np.asarray(intent, dtype=np.float32).reshape(-1)[:4])
        self._exec.append(np.asarray(executed, dtype=np.float32).reshape(-1)[:4])
        self._t.append(t - self._t0)
        self._ep.append(self.episode)
        return True

    def next_episode(self):
        self.episode += 1
        self._t0 = None

    def __len__(self):
        return len(self._obs)

    def save(self, path=None) -> Path:
        if not self._obs:
            raise ValueError("no samples recorded")
        DEMO_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path else DEMO_DIR / f"demo_{time.strftime('%Y%m%d-%H%M%S')}.npz"
        meta = dict(self.meta, n_samples=len(self._obs), saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        np.savez_compressed(path, obs=np.stack(self._obs), intent=np.stack(self._intent),
                            executed=np.stack(self._exec), t=np.array(self._t, dtype=np.float64),
                            episode=np.array(self._ep, dtype=np.int64), meta=json.dumps(meta))
        return path


def load_demo(path) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in ("obs", "intent", "executed", "t", "episode")}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def load_demos(paths: List) -> dict:
    parts = [load_demo(p) for p in paths]
    if not parts:
        raise ValueError("no demo files given")
    out, off = {k: [] for k in ("obs", "intent", "executed", "t", "episode")}, 0
    for p in parts:
        for k in ("obs", "intent", "executed", "t"):
            out[k].append(p[k])
        out["episode"].append(p["episode"] + off)
        off += int(p["episode"].max()) + 1
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["meta"] = [p["meta"] for p in parts]
    # Who demonstrated each sample ("human", "sim_teacher", "onboard_scripted_expert", ...),
    # so a trainer can weight sources differently when they are mixed.
    res["source"] = np.concatenate([np.full(len(p["obs"]), str(p["meta"].get("source", "unknown")))
                                    for p in parts])
    return res


def parse_source_weights(items) -> dict:
    """["human=3", "sim_teacher=1"] -> {"human": 3.0, "sim_teacher": 1.0}."""
    out = {}
    for it in items or []:
        name, sep, val = str(it).partition("=")
        if not sep or not name:
            raise ValueError(f"source weight '{it}' should look like name=number")
        out[name.strip()] = float(val)
    return out


def sample_weights(source: np.ndarray, weights: dict, default=1.0) -> np.ndarray:
    """Per-sample weights, normalised to mean 1 so the loss scale (and learning rate) is
    unchanged by the weighting itself; only the balance between sources moves."""
    w = np.array([weights.get(str(s), default) for s in source], dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("source weights must be >= 0")
    m = float(w.mean()) if len(w) else 1.0
    return w / m if m > 0 else w


def mc_returns(obs: np.ndarray, episode: np.ndarray, gamma=0.99, reach_radius=0.15) -> np.ndarray:
    """Discounted returns of the sim's reward, recomputed from the observation's relative vector.

    reward = -|hoop - pos|, +10 when within reach_radius (HoopFindAviary._computeReward).
    Used only to warm up a critic before PPO fine-tuning; demos carry no stored reward.

    Simulator-only: it reads obs[:, 72:75] as the hoop-minus-position vector, which only
    exists in HoopFindAviary's 75-number observation. The onboard-camera task has no such
    field and a different reward entirely (see onboard/gym_env.py), so this raises rather
    than silently returning returns computed from unrelated columns.
    """
    if obs.shape[1] != SIM_OBS_DIM:
        raise ValueError(
            f"mc_returns is specific to the simulator's {SIM_OBS_DIM}-number observation "
            f"(it reads obs[:, 72:75] as hoop - position); got width {obs.shape[1]}. "
            "For the onboard-camera task, warm the critic from its own env instead.")
    dist = np.linalg.norm(obs[:, 72:75], axis=1)
    r = -dist + 10.0 * (dist < reach_radius)
    G = np.zeros_like(r, dtype=np.float64)
    run = 0.0
    for i in range(len(r) - 1, -1, -1):
        if i == len(r) - 1 or episode[i] != episode[i + 1]:
            run = 0.0
        run = r[i] + gamma * run
        G[i] = run
    return G.astype(np.float32)
