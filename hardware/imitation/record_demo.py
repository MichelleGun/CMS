"""
Demonstration recording and loading, in the simulator's own formats so one demo file can train
(or fine-tune) the same policy the sim trains:

    obs      (N, 75)  float32   exactly HoopFindAviary's observation
    intent   (N, 4)   float32   what the operator / expert asked for: [dx, dy, dz, speed_fraction]
    executed (N, 4)   float32   what was actually commanded after geofence + speed ramp
    t        (N,)               seconds since the first sample
    episode  (N,)       int     episode index (one flight = one episode)
    meta     JSON string        hoop, source, profile, speed_scale, reached, ...

BC trains on `intent` by default: the human wanted that motion, even if the fence trimmed it.
"""
import json
import time
from pathlib import Path
from typing import List

import numpy as np

HW = Path(__file__).resolve().parents[1]
DEMO_DIR = HW / "flights" / "demos"


class DemoRecorder:
    def __init__(self, meta: dict = None, episode=0):
        self.meta = dict(meta or {})
        self.episode = episode
        self._obs, self._intent, self._exec, self._t, self._ep = [], [], [], [], []
        self._t0 = None

    def add(self, obs, intent, executed, t):
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape[0] != 75 or not np.all(np.isfinite(o)):
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
    return res


def mc_returns(obs: np.ndarray, episode: np.ndarray, gamma=0.99, reach_radius=0.15) -> np.ndarray:
    """Discounted returns of the sim's reward, recomputed from the observation's relative vector.

    reward = -|hoop - pos|, +10 when within reach_radius (HoopFindAviary._computeReward).
    Used only to warm up a critic before PPO fine-tuning; demos carry no stored reward.
    """
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
