"""
Observation history and the critic-only privileged inputs (2026-10-14, overnight laps runs).

Why. On a same-side lap the drone flies round the OUTSIDE of the ring with the ring out of view,
and the Tello's velocity input is blanked (obs.BLANK_VELOCITY). A memoryless actor sees almost the
same 41 numbers all the way round, so it cannot tell where on the loop it is; the DAgger students
and the PPO laps runs stalled at 0-0.6 laps / 90 s while the privileged teacher flies 4.8
(hardware/privileged/reports/morning.md). Two standard fixes, both kept deployable:

  * HistoryStack: the actor sees the current observation plus the observations `lags` ticks ago
    (default 0, 1, 2, 4, 8, 16: about 1 s of memory at 15 Hz, log-spaced so it stays small). The
    differences between frames carry the velocity the observation no longer has. Frame stacking is
    the memory used by Mnih et al. 2015 (DQN); the SAME class builds the stack in the simulator
    (gym_env) and on the real drone (policy.load_policy), so the two cannot drift apart.

  * privileged critic inputs (asymmetric actor-critic, Pinto et al. 2018, "Asymmetric Actor Critic
    for Image-Based Robot Learning", arXiv:1710.06542): the VALUE network also gets the true state
    (where the drone is in the ring's frame, its velocity, which leg of the lap it is on, time left).
    The value only exists in training, so it may use what the real drone never measures; the actor
    has those inputs masked to zero (aac_policy.MaskedActorCriticPolicy), so at deploy time they are
    simply padded with zeros and the actions are unchanged.

Layout of a history observation:  [obs_t, obs_{t-l1}, ..., obs_{t-lk}, priv (PRIV_DIM, optional)].
The current frame comes first, so a policy trained without history is widened onto it with zero
weights on everything else (vec_util.widen_policy_input, insert index = base width).
"""
from collections import deque
from typing import Optional, Sequence

import numpy as np

DEFAULT_LAGS = (0, 1, 2, 4, 8, 16)

PRIV_FIELDS = (
    "rel_axial", "rel_side", "rel_up",          # drone - ring centre, in the ring's frame (m)
    "vel_axial", "vel_side", "vel_up",          # true velocity in the ring's frame (m/s)
    "fwd_axial", "fwd_side",                    # camera direction in the ring's frame (cos, sin)
    "dir", "armed", "returned", "ret_side",     # lap bookkeeping (gym_env)
    "wp_axial", "wp_side", "wp_up",             # same-side laps: to the way-round point (0 otherwise)
    "time_left", "passes", "rim_dist", "net_dist", "height",
)
PRIV_DIM = len(PRIV_FIELDS)


def parse_lags(text) -> Optional[tuple]:
    """'0,1,2,4,8,16' / [0, 1, ...] / None -> a sorted tuple starting at 0 (or None = no history)."""
    if text is None or text == "" or text == "none":
        return None
    if isinstance(text, str):
        lags = [int(x) for x in text.replace(" ", "").split(",") if x != ""]
    else:
        lags = [int(x) for x in text]
    lags = tuple(sorted(set(lags)))
    if not lags or lags[0] != 0 or any(lag < 0 for lag in lags):
        raise ValueError(f"history lags must start at 0 and be >= 0, got {text!r}")
    return lags


def stacked_width(base_width: int, lags: Optional[Sequence[int]], privileged: bool = False) -> int:
    return base_width * (len(lags) if lags else 1) + (PRIV_DIM if privileged else 0)


class HistoryStack:
    """Keeps the last max(lags) + 1 observations and returns them stacked, newest first.

    reset() empties it; the first push after a reset fills the whole history with that frame
    (the drone was not anywhere else before the episode started)."""

    def __init__(self, base_width: int, lags: Sequence[int] = DEFAULT_LAGS):
        self.base_width = int(base_width)
        self.lags = tuple(int(x) for x in lags)
        self._buf = deque(maxlen=max(self.lags) + 1)

    @property
    def width(self) -> int:
        return self.base_width * len(self.lags)

    def reset(self):
        self._buf.clear()

    def push(self, obs) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape[0] != self.base_width:
            raise ValueError(f"history expects {self.base_width}-wide observations, got {o.shape[0]}")
        if not self._buf:
            for _ in range(self._buf.maxlen):
                self._buf.append(o)
        else:
            self._buf.append(o)
        n = len(self._buf)
        return np.concatenate([self._buf[n - 1 - lag] for lag in self.lags]).astype(np.float32)


def privileged_features(env) -> np.ndarray:
    """PRIV_DIM floats of true state from an OnboardHoopEnv (critic only; see the module docstring)."""
    from .sim import CAGE_XY
    sim = env.sim
    c, n = sim.ring.centre, sim.ring.normal
    u = env._side_axis()
    z = np.array([0.0, 0.0, 1.0])

    def frame(v):
        return [float(np.dot(v, n)), float(np.dot(v, u)), float(np.dot(v, z))]

    rel = frame(sim.pos - c)
    vel = frame(np.asarray(sim.vel, dtype=float))
    fwd = np.asarray(sim.forward, dtype=float)
    if env.lap_dir == "same" and not env._returned:
        wp = frame(env._return_wp() - sim.pos)
    else:
        wp = [0.0, 0.0, 0.0]
    time_left = max(0, env.max_ticks - env._ticks) / max(1, env.max_ticks)
    p = sim.pos
    f = [*rel, *vel, float(np.dot(fwd, n)), float(np.dot(fwd, u)),
         float(env._dir), float(env._armed), float(env._returned), float(env._ret_side),
         *wp, time_left, min(env.passes, 10) / 10.0,
         float(np.clip(sim.rim_distance(), 0.0, 1.0)),
         float(CAGE_XY - max(abs(p[0]), abs(p[1]))), float(p[2])]
    return np.asarray(f, dtype=np.float32)
