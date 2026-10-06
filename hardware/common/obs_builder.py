"""
Builds the simulator's observation from a fused hardware state.

Layout (HoopFindAviary, ObservationType.KIN, ActionType.VEL, ctrl 30 Hz), shape (1, 75):
    [ pos(3), rpy(3), vel(3), angvel(3),          12  world frame, rad, m/s, rad/s
      action_buffer(15 x 4, oldest first),        60  last 0.5 s of *executed* actions
      hoop - pos (3) ]                             3
No normalisation (the sim training used none). sim_check/obs_parity.py checks this
against the real environment.
"""
from collections import deque

import numpy as np

from .constants import ACT_BUFFER, ACT_DIM, CTRL_DT, OBS_DIM
from .smoothing import wrap_pi


class ObsBuilder:
    def __init__(self, hoop_xyz, lpf_alpha=0.35):
        self.hoop = np.asarray(hoop_xyz, dtype=float).reshape(3)
        self.alpha = lpf_alpha
        self.reset()

    def reset(self):
        """Clear history. Hardware starts every flight with a zero action buffer."""
        self.buffer = deque([np.zeros(ACT_DIM) for _ in range(ACT_BUFFER)], maxlen=ACT_BUFFER)
        self._prev_rpy = None
        self._prev_t = None
        self._angvel = np.zeros(3)

    def push_action(self, executed_action):
        """Record the action that was actually commanded (after geofence / manual override)."""
        a = np.asarray(executed_action, dtype=float).reshape(-1)[:ACT_DIM]
        self.buffer.append(np.nan_to_num(a))

    def angular_velocity(self, rpy, t):
        """Low-passed d(rpy)/dt (small-angle approximation of the world-frame rate)."""
        rpy = np.asarray(rpy, dtype=float)
        if self._prev_rpy is not None and t > self._prev_t:
            dt = t - self._prev_t
            d = np.array([wrap_pi(a - b) for a, b in zip(rpy, self._prev_rpy)]) / dt
            self._angvel = self._angvel + self.alpha * (d - self._angvel)
        self._prev_rpy, self._prev_t = rpy.copy(), t
        return self._angvel.copy()

    def build(self, pos, rpy, vel, t, angvel=None) -> np.ndarray:
        pos = np.asarray(pos, dtype=float)
        rpy = np.asarray(rpy, dtype=float)
        vel = np.asarray(vel, dtype=float)
        if angvel is None:
            angvel = self.angular_velocity(rpy, t)
        kin = np.hstack([pos, rpy, vel, np.asarray(angvel, dtype=float)])
        buf = np.hstack(list(self.buffer))
        rel = self.hoop - pos
        obs = np.hstack([kin, buf, rel]).astype(np.float32).reshape(1, OBS_DIM)
        return obs
