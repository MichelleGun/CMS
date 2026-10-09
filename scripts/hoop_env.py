"""
RL environment for the 2x2x2 m hardware-demo cage: find a hoop that could be
anywhere in the room.

Unlike HoverAviary (fixed target at [0, 0, 1], same every episode), here the
hoop position AND the drone's starting position are resampled every reset,
uniformly within the room. The relative vector to the hoop is appended to the
observation -- without it the task is a POMDP (the policy has no way to know
where this episode's hoop is) and can't be solved.

Action space defaults to 'vel' (bounded velocity setpoints), not 'one_d_rpm'
or 'pid':
- one_d_rpm applies the same RPM to all four motors, so it can only move
  vertically -- it can never steer sideways to reach a hoop that isn't
  directly overhead (see PROGRESS.md).
- 'pid' first looked like the right fit (its action is a position destination,
  and BaseRLAviary bounds that to [-1, 1] m per axis -- almost exactly this
  room). It trains badly here, though: the room is small enough that the
  destination is very often *more than the controller's 1 m per-step cap away*
  from the current position, so a policy that hasn't learned smooth motion yet
  can command a ~1 m instantaneous setpoint jump every 33 ms control step.
  DSLPIDControl responds with an aggressive tilt to chase it, and early in
  training that reliably tips the drone past the tilt-truncation threshold in
  under 10 control steps, before it has moved almost at all -- the policy
  never lives long enough to learn anything about reaching the hoop, only to
  "fall over slightly slower." 'vel' instead asks for a direction and a
  fraction of `SPEED_LIMIT` (~0.25 m/s for the CF2X), which is inherently
  smooth and physically achievable, and trains stably.
"""
import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

ROOM = np.array([2.0, 2.0, 2.0])   # x, y, z (m) -- the hardware-demo cage, floor at z=0
XY_MARGIN = 0.25                   # keep hoop/start this far from the side walls
Z_RANGE = (0.3, 1.6)               # rim height range -- most of the room, clear of floor/ceiling
START_Z = 0.1
REACH_RADIUS = 0.15                # "found the hoop" distance (m)


class HoopFindAviary(BaseRLAviary):
    """Single-agent RL problem: fly to a hoop whose position is randomised every episode."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 room=ROOM,
                 seed=None,
                 start_pos=None,
                 target_pos=None):
        """start_pos / target_pos: optional (3,) array-like overrides. When given, that
        position is used every reset instead of being randomly resampled, so a dashboard
        (or a script) can pin the drone's start and/or the hoop's position while still
        letting the other one randomise. `None` (the default for both) reproduces the
        original random-every-episode behaviour exactly."""
        self.ROOM = np.array(room, dtype=float)
        self.EPISODE_LEN_SEC = 15   # SPEED_LIMIT (~0.25 m/s) is conservative; give it time to cross the room
        self._rng = np.random.default_rng(seed)
        self._start_override = None if start_pos is None else np.asarray(start_pos, dtype=float)
        self._target_override = None if target_pos is None else np.asarray(target_pos, dtype=float)
        self.TARGET_POS = (self._target_override if self._target_override is not None
                            else self._sample_target(self._rng))
        start = (self._start_override if self._start_override is not None
                 else self._sample_start(self._rng))
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=start.reshape(1, 3),
                         initial_rpys=None,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         )

    ################################################################################
    # randomisation

    def _sample_target(self, rng):
        half_xy = self.ROOM[:2] / 2 - XY_MARGIN
        xy = rng.uniform(-half_xy, half_xy)
        z = rng.uniform(*Z_RANGE)
        return np.array([xy[0], xy[1], z])

    def _sample_start(self, rng):
        half_xy = self.ROOM[:2] / 2 - XY_MARGIN
        xy = rng.uniform(-half_xy, half_xy)
        return np.array([xy[0], xy[1], START_Z])

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        self._rng = rng
        self.TARGET_POS = (self._target_override if self._target_override is not None
                            else self._sample_target(rng))
        start = (self._start_override if self._start_override is not None
                 else self._sample_start(rng))
        self.INIT_XYZS = start.reshape(1, 3)
        return super().reset(seed=seed, options=options)

    def onboard_frame(self, res=(200, 150)):
        """The drone's camera frame (uint8 RGB), callable through VecEnv.env_method (works in a
        SubprocVecEnv worker too)."""
        from camera_capture import get_onboard_frame
        return get_onboard_frame(self, nth_drone=0, res=res)

    ################################################################################
    # observation: parent's kinematic vector + relative vector to the hoop

    def _observationSpace(self):
        base = super()._observationSpace()
        room_tile = np.tile(self.ROOM, (self.NUM_DRONES, 1))
        lo = np.hstack([base.low, -room_tile])
        hi = np.hstack([base.high, room_tile])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        base = super()._computeObs()
        pos = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        rel = self.TARGET_POS - pos
        return np.hstack([base, rel]).astype(np.float32)

    ################################################################################
    # reward / termination

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        reward = -dist
        if dist < REACH_RADIUS:
            reward += 10.0
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        dist = np.linalg.norm(self.TARGET_POS - state[0:3])
        return dist < REACH_RADIUS

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        half_xy = self.ROOM[:2] / 2 + 0.3       # slack past the walls before calling it a fly-away
        if (abs(state[0]) > half_xy[0] or abs(state[1]) > half_xy[1]
                or state[2] > self.ROOM[2] + 0.3
                or abs(state[7]) > .5 or abs(state[8]) > .5):   # too tilted (~29 deg -- real banking, not falling over)
            return True
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {"answer": 42}
