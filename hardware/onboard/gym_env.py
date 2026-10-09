"""
Gymnasium environment over sim.py, so the traversal task can be trained by RL.

The reward is shaped for PASSING THROUGH a ring, which is deliberately not what the
project's PyBullet task does. scripts/hoop_env.py rewards -distance to a point and
terminates the moment the drone is within 0.15 m of it, so a policy trained there is
rewarded for stopping at the centre and never learns an approach direction. Here the
geometry is decomposed into the two quantities a traversal actually depends on:

    axial  = signed distance along the ring's normal  (how far in front/behind)
    radial = distance from the ring's axis            (how far off-centre)

and progress is rewarded on both. Reducing `radial` lines the drone up with the hole;
reducing |axial| flies it down the axis. That pair is what makes the approach direction
matter, which is the whole difference between reaching a point and going through a ring.

The episode ends when the drone crosses the ring's plane, and the terminal reward says
which of the three things happened: through the opening, into the noodle, or past the
ring entirely.
"""
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:                                  # pragma: no cover
    raise ImportError("gymnasium is required for gym_env (pip install gymnasium)") from e

from . import spec
from .obs import ObsBuilder
from .ring_detect import DetectorParams
from .sim import OnboardSim, Ring

# Reward weights. Progress terms are per metre, so they are comparable to each other and
# to the terminal bonuses.
W_AXIAL = 3.0
W_RADIAL = 2.5
W_TIME = 0.02
R_PASSED = 20.0
R_STRUCK = -10.0
R_MISSED = -5.0
R_OUT_OF_CAGE = -10.0

MAX_EPISODE_SEC = 30.0


class OnboardHoopEnv(gym.Env):
    """Observation: spec.OBS_DIM floats. Action: 4 floats in [-1, 1] = [lr, fb, ud, yaw].

    The observation is built by the SAME obs.ObsBuilder the real flight uses, from the
    SAME ring_detect output shape, so a policy trained here can be loaded by policy.py
    and flown without any adaptation layer. That is the point of spec.py.
    """

    metadata = {"render_modes": []}

    def __init__(self, profile="typical", seed: int = 0, render_frames: bool = False,
                 ring: Optional[Ring] = None, detector: Optional[DetectorParams] = None,
                 max_episode_sec: float = MAX_EPISODE_SEC,
                 drone_width_m: float = spec.DRONE_WIDTH_M):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(spec.OBS_DIM,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(spec.ACT_DIM,), dtype=np.float32)
        self._kwargs = dict(profile=profile, render=render_frames, ring=ring,
                            detector=detector, drone_width_m=drone_width_m)
        self._seed = int(seed)
        self.max_ticks = int(max_episode_sec * spec.CTRL_HZ)
        self.sim: Optional[OnboardSim] = None
        self.builder = ObsBuilder()
        self._ticks = 0

    # -- geometry ---------------------------------------------------------------------
    def _axial_radial(self):
        d = self.sim.pos - self.sim.ring.centre
        axial = float(np.dot(d, self.sim.ring.normal))
        radial = float(np.linalg.norm(d - axial * self.sim.ring.normal))
        return axial, radial

    # -- gym api ----------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        s = self._seed if seed is None else int(seed)
        self._seed += 1                                  # fresh layout each episode
        if self.sim is None:
            self.sim = OnboardSim(seed=s, **self._kwargs)
        ring_obs = self.sim.reset(seed=s)
        self.builder.reset()
        self._ticks = 0
        self._axial, self._radial = self._axial_radial()
        obs = self.builder.build(ring_obs, self.sim.state(), self.sim.t)
        return obs, {}

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(-1)[:spec.ACT_DIM]
        ring_obs = self.sim.step(a)
        self.builder.push_action(a)
        obs = self.builder.build(ring_obs, self.sim.state(), self.sim.t)
        self._ticks += 1

        axial, radial = self._axial_radial()
        reward = (W_AXIAL * (abs(self._axial) - abs(axial))
                  + W_RADIAL * (self._radial - radial)
                  - W_TIME)
        self._axial, self._radial = axial, radial

        terminated = False
        info = {"phase": "flying"}
        o = self.sim.outcome
        if o.crossed:
            terminated = True
            if o.passed:
                reward += R_PASSED
                info["phase"] = "passed"
            elif o.struck:
                reward += R_STRUCK
                info["phase"] = "struck"
            else:
                reward += R_MISSED
                info["phase"] = "missed"
            info["radial_m"] = o.radial_m
        elif self.sim.out_of_cage():
            terminated = True
            reward += R_OUT_OF_CAGE
            info["phase"] = "out_of_cage"

        truncated = (not terminated) and self._ticks >= self.max_ticks
        if terminated or truncated:
            # SB3's EvalCallback reads this at episode end: the dashboard's success rate.
            info["is_success"] = info["phase"] == "passed"
        info["ring_found"] = bool(ring_obs.found)
        info["range_m"] = float(ring_obs.range_m)
        return obs, float(reward), terminated, truncated, info

    def set_episode_seed(self, seed: int):
        """Seed of the next reset() that is not given one (callable through VecEnv.env_method,
        which, unlike set_attr, reaches the env through its Monitor wrapper)."""
        self._seed = int(seed)

    def close(self):
        self.sim = None


def rollout(policy_fn, episodes: int = 20, profile="typical", seed: int = 100,
            render_frames: bool = False, max_episode_sec: float = MAX_EPISODE_SEC,
            drone_width_m: float = spec.DRONE_WIDTH_M):
    """Score any callable obs -> action. Used by the BC trainer and the tests.

    Returns pass/strike/miss counts rather than only a mean reward, because for this task
    "did it go through" is the number that matters and a mean reward hides it.

    If `policy_fn` has a .reset(), it is called at the start of each episode. A stateful
    controller (the scripted Expert runs a phase machine, and ends an attempt in DONE or
    ABORT) would otherwise carry its terminal state into every later episode and score
    zero from the second one onward.
    """
    env = OnboardHoopEnv(profile=profile, seed=seed, render_frames=render_frames,
                         max_episode_sec=max_episode_sec, drone_width_m=drone_width_m)
    tally = {"passed": 0, "struck": 0, "missed": 0, "out_of_cage": 0, "timeout": 0}
    rewards, ticks = [], []
    try:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            if hasattr(policy_fn, "reset"):
                policy_fn.reset()
            total, n = 0.0, 0
            while True:
                obs, r, term, trunc, info = env.step(policy_fn(obs))
                total += r
                n += 1
                if term or trunc:
                    tally[info["phase"] if term else "timeout"] += 1
                    break
            rewards.append(total)
            ticks.append(n)
    finally:
        env.close()
    return {"episodes": episodes, "pass_rate": tally["passed"] / episodes,
            "mean_reward": float(np.mean(rewards)), "mean_ticks": float(np.mean(ticks)),
            **tally}
