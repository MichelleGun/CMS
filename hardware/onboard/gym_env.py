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
# Once, the first time the ring comes into view in an episode that started without it in
# view (hard starts: behind / beside / below, facing away). Helps the first searches succeed;
# it can only be earned once per episode, so turning away and back again earns nothing.
R_FOUND = 0.5

MAX_EPISODE_SEC = 30.0
HARD_START_EPISODE_SEC = 40.0      # searching for the ring and repositioning take longer

# the weights above, by name (saved in a run's meta.json so the dashboard can show the MDP)
REWARD_WEIGHTS = {"w_axial": W_AXIAL, "w_radial": W_RADIAL, "w_time": W_TIME,
                  "r_passed": R_PASSED, "r_struck": R_STRUCK, "r_missed": R_MISSED,
                  "r_out_of_cage": R_OUT_OF_CAGE, "r_found": R_FOUND}


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
                 drone_width_m: float = spec.DRONE_WIDTH_M,
                 dr: float = 0.0, sway_deg: float = 15.0, hard_start_p: float = 0.0,
                 ramp: float = 1.0, open_plane: Optional[bool] = None):
        """dr / sway_deg / hard_start_p / ramp / open_plane: robustness, see sim.py. open_plane
        defaults to on whenever starts can be away from the front runway."""
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(spec.OBS_DIM,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(spec.ACT_DIM,), dtype=np.float32)
        if open_plane is None:
            open_plane = hard_start_p > 0
        self._kwargs = dict(profile=profile, render=render_frames, ring=ring,
                            detector=detector, drone_width_m=drone_width_m, dr=dr,
                            sway_deg=sway_deg, hard_start_p=hard_start_p, ramp=ramp,
                            open_plane=open_plane)
        self._seed = int(seed)
        self.base_ticks = self.max_ticks = int(max_episode_sec * spec.CTRL_HZ)
        self.hard_ticks = int(max(max_episode_sec, HARD_START_EPISODE_SEC) * spec.CTRL_HZ)
        self._found_once = True
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
        self.max_ticks = self.base_ticks if self.sim.start_type == "front" else self.hard_ticks
        self._found_once = bool(ring_obs.found)          # in view from the start: no bonus
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
        if ring_obs.found and not self._found_once:
            self._found_once = True
            reward += R_FOUND

        terminated = False
        info = {"phase": "flying", "start_type": self.sim.start_type}
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
            info["passes"] = int(info["is_success"])     # the dashboard HUD's pass count
        info["reversed"] = False
        info["ring_found"] = bool(ring_obs.found)
        info["range_m"] = float(ring_obs.range_m)
        return obs, float(reward), terminated, truncated, info

    # -- dashboard (callable through VecEnv.env_method, so they work in subprocess envs) -----
    def telemetry(self) -> dict:
        """The drone's pose and the ring, in the same cage frame as the dashboard's 3D view
        (x, y in [-1, 1] m, z up from the floor); a small dict, cheap to send between processes."""
        if self.sim is None:
            return {}
        r = self.sim.ring
        return {"pos": [float(v) for v in self.sim.pos],
                "rpy": [0.0, 0.0, float(np.radians(self.sim.yaw_deg))],
                "target": [float(v) for v in r.centre],
                "hoop_normal": [float(v) for v in r.normal],
                "hoop_radius": float(r.radius)}

    def onboard_frame(self, res=(200, 150)):
        """What the drone's camera sees (RGB uint8), for the dashboard's POV. Leaves the
        simulator's random stream untouched (frame() draws its dropped-frame chance from it),
        so watching never changes what is learned."""
        import cv2
        if self.sim is None:
            return np.zeros((res[1], res[0], 3), np.uint8)
        state = self.sim.rng.bit_generator.state
        try:
            bgr = self.sim.frame()
        finally:
            self.sim.rng.bit_generator.state = state
        return cv2.cvtColor(cv2.resize(bgr, tuple(res)), cv2.COLOR_BGR2RGB)

    def set_ramp(self, ramp: float):
        """Curriculum hook: how much of dr / hard_start_p applies (0..1), from the next episode."""
        self._kwargs["ramp"] = float(ramp)
        if self.sim is not None:
            self.sim.ramp = float(ramp)

    def set_sway_deg(self, deg: float):
        self._kwargs["sway_deg"] = float(deg)
        if self.sim is not None:
            self.sim.sway_deg = float(deg)

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
