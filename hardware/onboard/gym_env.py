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
ring entirely. (Lap mode, laps != 1: a pass does not end it; see below.)

Changed 2026-10-11 night, after measuring why a trained policy lined up with the ring but would
not go through, and flew into the net from hard starts (hardware/onboard/README.md, "Why the
policy lined up but didn't go through"):
  * A failure (strike, out of the cage, a miss) now also costs the time penalty of every tick
    left in the episode (`drain`), on top of a fixed penalty. Before, a strike (-20) cost exactly
    what a pass earned (+20), so attempting only paid above 50 % confidence, and out-of-cage
    (-10) was cheaper than the -12..-18 of searching to the time limit, so the net was a shortcut.
    Now ending an episode early is never cheaper than flying on, and an attempt pays from ~29 %.
  * The potential has a facing term: the camera pointing at the point just behind the ring
    (along the pass direction). It pays for turning towards a ring that is out of view, and after
    a pass for turning back round; potential-based, so hovering or spinning earns nothing.
  * A pass only counts flown camera-first (forward . pass direction >= PASS_FACING_MIN): no
    blind backwards or sideways passes.

v5 (2026-10-11, later that night):
  * lap_dir="same": every pass goes the same way. After a pass the drone has to fly round the
    OUTSIDE of the ring back to the side it entered from (sim.pop_clear_crossing: a crossing of
    the plane clear of the noodle), then line up and go again. Back through the opening is
    wrong-way. The shaping leads via a point RETURN_SIDE_M beside the ring, and while going
    round the facing term looks at the ring's centre (circle it while watching it).
  * A streak bonus: pass k of an episode pays R_PASSED x min(1 + streak x (k - 1), streak_cap).
  * start_mode="zones": starts cycle through 9 zones of the cage (sim.py).

v6 (2026-10-14 overnight, hardware/onboard/history.py):
  * history_lags: the actor sees the current observation plus older ones (memory for the way round
    the outside, where the ring is out of view and the velocity input is blank).
  * privileged: PRIV_DIM true-state numbers appended for the CRITIC only (asymmetric actor-critic,
    Pinto et al. 2018); aac_policy masks them out of the actor, so the real drone pads them with zeros.
  * w_dwell: a per-tick cost for loitering in the opening (within DWELL_BAND_M of the ring's plane,
    inside the rim) while not pushing forward. The real noodle hardly swings when the drone goes
    straight through, but it does when the drone hovers in it. 0 = off (the default before v6).
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
from .sim import DEFAULT_SWAY_DEG, OnboardSim, Ring

# Reward weights. Shaping is potential-based and per metre (summed over an episode it is
# phi_end - phi_start, so circling or parking earns nothing), comparable to the bonuses.
# The potential leads to a point in FRONT of the ring, then to one BEHIND it: it keeps rising
# through the plane, so stopping at the centre or backing out loses what the approach earned.
W_SHAPE = 3.0
AXIS_PULL = 1.0         # near the plane, also pulled onto the axis (x radial metres)
AXIS_PULL_LEN_M = 0.30
APPROACH_M = 0.30       # the waypoints in front of / behind the ring
W_TIME = 0.02
R_PASSED = 20.0
R_STRUCK = -8.0         # plus the drain below: a real strike usually ends the flight (was -20)
R_MISSED = -5.0
R_OUT_OF_CAGE = -10.0
# Failures also cost W_TIME x the ticks left in the episode, so ending early is never cheaper
# than flying on (the PyBullet budget mode's "drain the budget" idea).
DRAIN = True
# Facing potential (x W_SHAPE, like the metres terms): cos of the angle between the camera and
# the direction to the point FACE_AHEAD_M behind the ring along the pass direction.
W_FACE = 0.5
FACE_AHEAD_M = 0.30
PASS_FACING_MIN = 0.5   # a pass counts only with forward . pass direction >= this
# Once, the first time the ring comes into view in an episode that started without it in
# view (hard starts: behind / beside / below, facing away). Helps the first searches succeed;
# it can only be earned once per episode, so turning away and back again earns nothing.
R_FOUND = 0.5
# Per step, rising from 0 at RIM_MARGIN_M of clearance to W_RIM at touching. Progress shaping
# alone pays almost the same for ending 5 cm or 12 cm off-axis; this is the gradient away from
# the noodle that the single terminal strike penalty does not give.
W_RIM = 0.3
RIM_MARGIN_M = 0.08
# Lap mode (laps != 1): a pass does not end the episode. The required direction flips and the
# next pass only counts once the drone is ARM_DIST_M clear on the exit side -- the same rules as
# scripts/hoop_net_env.py. Going back through the opening the wrong way (backing out, or turning
# straight round before re-arming) costs R_WRONG_WAY and counts for nothing.
R_WRONG_WAY = -5.0
ARM_DIST_M = 0.25

MAX_EPISODE_SEC = 30.0
HARD_START_EPISODE_SEC = 40.0      # searching for the ring and repositioning take longer
LAP_EPISODE_SEC = 90.0             # same-side laps: a way round the ring is ~10-15 s
STREAK_GAIN = 0.5                  # pass k pays R_PASSED x min(1 + this x (k - 1), STREAK_CAP)
STREAK_CAP = 3.0
RETURN_SIDE_M = 0.55               # same-side laps: the way round passes this far beside the ring
LAP_DIRS = ("alternate", "same")
W_DWELL = 0.0                      # v6: per tick loitering in the opening (train_onboard sets 0.05)
DWELL_BAND_M = 0.15
DWELL_FB_MIN = 0.5                 # forward stick below this counts as loitering

# the weights above, by name (saved in a run's meta.json so the dashboard can show the MDP)
REWARD_WEIGHTS = {"w_shape": W_SHAPE, "axis_pull": AXIS_PULL, "w_time": W_TIME,
                  "r_passed": R_PASSED, "r_struck": R_STRUCK, "r_missed": R_MISSED,
                  "r_out_of_cage": R_OUT_OF_CAGE, "r_found": R_FOUND,
                  "w_rim": W_RIM, "rim_margin_m": RIM_MARGIN_M,
                  "r_wrong_way": R_WRONG_WAY, "arm_dist_m": ARM_DIST_M,
                  "drain": DRAIN, "w_face": W_FACE, "face_ahead_m": FACE_AHEAD_M,
                  "pass_facing_min": PASS_FACING_MIN, "streak": STREAK_GAIN,
                  "streak_cap": STREAK_CAP, "return_side_m": RETURN_SIDE_M,
                  "w_dwell": W_DWELL}


def rim_penalty(rim_distance_m: float) -> float:
    return W_RIM * float(np.clip(1.0 - rim_distance_m / RIM_MARGIN_M, 0.0, 1.0))


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
                 dr: float = 0.0, sway_deg: float = DEFAULT_SWAY_DEG, hard_start_p: float = 0.0,
                 ramp: float = 1.0, open_plane: Optional[bool] = None, laps: int = 1,
                 wash_deg: float = 0.0, lap_dir: str = "alternate", start_mode: str = "ring",
                 streak: float = STREAK_GAIN, streak_cap: float = STREAK_CAP,
                 ring_tracker: bool = False, reward_weights: Optional[dict] = None,
                 history_lags=None, privileged: bool = False, near_pass_p: float = 0.0,
                 oblique_p: float = 0.0, stall_sec: float = 0.0, stress_p: float = 0.0):
        """dr / sway_deg / hard_start_p / ramp / open_plane / wash_deg: see sim.py. open_plane
        defaults to on whenever starts can be away from the front runway. laps: passes per
        episode (1: the episode ends at the first crossing; 0: endless). lap_dir: "alternate"
        (back-and-forth) or "same" (round the outside, same side every time). start_mode: "ring"
        (front runway / hard starts) or "zones" (sim.py). streak / streak_cap: the streak bonus.
        ring_tracker: append obs.RingTracker's 4 fields (spec.TRACKER_OBS_DIM inputs).
        reward_weights: overrides of REWARD_WEIGHTS by name (a typo raises). The module constants
        are the defaults, so None behaves exactly as before; streak / streak_cap given here win
        over the streak / streak_cap arguments.
        history_lags / privileged: hardware/onboard/history.py (None / False = the plain observation).
        stall_sec > 0 (training): cut the episode short (truncated, like a time limit, so PPO still
        bootstraps the value) after this many seconds without a counted pass. A policy that parks
        then wastes seconds, not the rest of a 90 s episode, and gets more fresh starts."""
        super().__init__()
        if lap_dir not in LAP_DIRS:
            raise ValueError(f"lap_dir must be one of {LAP_DIRS}")
        self.lap_dir = lap_dir
        self.rw = dict(REWARD_WEIGHTS)
        for k, v in (reward_weights or {}).items():
            if k not in self.rw:
                raise ValueError(f"unknown reward weight '{k}'; expected one of {sorted(self.rw)}")
            self.rw[k] = bool(v) if k == "drain" else float(v)
        self.streak = self.rw["streak"] if "streak" in (reward_weights or {}) else float(streak)
        self.streak_cap = (self.rw["streak_cap"] if "streak_cap" in (reward_weights or {})
                           else float(streak_cap))
        self.rw["streak"], self.rw["streak_cap"] = self.streak, self.streak_cap
        from .history import HistoryStack, parse_lags, stacked_width
        self.obs_dim = spec.TRACKER_OBS_DIM if ring_tracker else spec.OBS_DIM   # one frame
        self.history_lags = parse_lags(history_lags)
        self.history = HistoryStack(self.obs_dim, self.history_lags) if self.history_lags else None
        self.privileged = bool(privileged)
        self.actor_dim = stacked_width(self.obs_dim, self.history_lags)
        self.full_dim = stacked_width(self.obs_dim, self.history_lags, self.privileged)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.full_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(spec.ACT_DIM,), dtype=np.float32)
        self.laps = int(laps)
        if open_plane is None:
            open_plane = hard_start_p > 0 or self.laps != 1   # laps: turning round the outside is fine
        self._kwargs = dict(profile=profile, render=render_frames, ring=ring,
                            detector=detector, drone_width_m=drone_width_m, dr=dr,
                            sway_deg=sway_deg, hard_start_p=hard_start_p, ramp=ramp,
                            open_plane=open_plane, wash_deg=wash_deg, start_mode=start_mode)
        self._seed = int(seed)
        # curriculum (training only): this share of episodes starts lined up just in front of
        # the ring (sim.place_near_pass), so most of the episode is the loop after the pass.
        # Its own random stream, so near_pass_p = 0 leaves every other draw unchanged.
        self.near_pass_p = float(near_pass_p)
        self._np_rng = np.random.default_rng(self._seed + 7_777)
        # and this share starts close to the ring at a slant (sim.place_oblique), own stream too
        self.oblique_p = float(oblique_p)
        self.stress_p = float(stress_p)          # training only: sim.STRESS_* drone, a share of episodes
        self._ob_rng = np.random.default_rng(self._seed + 8_888)
        if self.laps != 1:
            max_episode_sec = max(max_episode_sec, LAP_EPISODE_SEC)
        self.base_ticks = self.max_ticks = int(max_episode_sec * spec.CTRL_HZ)
        self.stall_ticks = int(round(float(stall_sec) * spec.CTRL_HZ))
        self._last_pass_tick = 0
        self._zone_weights = None        # training: failure-weighted start zones (set_zone_weights)
        self._next_zone_index = None     # evaluation: the next start zone (set_zone_index)
        self.hard_ticks = int(max(max_episode_sec, HARD_START_EPISODE_SEC) * spec.CTRL_HZ)
        self._found_once = True
        self.sim: Optional[OnboardSim] = None
        self.builder = ObsBuilder(width=self.obs_dim)
        self._ticks = 0
        self.passes = 0
        self._dir = 0            # required crossing direction along the ring normal; 0 = either
        self._armed = True
        self._returned = True    # same-side laps: back round the outside since the last pass
        self._ret_side = 1.0
        self.streak_mult = 1.0
        self._phi = 0.0

    # -- geometry ---------------------------------------------------------------------
    def _axial_radial(self):
        d = self.sim.pos - self.sim.ring.centre
        axial = float(np.dot(d, self.sim.ring.normal))
        radial = float(np.linalg.norm(d - axial * self.sim.ring.normal))
        return axial, radial

    def _leg_dir(self, axial: float) -> int:
        if self._dir:
            return self._dir
        return -1 if axial > 0 else 1                    # towards and through from this side

    def _potential(self) -> float:
        """Shaping potential (m, <= 0): minus the distance still to fly via the approach point in
        front of the ring to the exit point behind it, minus a pull onto the axis near the plane.
        Same structure as scripts/hoop_net_env.py's MDP v2."""
        c, n = self.sim.ring.centre, self.sim.ring.normal
        axial, radial = self._axial_radial()
        d = self._leg_dir(axial)
        if self.lap_dir == "same" and not self._returned:
            # round the outside: via the point beside the ring, watching the ring as it goes
            side = self._return_wp()
            phi = -(float(np.linalg.norm(self.sim.pos - side))
                    + float(np.linalg.norm(side - (c - n * d * APPROACH_M))) + 2 * APPROACH_M)
            return phi + self.rw["w_face"] * self._facing(c)
        s_eff = axial * d
        lined_up = (self._armed and -0.45 < s_eff < 0.05
                    and radial < 1.2 * self.sim.ring.radius)
        if lined_up:
            phi = -float(np.linalg.norm(self.sim.pos - (c + n * d * APPROACH_M)))
        else:
            phi = -(float(np.linalg.norm(self.sim.pos - (c - n * d * APPROACH_M))) + 2 * APPROACH_M)
        return (phi - self.rw["axis_pull"] * radial * float(np.exp(-(axial / AXIS_PULL_LEN_M) ** 2))
                + self.rw["w_face"] * self._facing(c + n * d * self.rw["face_ahead_m"]))

    def _side_axis(self) -> np.ndarray:
        n = self.sim.ring.normal
        u = np.array([-n[1], n[0], 0.0])
        return u / (np.linalg.norm(u) + 1e-12)

    def _return_wp(self) -> np.ndarray:
        c = self.sim.ring.centre
        wp = c + self._ret_side * self.rw["return_side_m"] * self._side_axis()
        wp[2] = c[2]
        return wp

    def _pick_return_side(self) -> float:
        """The side of the axis the drone is on, unless that way round runs into the net."""
        from .sim import CAGE_XY
        c, u = self.sim.ring.centre, self._side_axis()
        side = 1.0 if float(np.dot(self.sim.pos - c, u)) >= 0 else -1.0
        lim = CAGE_XY - 0.17
        for s in (side, -side):
            wp = c + s * self.rw["return_side_m"] * u
            if abs(wp[0]) <= lim and abs(wp[1]) <= lim:
                return s
        return side

    def _facing(self, point) -> float:
        """cos of the horizontal angle from the camera to `point`, faded out within 15 cm of it."""
        g = (point - self.sim.pos)[:2]
        dist = float(np.linalg.norm(g))
        if dist < 1e-6:
            return 0.0
        return float(np.dot(self.sim.forward[:2], g / dist)) * min(1.0, dist / 0.15)

    def _drain(self) -> float:
        """The time penalty of every tick left in this episode (0 with DRAIN off)."""
        return self.rw["w_time"] * max(0, self.max_ticks - self._ticks) if self.rw["drain"] else 0.0

    def _reset_shaping(self):
        """Restart the potential from where the drone is now (a new episode or a new leg), so a
        change of target pays nothing."""
        self._phi = self._potential()

    # -- gym api ----------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        s = self._seed if seed is None else int(seed)
        self._seed += 1                                  # fresh layout each episode
        if self.sim is None:
            self.sim = OnboardSim(seed=s, **self._kwargs)
            self.sim.stress_p = self.stress_p
        if self._next_zone_index is not None:
            self.sim._zone_i = self._next_zone_index
            self._next_zone_index = None
        self.sim.zone_weights = self._zone_weights
        ring_obs = self.sim.reset(seed=s)
        if self.near_pass_p > 0 and self._np_rng.random() < self.near_pass_p:
            ring_obs = self.sim.place_near_pass(self._np_rng)
        elif self.oblique_p > 0 and self._ob_rng.random() < self.oblique_p:
            ring_obs = self.sim.place_oblique(self._ob_rng)
        self.builder.reset()
        self._ticks = 0
        self._last_pass_tick = 0
        self.max_ticks = self.base_ticks if self.sim.start_type == "front" else self.hard_ticks
        self._found_once = bool(ring_obs.found)          # in view from the start: no bonus
        self.passes = 0
        self._armed = True
        self._returned = True
        self.streak_mult = 1.0
        axial, _ = self._axial_radial()
        # through the ring from the side it starts on; starting in (or near) the ring's plane,
        # beside or below it, either direction counts for the first pass
        self._dir = 0 if abs(axial) < self.rw["arm_dist_m"] else (-1 if axial > 0 else 1)
        self._reset_shaping()
        obs = self.builder.build(ring_obs, self.sim.state(), self.sim.t)
        if self.history is not None:
            self.history.reset()
        return self._wrap_obs(obs), {}

    def _wrap_obs(self, obs):
        """One frame -> what the policy sees (history stack, then the critic's privileged inputs)."""
        if self.history is not None:
            obs = self.history.push(obs)
        if self.privileged:
            from .history import privileged_features
            obs = np.concatenate([obs, privileged_features(self)]).astype(np.float32)
        return obs

    def step(self, action):
        a = np.asarray(action, dtype=float).reshape(-1)[:spec.ACT_DIM]
        ring_obs = self.sim.step(a)
        self.builder.push_action(a)
        obs = self.builder.build(ring_obs, self.sim.state(), self.sim.t)
        self._ticks += 1

        cross = self.sim.pop_clear_crossing()
        returned_now = False
        if (self.lap_dir == "same" and not self._returned and self._dir
                and cross == -self._dir):                # back to the entry side round the outside
            self._returned = returned_now = True
        if not self._armed and self._returned:
            axial, _ = self._axial_radial()
            if axial * self._dir < -self.rw["arm_dist_m"]:   # flown clear: the next pass counts
                self._armed = True
        phi = self._potential()
        if returned_now:
            self._phi = phi                              # a new stage of the lap: no jump paid
        reward = (self.rw["w_shape"] * (phi - self._phi)
                  - self.rw["w_time"]
                  - self.rw["w_rim"] * float(np.clip(
                      1.0 - self.sim.rim_distance() / self.rw["rim_margin_m"], 0.0, 1.0)))
        self._phi = phi
        if self.rw["w_dwell"] > 0:
            axial, radial = self._axial_radial()
            if (abs(axial) < DWELL_BAND_M and radial < self.sim.ring.radius
                    and float(a[spec.ACT_FB]) < DWELL_FB_MIN):
                reward -= self.rw["w_dwell"]
        if ring_obs.found and not self._found_once:
            self._found_once = True
            reward += self.rw["r_found"]

        terminated = False
        info = {"phase": "flying", "start_type": self.sim.start_type, "wrong_way": False}
        o = self.sim.outcome
        if o.done:
            info["contact"] = bool(o.contact)
            info["radial_m"] = o.radial_m
            facing = float(np.dot(self.sim.forward, self.sim.ring.normal * o.direction))
            right_way = self.laps == 1 or (self._armed and self._dir in (0, o.direction))
            if o.passed and right_way and facing < self.rw["pass_facing_min"]:
                info["phase"] = "blind_pass"             # backwards / sideways: counts for nothing
                self.sim.clear_outcome()
                self._reset_shaping()
            elif o.passed:
                if right_way:
                    self.streak_mult = min(1.0 + self.streak * self.passes, max(1.0, self.streak_cap))
                    reward += self.rw["r_passed"] * self.streak_mult
                    self.passes += 1
                    self._last_pass_tick = self._ticks
                    info["phase"] = "passed"
                    terminated = self.laps > 0 and self.passes >= self.laps
                    if not terminated:
                        if self.lap_dir == "same":       # same way in: round the outside first
                            self._dir = o.direction
                            self._returned = False
                            self._ret_side = self._pick_return_side()
                        else:                            # turn round and come back through
                            self._dir = -o.direction
                        self._armed = False
                        self._found_once = False         # finding it again is paid again
                else:
                    reward += self.rw["r_wrong_way"]
                    info["phase"] = "wrong_way"
                    info["wrong_way"] = True
                if not terminated:
                    self.sim.clear_outcome()
                    self._reset_shaping()
            elif o.struck:
                terminated = True
                reward += self.rw["r_struck"] - self._drain()
                info["phase"] = "struck"
            else:
                terminated = True
                reward += self.rw["r_missed"] - self._drain()
                info["phase"] = "missed"
        if not terminated and self.sim.out_of_cage():
            terminated = True
            reward += self.rw["r_out_of_cage"] - self._drain()
            info["phase"] = "out_of_cage"

        truncated = (not terminated) and self._ticks >= self.max_ticks
        if (not terminated and not truncated and self.stall_ticks > 0
                and self._ticks - self._last_pass_tick >= self.stall_ticks):
            truncated = True
            info["stalled"] = True
        info["passes"] = self.passes                     # the dashboard HUD's pass count
        if terminated or truncated:
            # SB3's EvalCallback reads this at episode end: the dashboard's success rate.
            info["is_success"] = self.passes >= 1
        info["reversed"] = False
        info["ring_found"] = bool(ring_obs.found)
        info["wash_tilt_deg"] = self.sim.wash_tilt_deg
        info["lap_dir"] = self.lap_dir
        info["returned"] = bool(self._returned)
        info["streak_mult"] = float(self.streak_mult)
        info["range_m"] = float(ring_obs.range_m)
        return self._wrap_obs(obs), float(reward), terminated, truncated, info

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
                "hoop_radius": float(r.radius), "passes": int(self.passes)}

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

    def set_wash_deg(self, deg: float):
        self._kwargs["wash_deg"] = float(deg)
        if self.sim is not None:
            self.sim.wash_deg = float(deg)

    def set_sway_deg(self, deg: float):
        self._kwargs["sway_deg"] = float(deg)
        if self.sim is not None:
            self.sim.sway_deg = float(deg)

    def set_zone_index(self, index: int):
        """Start zone (0-8) of the next reset (zones mode): rewinds the round-robin, so evaluations
        pair the same seeds with the same zones every time (train_speedups.reset_eval_layouts)."""
        self._next_zone_index = int(index)

    def set_zone_weights(self, weights):
        """Training only: draw start zones with these 9 probabilities (None = round-robin)."""
        self._zone_weights = None if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)

    def set_stall_sec(self, sec: float):
        self.stall_ticks = int(round(float(sec) * spec.CTRL_HZ))

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
