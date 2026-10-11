"""
RL environment that mirrors the real hardware cage: a 2x2x2 m net with a physical
vertical ring near its centre. The drone starts on the floor (default) or hovering at the
Tello's takeoff height (start_z, used by training since 2026-10-11 night), finds its way to the
ring and passes through it in the required direction.

Compared with hoop_env.HoopFindAviary (a point to reach, no net, no ring geometry):
- the ring has an opening, a tube, a normal (facing) and is slightly re-posed every
  episode (position, yaw/tilt of its face, opening size) to cover repositioning between
  flights;
- the ring is a light pool noodle hung from a rope tied to its top, so it also moves *during*
  an episode: in a per-episode wind it swings like a pendulum about that top knot (the knot
  barely drifts, the bottom swings furthest) and is shoved when the drone clips its rim,
  then swings back. wash_deg adds what the real noodle actually does: the drone's prop wash
  pushes it away as the drone comes close, tilting it up about the rope (its bottom moves away
  from the drone). The observation always holds the ring's live pose;
- success is a *pass through the opening* in the +normal direction, not proximity;
- touching the net ends the episode (net_terminate=True, the default: the real drone lands
  when it hits the net); with net_terminate=False it bounces the drone back with a small
  penalty and the episode continues. Clipping the ring's rim always bounces and continues;
- start_z=None starts on the floor; start_z=0.8 starts hovering there (+-0.1 m), which is where
  the real Tello hands over to the policy after its automatic takeoff (fly_hoop.py);
- laps=0 is endless back-and-forth: after each pass the required direction flips, and the
  next pass only counts once the drone has flown clear of the ring on the exit side ("armed").
  armed_obs=True puts that flag in the observation, otherwise "just passed, not re-armed" and
  "lined up, go" look identical to the policy. lap_dir="same" (v5) keeps the direction instead:
  after a pass the drone has to fly round the OUTSIDE of the ring back to the side it entered
  from (crossing the ring's plane clear of the noodle), then line up and go through again the
  same way; going back through the opening is a wrong-way crossing and re-arms nothing;
- start_mode="zones" (v5) starts each episode in the next of 9 zones, round-robin: the 8 outer
  cells of a 3x3 grid over the floor plan (top-left, top-middle, top-right, middle-right,
  bottom-right, bottom-middle, bottom-left, middle-left; top = +y, right = +x) and right under
  the hoop, so every area is trained equally often (random starts plus carry_start used to
  leave the policy passing from one corner).

Observation (82-D, KIN; 83-D with armed_obs=True; 84-D with streak_obs=True as well):
    [kin12, last-15-actions x4 (60), ring_normal (3), ring_velocity (3), ring_inner_radius (1),
     (armed (1), (streak = min(passes, 4) / 4 (1),)) hoop_centre - drone_pos (3)]
hoop_env's 75-D vector with the ring's normal, live velocity (it swings) and opening size
inserted before the final relative vector, so `obs[-3:]` is still "vector to the hoop" for the
telemetry code. (Policies trained before 2026-10-10 evening used a 78-D layout without the
velocity and radius.)

Reward, two modes (reward_mode):
- "budget" (MDP v3, used since 2026-10-11): -0.005 per step, +1 per pass, -0.05 per rim clip.
  Passes never end the episode (endless back-and-forth); it ends when the running total falls
  to -budget (default -3: 20 s for the first pass, and each pass buys 6.7 s more), when the drone
  touches the net (that drains what is left of the budget, so the total lands on -budget too),
  or at the episode_sec safety cap. No shaping by default: v2's shaping paid the drone for parking
  in front of the ring. MDP v4 = budget + reward weight budget_shape > 0: the same leg-aware
  potential as v2 (approach point -> through -> after a pass, out to the next approach point),
  potential-based so parking earns nothing, and kept out of the budget tally so it never changes
  when an episode ends. MDP v5 adds a streak bonus (budget_streak > 0): pass k of an episode pays
  budget_pass x min(1 + budget_streak x (k - 1), budget_streak_cap), the extra over budget_pass
  also kept out of the budget tally; and budget_wrong, a cost for going back through the opening.
- "shaped" (MDP v2, the default for backward compatibility): potential-based shaping towards a
  waypoint in front of the ring, then one behind it (plus a pull onto the ring's axis on the
  approach side), a centring-weighted pass bonus, and penalties for the rim, the net and
  wrong-way crossings.

Action: ActionType.VEL, `[dx, dy, dz, s]` (direction + speed fraction of ~0.25 m/s).

All geometry lives in `HoopSpec`. The nominal ring pose is the blue noodle measured in the
phone scan of the real cage (scripts/measure_scan_hoop.py).
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

# Names the dashboard backend (webapp/backend/config.py) imports from the env module.
ROOM = np.array([2.0, 2.0, 2.0])   # x, y, z (m) -- the net, floor at z=0
XY_MARGIN = 0.25                   # keep the ring this far from the side nets
START_MARGIN = 0.12                # drone starts anywhere on the floor this far from the nets
                                   # (drone radius 0.07 + a little: clear of the net-contact zone)
START_Z = 0.03                     # resting height on the floor
REACH_RADIUS = 0.15                # kept for dashboard compatibility (not used for success)


@dataclass(frozen=True)
class HoopSpec:
    """Ring + drone geometry (m / deg). The nominal pose is the noodle in the phone scan of the
    real cage, placed the way the dashboard overlays it (scripts/measure_scan_hoop.py)."""
    centre: tuple = (0.04, 0.07, 1.29)  # nominal ring centre (scan: 1.29 m up, a few cm off centre)
    inner_radius: float = 0.15          # 0.30 m opening
    tube_radius: float = 0.03           # 6 cm pool-noodle tube -> 0.42 m outer diameter
    normal_yaw_deg: float = -29.0       # nominal facing: +x turned 29 deg towards -y (scan)
    normal_tilt_deg: float = -18.0      # the hanging noodle leans: its face points 18 deg down (scan)
    # per-episode jitter (repositioning between flights)
    centre_jitter: tuple = (0.10, 0.10, 0.08)
    yaw_jitter_deg: float = 20.0
    tilt_jitter_deg: float = 8.0
    radius_jitter_frac: float = 0.10
    # live wind (within an episode): each episode draws a horizontal wind direction, a gust
    # strength in [0, 1] and a swing period; the ring hangs from a rope tied to its top, so it
    # swings about that knot along the wind (the knot itself only drifts a little)
    sway_tilt_deg: float = 20.0         # default largest swing about the top knot (full gust)
    sway_top_amp: float = 0.02          # m, knot drift at that default swing (scales with it)
    sway_period_s: tuple = (1.5, 3.0)   # swing period range (s)
    # the drone shoving the ring: clipping the rim hands the ring part of the drone's speed into it;
    # the ring then springs back to its episode pose (damped, at the swing period)
    push_gain: float = 0.5              # fraction of the drone's speed into the rim given to the ring
    push_max: float = 0.15              # m, largest shove away from the episode pose
    push_damping: float = 0.3           # damping ratio of the swing back
    # drone / net
    drone_radius: float = 0.07          # 0.14 m wide (hardware/flights meta.json)
    net_margin: float = 0.08            # net contact when the drone centre is this near a net face
    restitution: float = 0.4            # bounce: fraction of normal speed returned
    min_bounce_speed: float = 0.08      # m/s, so the drone always actually leaves the net
    approach_dist: float = 0.35         # shaping waypoint distance either side of the ring
    arm_dist: float = 0.28              # laps: fly this far out past the ring before the next pass counts
    # prop wash (wash_deg > 0): as the drone comes within ~wash_range of the ring its downwash pushes
    # the light noodle away, tilting it about the rope knot (bottom away from the drone); it follows
    # the push through a damped spring and swings back once the drone has gone
    wash_range: float = 0.45            # m, Gaussian reach of the push from the ring centre
    wash_strength: tuple = (0.5, 1.5)   # per-episode multiplier of wash_deg
    wash_damping: float = 0.6
    wash_side_soft: float = 0.10        # m, the push direction fades through the ring's plane
    start_z_jitter: float = 0.10        # m, hover starts: start_z +- this


# reward terms (per control step unless noted)
SHAPING_GAIN = 10.0        # x change in potential (m)
AXIS_PULL = 1.5            # potential weight of the distance off the ring's axis on the approach side
AXIS_PULL_LEN = 0.4        # m, how far in front of the ring that pull reaches (Gaussian length)
TIME_PENALTY = 0.005
PASS_BONUS = 20.0          # per pass ...
CENTRE_BONUS = 10.0        # ... plus up to this for a dead-centre one
RIM_PENALTY = 3.0
NET_PENALTY = 5.0          # net_terminate=True: touching the net ends the episode
NET_BOUNCE_PENALTY = 1.0   # net_terminate=False: bounce and carry on
WRONG_WAY_PENALTY = 5.0

# reward_mode="budget" (MDP v3)
BUDGET_TIME = 0.005        # per control step (1 per 6.7 s)
BUDGET_PASS = 1.0
BUDGET_RIM = 0.05
BUDGET_NET = 0.1           # the minimum net cost; touching the net also ends the episode at -budget
BUDGET_SHAPE = 0.0         # MDP v4: x metres of potential progress (0 = plain v3)
BUDGET_STREAK = 0.0        # MDP v5: pass k pays x min(1 + this x (k - 1), cap) (0 = flat)
BUDGET_STREAK_CAP = 3.0
BUDGET_WRONG = 0.0         # MDP v5: back through the opening the wrong way
LAP_DIRS = ("alternate", "same")
START_MODES = ("random", "zones")
# zones 1-8: (column, row) of the outer cells of a 3x3 grid, top = +y, right = +x; zone 9 = under the hoop
ZONE_CELLS = ((-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0))
ZONE_NAMES = ("top-left", "top-middle", "top-right", "middle-right", "bottom-right",
              "bottom-middle", "bottom-left", "middle-left", "under the hoop")
RETURN_SIDE_M = 0.55       # same-side laps: the way round passes this far beside the ring's centre
REWARD_MODES = ("shaped", "budget")

# the weights above, by name: reward_weights= (and the dashboard) can override any of them per run
DEFAULT_REWARD_WEIGHTS = {
    "shaping_gain": SHAPING_GAIN, "axis_pull": AXIS_PULL, "time_penalty": TIME_PENALTY,
    "pass_bonus": PASS_BONUS, "centre_bonus": CENTRE_BONUS, "rim_penalty": RIM_PENALTY,
    "net_penalty": NET_PENALTY, "net_bounce_penalty": NET_BOUNCE_PENALTY,
    "wrong_way_penalty": WRONG_WAY_PENALTY,
    "budget_time": BUDGET_TIME, "budget_pass": BUDGET_PASS, "budget_rim": BUDGET_RIM,
    "budget_net": BUDGET_NET, "budget_shape": BUDGET_SHAPE,
    "budget_streak": BUDGET_STREAK, "budget_streak_cap": BUDGET_STREAK_CAP,
    "budget_wrong": BUDGET_WRONG,
}


def merge_reward_weights(overrides=None) -> dict:
    """DEFAULT_REWARD_WEIGHTS with `overrides` merged in (unknown names raise)."""
    w = dict(DEFAULT_REWARD_WEIGHTS)
    for k, v in (overrides or {}).items():
        if k not in w:
            raise ValueError(f"unknown reward weight '{k}' (known: {', '.join(w)})")
        w[k] = float(v)
    return w

OBS_EXTRA = 3 + 3 + 1 + 3  # ring normal, ring velocity, ring radius, ring centre - drone
ARMED_INDEX = -4           # armed_obs=True: the flag sits just before the relative vector


DEFAULT_SPEC = HoopSpec()
# ring centre height range of the per-episode pose (the dashboard reads it)
Z_RANGE = (DEFAULT_SPEC.centre[2] - DEFAULT_SPEC.centre_jitter[2],
           DEFAULT_SPEC.centre[2] + DEFAULT_SPEC.centre_jitter[2])


def ring_normal(yaw_deg, tilt_deg):
    """Unit normal from a yaw about z (0 deg = +x) and a tilt (pitch) about the ring's y axis."""
    y, t = np.radians(yaw_deg), np.radians(tilt_deg)
    return np.array([np.cos(t) * np.cos(y), np.cos(t) * np.sin(y), np.sin(t)])


def ring_up(n):
    """Unit vector in the ring's plane closest to world up (towards its top)."""
    up = np.array([0.0, 0.0, 1.0]) - n[2] * np.asarray(n, dtype=float)
    return up / np.linalg.norm(up)


def rotate(v, axis, angle):
    """Rodrigues rotation of v about a unit axis by angle (rad)."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(axis @ v) * (1 - c)


class HoopNetAviary(BaseRLAviary):
    """Single-agent RL problem: take off from the floor, thread the ring, avoid the net."""

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
                 target_pos=None,
                 spec: HoopSpec = DEFAULT_SPEC,
                 ring_jitter: float = 1.0,
                 hoop_radius_scale: float = 1.0,
                 laps: int = 1,
                 wind: float = 1.0,
                 sway_deg: float = None,
                 carry_start: float = 0.7,
                 reverse_prob: float = 0.5,
                 lean_flip_prob: float = 0.3,
                 net_terminate: bool = True,
                 episode_sec: float = None,
                 reward_mode: str = "shaped",
                 budget: float = 3.0,
                 reward_weights: dict = None,
                 wash_deg: float = 0.0,
                 start_z: float = None,
                 armed_obs: bool = False,
                 streak_obs: bool = False,
                 lap_dir: str = "alternate",
                 start_mode: str = "random"):
        """start_pos: optional (x, y[, z]) -- only x/y are used, the drone always starts on the
        floor. target_pos: optional ring centre override (jitter on the centre is then off).
        ring_jitter: 0 disables per-episode ring jitter, 1 = full. hoop_radius_scale > 1 widens
        the opening (training curriculum). wind: 0 keeps the ring still within an episode (no
        sway, no shove), 1 = full live wind. sway_deg: largest swing of the ring about its top knot at
        full gust (deg, default spec.sway_tilt_deg). laps > 1 flips the required crossing direction after
        each pass, so the drone has to turn round and come back through; laps=0 does that endlessly
        (only the net, a tip-over or the time limit end the episode).
        carry_start: chance an episode starts where the previous one ended (as if the drone landed
        there), otherwise anywhere on the floor. reverse_prob: chance the ring must be flown through
        back to front. lean_flip_prob: chance the ring leans the other way (wind holding it over).
        net_terminate: touching the net ends the episode (else bounce and continue).
        episode_sec: time limit (s; default 20, or 120 in budget mode where it is only a safety cap).
        reward_mode: "shaped" or "budget" (see the module docstring). Budget mode always flies
        endless laps and always ends the episode at the net. budget: the episode ends when its
        total reward falls to -budget. reward_weights: overrides of DEFAULT_REWARD_WEIGHTS.
        wash_deg: largest prop-wash tilt of the ring (deg, 0 = off). start_z: None = floor start,
        else hover at this height (+- spec.start_z_jitter). armed_obs: add the lap "armed" flag to
        the observation (83-D). streak_obs: also add the pass count (84-D, needs armed_obs).
        lap_dir: "alternate" (back-and-forth) or "same" (round the outside, same side every time).
        start_mode: "random" (anywhere, or carried over) or "zones" (9 zones round-robin)."""
        if lap_dir not in LAP_DIRS:
            raise ValueError(f"lap_dir must be one of {LAP_DIRS}")
        if start_mode not in START_MODES:
            raise ValueError(f"start_mode must be one of {START_MODES}")
        if streak_obs and not armed_obs:
            raise ValueError("streak_obs needs armed_obs")
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        self.reward_mode = reward_mode
        self.budget = float(budget)
        self.rw = merge_reward_weights(reward_weights)
        if reward_mode == "budget":
            laps, net_terminate = 0, True
        if episode_sec is None:
            episode_sec = 120.0 if reward_mode == "budget" else 20.0
        self.ROOM = np.array(room, dtype=float)
        self.EPISODE_LEN_SEC = float(episode_sec)
        self.spec = spec
        self.ring_jitter = float(ring_jitter)
        self.radius_scale = float(hoop_radius_scale)
        self.laps = int(laps)
        self.net_terminate = bool(net_terminate)
        self.wind = float(wind)
        self.sway_deg = float(spec.sway_tilt_deg if sway_deg is None else sway_deg)
        self.carry_start = float(carry_start)
        self.reverse_prob = float(reverse_prob)
        self.lean_flip_prob = float(lean_flip_prob)
        self.wash_deg = float(wash_deg)
        self.start_z = None if start_z is None else float(start_z)
        self.armed_obs = bool(armed_obs)
        self.streak_obs = bool(streak_obs)
        self.lap_dir = lap_dir
        self.start_mode = start_mode
        self.start_zone = 0
        self._drone_pos = None
        self._has_landing = False      # no previous episode yet to carry the start from
        self._lean_flipped = False
        self._ring_ids = []
        self._rng = np.random.default_rng(seed)
        self._start_override = None if start_pos is None else np.asarray(start_pos, dtype=float)
        self._target_override = None if target_pos is None else np.asarray(target_pos, dtype=float)
        # zones: each env starts the round-robin at its own (seeded) zone
        self._zone_i = int(self._rng.integers(9)) if start_mode == "zones" else 0
        self._sample_ring(self._rng)
        self._reset_episode_state()
        start = self._sample_start(self._rng)
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
                         act=act)

    ################################################################################
    # randomisation

    @property
    def inner_radius(self):
        return self.spec.inner_radius * self.ring_r_factor * self.radius_scale

    def set_hoop_radius_scale(self, scale):
        """Curriculum hook (callable through VecEnv.env_method). Applies from the next step."""
        self.radius_scale = float(scale)

    def set_wind(self, scale):
        """Curriculum hook (callable through VecEnv.env_method): live wind strength, 0..1."""
        self.wind = float(scale)

    def set_variety(self, carry_start=None, reverse_prob=None, lean_flip_prob=None):
        """Curriculum hook (callable through VecEnv.env_method): change the episode mix."""
        if carry_start is not None:
            self.carry_start = float(carry_start)
        if reverse_prob is not None:
            self.reverse_prob = float(reverse_prob)
        if lean_flip_prob is not None:
            self.lean_flip_prob = float(lean_flip_prob)

    def set_wash_deg(self, deg):
        """Curriculum hook (callable through VecEnv.env_method): largest prop-wash tilt."""
        self.wash_deg = float(deg)

    def set_sway_deg(self, deg):
        """Curriculum hook (callable through VecEnv.env_method): largest swing about the top knot."""
        self.sway_deg = float(deg)

    def onboard_frame(self, res=(200, 150)):
        """The drone's camera frame (uint8 RGB). Callable through VecEnv.env_method, so the
        dashboard's live camera also works when this env runs in a SubprocVecEnv worker."""
        from camera_capture import get_onboard_frame
        return get_onboard_frame(self, nth_drone=0, res=res)

    @property
    def sway_top_amp(self):
        """Knot drift at full gust: scales with the swing, so the top always moves least."""
        s = self.spec
        return s.sway_top_amp * self.sway_deg / s.sway_tilt_deg if s.sway_tilt_deg else 0.0

    def ring_top(self, c, n):
        """Top point of a ring with centre c and normal n (where the rope is tied)."""
        return c + (self.inner_radius + self.spec.tube_radius) * ring_up(n)

    def _sample_ring(self, rng):
        s, j = self.spec, self.ring_jitter
        centre = np.array(s.centre, dtype=float)
        if self._target_override is not None:
            centre = self._target_override.copy()
        else:
            centre = centre + j * rng.uniform(-1, 1, 3) * np.array(s.centre_jitter)
        yaw = s.normal_yaw_deg + j * rng.uniform(-1, 1) * s.yaw_jitter_deg
        # sometimes the wind holds the hanging ring over the other way: mirror its lean
        self._lean_flipped = bool(rng.uniform() < self.lean_flip_prob)
        lean = -s.normal_tilt_deg if self._lean_flipped else s.normal_tilt_deg
        tilt = lean + j * rng.uniform(-1, 1) * s.tilt_jitter_deg
        self.ring_c0 = centre                 # this episode's resting pose
        self.ring_n0 = ring_normal(yaw, tilt)
        self.ring_r_factor = 1.0 + j * rng.uniform(-1, 1) * s.radius_jitter_frac
        # live wind for this episode
        phi = rng.uniform(0, 2 * np.pi)
        self._wind_dir = np.array([np.cos(phi), np.sin(phi), 0.0])
        self._wind_axis = np.array([-np.sin(phi), np.cos(phi), 0.0])   # horizontal, across the wind
        self._gust = rng.uniform(0, 1)
        self._sway_w = 2 * np.pi / rng.uniform(*s.sway_period_s)
        self._sway_phase = rng.uniform(0, 2 * np.pi)
        self._ring_t = 0.0
        self.push_off = np.zeros(3)
        self._push_vel = np.zeros(3)
        # prop wash: drawn only when it is on, so runs without it keep their random streams
        self._wash_k = float(rng.uniform(*s.wash_strength)) if self.wash_deg > 0 else 1.0
        self.wash_ang = 0.0           # deg, current tilt (sign: see _update_ring)
        self._wash_rate = 0.0
        self._drone_pos = None
        self._update_ring(0.0)
        self.ring_vel = np.zeros(3)

    def _update_ring(self, dt):
        """Advance the ring's live pose by dt seconds: wind sway + the damped shove-back.
        Also sets ring_vel, the centre's velocity over that step (kept as is when dt == 0)."""
        s, k = self.spec, self.wind
        prev_c = getattr(self, "ring_c", None)
        self._ring_t += dt
        if dt > 0 and k > 0:
            w, z = self._sway_w, s.push_damping
            acc = -w * w * self.push_off - 2 * z * w * self._push_vel
            self._push_vel = self._push_vel + acc * dt
            self.push_off = self.push_off + self._push_vel * dt
            n = np.linalg.norm(self.push_off)
            if n > s.push_max:
                self.push_off *= s.push_max / n
        swing = k * self._gust * np.sin(self._sway_w * self._ring_t + self._sway_phase)
        # the ring hangs from a rope at its top: the knot drifts a little downwind and the ring
        # swings about it (about the across-wind axis; the negative angle sends the bottom downwind)
        top0 = self.ring_top(self.ring_c0, self.ring_n0)
        pivot = top0 + swing * self.sway_top_amp * self._wind_dir
        ang = -np.radians(swing * self.sway_deg)
        # prop wash: a tilt about the horizontal in-plane axis h through the knot, which moves the
        # bottom along the normal. Positive wash_ang = bottom towards +n (away from a drone on -n).
        arm0 = self.ring_c0 - top0
        h = np.cross(self.ring_n0, [0.0, 0.0, 1.0])
        h = h / (np.linalg.norm(h) + 1e-12)
        if float(np.cross(h, arm0) @ self.ring_n0) < 0:
            h = -h
        if dt > 0 and self.wash_deg > 0 and self._drone_pos is not None:
            d = self._drone_pos - getattr(self, "ring_c", self.ring_c0)
            side = float(d @ self.ring_n0)
            # tanh, not sign: inside the opening the push has no side (no snap-back into the drone)
            target = (-np.tanh(side / s.wash_side_soft) * self.wash_deg * self._wash_k
                      * np.exp(-(np.linalg.norm(d) / s.wash_range) ** 2))
            w, z = self._sway_w, s.wash_damping
            self._wash_rate += (w * w * (target - self.wash_ang) - 2 * z * w * self._wash_rate) * dt
            self.wash_ang += self._wash_rate * dt
        wash = np.radians(self.wash_ang)
        self.ring_knot = pivot + k * self.push_off    # where the rope is tied (the shove moves it too)
        self.ring_c = self.ring_knot + rotate(rotate(arm0, h, wash), self._wind_axis, ang)
        self.ring_n = rotate(rotate(self.ring_n0, h, wash), self._wind_axis, ang)
        if dt > 0 and prev_c is not None:
            self.ring_vel = (self.ring_c - prev_c) / dt
        # kept so the dashboard / legacy code can read the "target" like HoopFindAviary
        self.TARGET_POS = self.ring_c

    def _sample_start(self, rng, landing=None):
        """On the floor, or hovering at start_z (+- jitter). A pinned start_pos wins; otherwise, with probability carry_start,
        where the previous episode ended (`landing`, as if the drone landed straight down there);
        otherwise anywhere on the floor, including the corners, beside and under the ring (its
        bottom is ~1 m up)."""
        half = self.ROOM[:2] / 2 - START_MARGIN
        self.carried_start = False
        self.start_zone = 0
        under = False
        if self._start_override is not None:
            xy = self._start_override[:2]
        elif self.start_mode == "zones":
            xy, under = self._zone_xy(rng, half)
        elif landing is not None and rng.uniform() < self.carry_start:
            xy = np.clip(np.asarray(landing, dtype=float)[:2], -half, half)
            self.carried_start = True
        else:
            xy = rng.uniform(-half, half)
        z = START_Z
        if self.start_z is not None:
            j = self.spec.start_z_jitter
            z = self.start_z + float(rng.uniform(-j, j))
            if under:                                  # keep clear below the ring's bottom
                bottom = self.ring_c0[2] - self.inner_radius - self.spec.tube_radius
                z = min(z, bottom - self.spec.drone_radius - 0.1)
        return np.array([xy[0], xy[1], z])

    def _zone_xy(self, rng, half):
        """The next zone of the round-robin: a random point in that cell (re-drawn if it is within
        0.25 m of the ring's centre), or under the hoop (zone 9). -> (xy, under)."""
        k = self._zone_i % 9
        self._zone_i += 1
        self.start_zone = k + 1
        if k == 8:
            xy = self.ring_c0[:2] + rng.uniform(-0.15, 0.15, 2)
            return np.clip(xy, -half, half), True
        w = 2 * half / 3
        col, row = ZONE_CELLS[k]
        centre = np.array([col, row]) * w
        for _ in range(20):
            xy = centre + rng.uniform(-w / 2, w / 2)
            if np.linalg.norm(xy - self.ring_c0[:2]) > 0.25:
                break
        return np.clip(xy, -half, half), False

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        self._rng = rng
        # where the episode that just finished left the drone (before the simulation is reset)
        landing = self.pos[0].copy() if self._has_landing else None
        self._sample_ring(rng)
        self._reset_episode_state()
        self._dir = -1.0 if rng.uniform() < self.reverse_prob else 1.0   # back -> front this time
        self._start_dir = self._dir
        self.INIT_XYZS = self._sample_start(rng, landing).reshape(1, 3)
        self._has_landing = True
        return super().reset(seed=seed, options=options)

    def _reset_episode_state(self):
        self._dir = 1.0              # required crossing direction along ring_n (flips each lap)
        self._start_dir = 1.0
        self.passes = 0
        self.net_contacts = 0
        self.rim_contacts = 0
        self.wrong_way = 0
        self._ev = dict(net=False, rim=False, passed=False, wrong=False, returned=False)
        self._reward = 0.0
        self._fresh = True
        self._prev_pos = None
        self._prev_s = 0.0
        self._prev_phase = 0
        self._prev_phi = 0.0
        self._armed = True           # the next forward crossing counts as a pass
        self._returned = True        # same-side laps: back round the outside since the last pass
        self._ret_side = 1.0         # ... on this side of the ring's axis (fixed at each pass)
        self.streak_mult = 1.0       # the multiplier the last pass was paid with
        self._net_end = False        # net_terminate: the net was touched this episode
        self.score = 0.0             # running total of this episode's reward
        self.budget_score = 0.0      # budget mode: the total without shaping (decides the end)
        self.phase = 0

    ################################################################################
    # scene (visual only -- the net and the ring have no PyBullet collision bodies)

    def _housekeeping(self):
        super()._housekeeping()
        self._fresh = True
        self._build_scene()

    def _build_scene(self):
        from court import build_net, build_ring
        build_net(self.CLIENT, self.ROOM, show_grid=self.GUI)
        self._ring_ids = build_ring(self.CLIENT, self.ring_c, self.ring_n, self.inner_radius,
                                    self.spec.tube_radius)

    def _move_ring_visual(self):
        if self.GUI and self._ring_ids:
            from court import move_ring
            move_ring(self.CLIENT, self._ring_ids, self.ring_c, self.ring_n,
                      self.inner_radius + self.spec.tube_radius)

    ################################################################################
    # observation

    def _observationSpace(self):
        base = super()._observationSpace()
        room_tile = np.tile(self.ROOM, (self.NUM_DRONES, 1))
        ones = np.ones((self.NUM_DRONES, 3))
        one = np.ones((self.NUM_DRONES, 1))
        k = int(self.armed_obs) + int(self.streak_obs)
        flag = [np.zeros((self.NUM_DRONES, k)), np.ones((self.NUM_DRONES, k))]
        lo = np.hstack([base.low, -ones, -2 * ones, 0 * one, flag[0], -room_tile])
        hi = np.hstack([base.high, ones, 2 * ones, 0.5 * one, flag[1], room_tile])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        base = super()._computeObs()
        pos = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        rel = self.ring_c - pos
        nd = self.NUM_DRONES
        normal = np.tile(self.ring_n * self._dir, (nd, 1))
        ring_vel = np.tile(self.ring_vel, (nd, 1))
        radius = np.full((nd, 1), self.inner_radius)
        extra = [np.full((nd, 1), float(self._armed))] if self.armed_obs else []
        if self.streak_obs:
            extra.append(np.full((nd, 1), min(self.passes, 4) / 4.0))
        return np.hstack([base, normal, ring_vel, radius, *extra, rel]).astype(np.float32)

    ################################################################################
    # contact, pass-through and reward bookkeeping (once per control step)

    def _updateAndStoreKinematicInformation(self):
        super()._updateAndStoreKinematicInformation()
        self._post_physics()

    def _set_state(self, pos, vel):
        i = self.DRONE_IDS[0]
        p.resetBasePositionAndOrientation(i, pos, self.quat[0], physicsClientId=self.CLIENT)
        p.resetBaseVelocity(i, vel, self.ang_v[0], physicsClientId=self.CLIENT)
        self.pos[0], self.vel[0] = pos, vel

    @staticmethod
    def _bounce(vel, normal, sp):
        """Reflect the velocity component going into a surface (normal points away from it)."""
        vn = float(vel @ normal)
        if vn < 0:
            vel = vel - vn * normal - sp.restitution * vn * normal
        vn = float(vel @ normal)
        if vn < sp.min_bounce_speed:
            vel = vel + (sp.min_bounce_speed - vn) * normal
        return vel

    def _post_physics(self):
        sp = self.spec
        if not hasattr(self, "_fresh"):
            return
        pos, vel = self.pos[0].copy(), self.vel[0].copy()
        ev = dict(net=False, rim=False, passed=False, wrong=False, returned=False)
        moved = False

        if not self._fresh:                   # the ring moves on by one control step
            self._drone_pos = pos
            self._update_ring(1.0 / self.CTRL_FREQ)
            if self.wind > 0 or self.wash_deg > 0:
                self._move_ring_visual()
        n0, c = self.ring_n, self.ring_c

        if self._fresh:                       # first update after a (re)set: just seed the state
            self._fresh = False
            d = pos - c
            phase = self._phase(pos, c, n0)
            self._prev_pos, self._prev_s, self._prev_phase = pos.copy(), float(d @ n0), phase
            self._prev_phi = self._potential(pos, c, n0, phase)
            self.phase = phase
            self._ev, self._reward = ev, 0.0
            return

        # ---- pass-through test (before any bounce moves the drone) ----
        # both ends measured against the ring's *current* pose, so a ring swinging past a still
        # drone is not a crossing and a moving ring cannot hide one
        d = pos - c
        s = float(d @ n0)
        prev_s = float((self._prev_pos - c) @ n0)
        crossed_fwd = (self._dir > 0 and prev_s < 0 <= s) or (self._dir < 0 and prev_s > 0 >= s)
        crossed_back = (self._dir > 0 and prev_s > 0 >= s) or (self._dir < 0 and prev_s < 0 <= s)
        r_ok = self.inner_radius - sp.drone_radius        # clear opening for the drone's centre
        r_cross = r_ok
        if crossed_fwd or crossed_back:
            t = prev_s / (prev_s - s)
            cp = self._prev_pos + t * (pos - self._prev_pos) - c
            r_cross = np.linalg.norm(cp - (cp @ n0) * n0)
            if r_cross < r_ok:
                if crossed_back:
                    ev["wrong"] = True
                elif self._armed:                 # laps: not until it has flown clear after the last pass
                    ev["passed"] = True
            elif (crossed_back and self.lap_dir == "same" and not self._returned
                  and r_cross > self.inner_radius + sp.tube_radius + sp.drone_radius):
                ev["returned"] = True             # back to the entry side round the outside

        # ---- net: soft walls on x, y and the top (the floor is the physical plane) ----
        lim = np.array([self.ROOM[0] / 2, self.ROOM[1] / 2, self.ROOM[2]]) - sp.net_margin
        for ax in range(3):
            for sign in (-1.0, 1.0):
                if ax == 2 and sign < 0:
                    continue
                if sign * pos[ax] > lim[ax]:
                    nrm = np.zeros(3)
                    nrm[ax] = -sign
                    pos[ax] = sign * lim[ax]
                    vel = self._bounce(vel, nrm, sp)
                    ev["net"], moved = True, True

        # ---- ring rim: distance to the tube's centre circle ----
        d = pos - c
        s = float(d @ n0)
        rv = d - s * n0
        r = np.linalg.norm(rv)
        u = rv / r if r > 1e-9 else np.cross(n0, [0.0, 0.0, 1.0])
        u = u / (np.linalg.norm(u) + 1e-12)
        q = c + (self.inner_radius + sp.tube_radius) * u
        vec = pos - q
        dist = np.linalg.norm(vec)
        if dist < sp.tube_radius + sp.drone_radius:
            nrm = vec / dist if dist > 1e-9 else n0
            pos = q + nrm * (sp.tube_radius + sp.drone_radius)
            v_in = -float(vel @ nrm)          # drone speed into the noodle shoves the ring away
            if v_in > 0 and self.wind > 0:
                self._push_vel = self._push_vel - sp.push_gain * v_in * nrm
            vel = self._bounce(vel, nrm, sp)
            ev["rim"], moved = True, True

        if moved:
            self._set_state(pos, vel)

        # ---- bookkeeping / reward ----
        self.net_contacts += ev["net"]
        self.rim_contacts += ev["rim"]
        self.wrong_way += ev["wrong"]
        d = pos - c
        if ev["returned"]:
            self._returned = True
        if (not self._armed and self._returned
                and float(d @ n0) * self._dir < -sp.arm_dist):
            self._armed = True                             # flown clear: the next pass counts
        phase = self._phase(pos, c, n0)
        phi = self._potential(pos, c, n0, phase)
        if ev["returned"]:
            self._prev_phi = phi                           # a new stage of the lap: no jump paid
        mult = 1.0
        if ev["passed"]:                                   # streak: pass k pays more (capped)
            w = self.rw
            mult = min(1.0 + w["budget_streak"] * self.passes, max(1.0, w["budget_streak_cap"]))
            self.streak_mult = mult
        if self.reward_mode == "budget":
            reward = self._budget_reward(ev)
            self.budget_score += reward
            if ev["passed"]:
                reward += self.rw["budget_pass"] * (mult - 1.0)   # kept out of the budget tally
            if self.rw["budget_shape"]:                    # MDP v4: potential-based shaping on top
                reward += self.rw["budget_shape"] * (phi - self._prev_phi)
        else:
            reward = self._shaped_reward(ev, phi, r_cross, r_ok)
            if ev["passed"]:
                reward += self.rw["pass_bonus"] * (mult - 1.0)
        if ev["passed"]:
            self.passes += 1
            if self.laps == 0 or self.passes < self.laps:  # lap mode: go again
                if self.lap_dir == "same":                 # same way in: round the outside first
                    self._returned = False
                    self._ret_side = self._pick_return_side(pos, c, n0)
                else:                                      # back-and-forth: turn round
                    self._dir = -self._dir
                self._armed = False
                phase = self._phase(pos, c, n0)
                phi = self._potential(pos, c, n0, phase)   # the shaping restarts for the new leg
        self.score += reward
        self.phase = phase
        self._prev_pos, self._prev_s, self._prev_phase = pos.copy(), float(d @ n0), phase
        self._prev_phi = phi
        self._ev, self._reward = ev, float(reward)

    def _budget_reward(self, ev):
        """MDP v3: time, passes and rim clips only. Touching the net ends the episode and
        costs whatever budget is left (at least BUDGET_NET), so it is never cheaper than waiting."""
        w = self.rw
        reward = -w["budget_time"]
        if ev["rim"]:
            reward -= w["budget_rim"]
        if ev["wrong"]:
            reward -= w["budget_wrong"]
        if ev["passed"]:
            reward += w["budget_pass"]
        if ev["net"]:
            self._net_end = True
            reward = min(reward - w["budget_net"], -self.budget - self.budget_score)
        return reward

    def _shaped_reward(self, ev, phi, r_cross, r_ok):
        """MDP v2: potential-based shaping plus pass bonus and penalties."""
        # potential-based shaping: summed over an episode it is phi_end - phi_start, so flying
        # in circles (or in and out of a phase) earns nothing
        w = self.rw
        reward = w["shaping_gain"] * (phi - self._prev_phi) - w["time_penalty"]
        if ev["net"]:
            if self.net_terminate:
                reward -= w["net_penalty"]
                self._net_end = True
            else:
                reward -= w["net_bounce_penalty"]
        if ev["rim"]:
            reward -= w["rim_penalty"]
        if ev["wrong"]:
            reward -= w["wrong_way_penalty"]
        if ev["passed"]:
            reward += w["pass_bonus"] + w["centre_bonus"] * max(0.0, 1.0 - r_cross / r_ok)
        return reward

    def _phase(self, pos, c, n0):
        """1 once the drone is lined up just in front of the opening (aim behind the ring), else 0
        (aim at the approach point in front). Always 0 until a lap has re-armed."""
        if not self._armed:
            return 0
        d = pos - c
        s_eff = float(d @ n0) * self._dir
        r_lat = np.linalg.norm(d - (d @ n0) * n0)
        return int(-0.45 < s_eff < 0.05 and r_lat < 1.2 * self.inner_radius)

    def _side_axis(self, n0):
        """Horizontal unit vector in the ring's plane."""
        u = np.cross(n0, [0.0, 0.0, 1.0])
        return u / (np.linalg.norm(u) + 1e-12)

    def _return_wp(self, c, n0):
        """Same-side laps: the point beside the ring, in its plane, that the way round goes through."""
        return c + self._ret_side * RETURN_SIDE_M * self._side_axis(n0)

    def _pick_return_side(self, pos, c, n0):
        """The side of the ring's axis the drone is on, unless that way round would be too close to
        the net; then the other side."""
        u = self._side_axis(n0)
        side = 1.0 if float((pos - c) @ u) >= 0 else -1.0
        lim = self.ROOM[:2] / 2 - self.spec.net_margin - 0.12
        for s in (side, -side):
            wp = c + s * RETURN_SIDE_M * u
            if np.all(np.abs(wp[:2]) <= lim):
                return s
        return side

    def _potential(self, pos, c, n0, phase):
        """Shaping potential (m, <= 0): minus the distance still to fly via the approach point in
        front of the ring to the one behind it, minus a pull onto the ring's axis near its plane.
        Same-side laps, before the drone is back round: via the point beside the ring first."""
        wp = self.spec.approach_dist
        n_eff = n0 * self._dir
        d = pos - c
        s_eff = float(d @ n_eff)
        r_lat = np.linalg.norm(d - (d @ n0) * n0)
        if self.lap_dir == "same" and not self._returned:
            side = self._return_wp(c, n0)
            phi = -(np.linalg.norm(pos - side) + np.linalg.norm(side - (c - n_eff * wp)) + 2 * wp)
            return float(phi)
        if phase == 0:
            phi = -(np.linalg.norm(pos - (c - n_eff * wp)) + 2 * wp)
        else:
            phi = -np.linalg.norm(pos - (c + n_eff * wp))
        return float(phi - self.rw["axis_pull"] * np.exp(-(s_eff / AXIS_PULL_LEN) ** 2) * r_lat)

    ################################################################################
    # reward / termination

    def _computeReward(self):
        return self._reward

    def _success(self):
        return self.passes >= max(1, self.laps)

    def _computeTerminated(self):
        if self._net_end:
            return True
        if self.reward_mode == "budget":
            return self.budget_score <= -self.budget + 1e-9
        return self.laps > 0 and self.passes >= self.laps

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[7]) > .5 or abs(state[8]) > .5:       # tipped over (~29 deg)
            return True
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {"passes": self.passes, "success": self._success(),
                "is_success": self._success(),                # read by SB3's EvalCallback
                "net_ended": bool(self._net_end), "score": float(self.score),
                "armed": bool(self._armed), "wash_tilt_deg": float(self.wash_ang),
                "lap_dir": self.lap_dir, "returned": bool(self._returned),
                "streak_mult": float(self.streak_mult), "start_zone": int(self.start_zone),
                "net_contact": bool(self._ev["net"]), "rim_contact": bool(self._ev["rim"]),
                "net_contacts": self.net_contacts, "rim_contacts": self.rim_contacts,
                "wrong_way": self.wrong_way, "phase": self.phase,
                "hoop_centre": self.ring_c.tolist(), "hoop_normal": (self.ring_n * self._dir).tolist(),
                "hoop_radius": float(self.inner_radius),
                "wind": float(self.wind * self._gust), "wind_scale": float(self.wind),
                "swing_deg": float(self.wind * self._gust * self.sway_deg),
                "swing_period_s": float(2 * np.pi / self._sway_w), "push_offset": float(np.linalg.norm(self.push_off)),
                "reversed": bool(self._start_dir < 0), "lean_flipped": bool(self._lean_flipped),
                "carried_start": bool(getattr(self, "carried_start", False))}


def env_kwargs_from_meta(meta: dict) -> dict:
    """HoopNetAviary constructor kwargs for the settings a run trained with (its meta.json), so a
    replay or showcase flies the same task. Missing keys (older runs) keep the defaults."""
    keys = ("sway_deg", "reverse_prob", "lean_flip_prob", "carry_start", "laps", "net_terminate",
            "episode_sec", "reward_mode", "budget", "reward_weights", "wash_deg", "start_z",
            "lap_dir", "start_mode")
    kw = {k: meta[k] for k in keys if meta.get(k) is not None}
    if meta.get("obs_dim") in (72 + OBS_EXTRA + 1, 72 + OBS_EXTRA + 2):   # the lap "armed" flag
        kw["armed_obs"] = True
    if meta.get("obs_dim") == 72 + OBS_EXTRA + 2:                        # ... and the streak
        kw["streak_obs"] = True
    return kw
