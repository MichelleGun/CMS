"""
RL environment that mirrors the real hardware cage: a 2x2x2 m net with a physical
vertical ring near its centre. The drone always starts on the floor, must take off,
find its way to the ring and pass through it in the required direction.

Compared with hoop_env.HoopFindAviary (a point to reach, no net, no ring geometry):
- the ring has an opening, a tube, a normal (facing) and is slightly re-posed every
  episode (position, yaw/tilt of its face, opening size) to cover repositioning between
  flights;
- the ring is a light pool noodle hung from a rope tied to its top, so it also moves *during*
  an episode: in a per-episode wind it swings like a pendulum about that top knot (the knot
  barely drifts, the bottom swings furthest) and is shoved when the drone clips its rim,
  then swings back. The observation always holds the ring's live pose;
- success is a *pass through the opening* in the +normal direction, not proximity;
- the net is a soft boundary: touching it bounces the drone back (small penalty)
  and the episode continues; so does clipping the ring's rim;
- the drone always starts on the floor (like the real Tello takeoff).

Observation (78-D, KIN):
    [kin12, last-15-actions x4 (60), ring_normal (3), hoop_centre - drone_pos (3)]
Same layout as hoop_env's 75-D vector with the ring normal inserted before the final
relative vector, so `obs[-3:]` is still "vector to the hoop" for the telemetry code.

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
XY_MARGIN = 0.25                   # keep ring / start this far from the side nets
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
                 sway_deg: float = None):
        """start_pos: optional (x, y[, z]) -- only x/y are used, the drone always starts on the
        floor. target_pos: optional ring centre override (jitter on the centre is then off).
        ring_jitter: 0 disables per-episode ring jitter, 1 = full. hoop_radius_scale > 1 widens
        the opening (training curriculum). wind: 0 keeps the ring still within an episode (no
        sway, no shove), 1 = full live wind. sway_deg: largest swing of the ring about its top knot at
        full gust (deg, default spec.sway_tilt_deg). laps > 1 flips the required crossing direction after
        each pass, so the drone has to fly round and re-enter."""
        self.ROOM = np.array(room, dtype=float)
        self.EPISODE_LEN_SEC = 20
        self.spec = spec
        self.ring_jitter = float(ring_jitter)
        self.radius_scale = float(hoop_radius_scale)
        self.laps = int(laps)
        self.wind = float(wind)
        self.sway_deg = float(spec.sway_tilt_deg if sway_deg is None else sway_deg)
        self._ring_ids = []
        self._rng = np.random.default_rng(seed)
        self._start_override = None if start_pos is None else np.asarray(start_pos, dtype=float)
        self._target_override = None if target_pos is None else np.asarray(target_pos, dtype=float)
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

    def set_sway_deg(self, deg):
        """Curriculum hook (callable through VecEnv.env_method): largest swing about the top knot."""
        self.sway_deg = float(deg)

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
        tilt = s.normal_tilt_deg + j * rng.uniform(-1, 1) * s.tilt_jitter_deg
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
        self._update_ring(0.0)

    def _update_ring(self, dt):
        """Advance the ring's live pose by dt seconds: wind sway + the damped shove-back."""
        s, k = self.spec, self.wind
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
        self.ring_knot = pivot + k * self.push_off    # where the rope is tied (the shove moves it too)
        self.ring_c = self.ring_knot + rotate(self.ring_c0 - top0, self._wind_axis, ang)
        self.ring_n = rotate(self.ring_n0, self._wind_axis, ang)
        # kept so the dashboard / legacy code can read the "target" like HoopFindAviary
        self.TARGET_POS = self.ring_c

    def _sample_start(self, rng):
        if self._start_override is not None:
            xy = self._start_override[:2]
        else:
            half = self.ROOM[:2] / 2 - XY_MARGIN
            for _ in range(100):
                xy = rng.uniform(-half, half)
                if np.linalg.norm(xy - self.ring_c0[:2]) > 0.25:   # not under the ring
                    break
        return np.array([xy[0], xy[1], START_Z])

    def reset(self, seed=None, options=None):
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        self._rng = rng
        self._sample_ring(rng)
        self._reset_episode_state()
        self.INIT_XYZS = self._sample_start(rng).reshape(1, 3)
        return super().reset(seed=seed, options=options)

    def _reset_episode_state(self):
        self._dir = 1.0              # required crossing direction along ring_n (flips each lap)
        self.passes = 0
        self.net_contacts = 0
        self.rim_contacts = 0
        self.wrong_way = 0
        self._ev = dict(net=False, rim=False, passed=False, wrong=False)
        self._reward = 0.0
        self._fresh = True
        self._prev_pos = None
        self._prev_s = 0.0
        self._prev_phase = 0
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
        lo = np.hstack([base.low, -ones, -room_tile])
        hi = np.hstack([base.high, ones, room_tile])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _computeObs(self):
        base = super()._computeObs()
        pos = np.array([self._getDroneStateVector(i)[0:3] for i in range(self.NUM_DRONES)])
        rel = self.ring_c - pos
        normal = np.tile(self.ring_n * self._dir, (self.NUM_DRONES, 1))
        return np.hstack([base, normal, rel]).astype(np.float32)

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
        ev = dict(net=False, rim=False, passed=False, wrong=False)
        moved = False

        if not self._fresh:                   # the ring moves on by one control step
            self._update_ring(1.0 / self.CTRL_FREQ)
            if self.wind > 0:
                self._move_ring_visual()
        n0, c = self.ring_n, self.ring_c

        if self._fresh:                       # first update after a (re)set: just seed the state
            self._fresh = False
            d = pos - c
            self._prev_pos, self._prev_s, self._prev_phase = pos.copy(), float(d @ n0), 0
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
        if crossed_fwd or crossed_back:
            t = prev_s / (prev_s - s)
            cp = self._prev_pos + t * (pos - self._prev_pos) - c
            r_cross = np.linalg.norm(cp - (cp @ n0) * n0)
            if r_cross < self.inner_radius - sp.drone_radius:
                ev["passed" if crossed_fwd else "wrong"] = True

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
        n_eff = n0 * self._dir
        s_eff = float(d @ n_eff)
        r_lat = np.linalg.norm(d - (d @ n0) * n0)
        wp = sp.approach_dist
        target = c - n_eff * wp if self._prev_phase == 0 else c + n_eff * wp
        progress = np.linalg.norm(self._prev_pos - target) - np.linalg.norm(pos - target)
        reward = 10.0 * progress - 0.005
        if -0.6 < s_eff < 0:
            reward -= 0.05 * r_lat                         # line up with the opening
        if ev["net"] or ev["rim"]:
            reward -= 1.0
        if ev["wrong"]:
            reward -= 5.0
        phase = int(-0.45 < s_eff < 0.05 and r_lat < 1.2 * self.inner_radius)
        if ev["passed"]:
            reward += 20.0
            self.passes += 1
            if self.passes < self.laps:                    # lap mode: now come back through
                self._dir = -self._dir
                phase = 0
        self.phase = phase
        self._prev_pos, self._prev_s, self._prev_phase = pos.copy(), float(d @ n0), phase
        self._ev, self._reward = ev, float(reward)

    ################################################################################
    # reward / termination

    def _computeReward(self):
        return self._reward

    def _computeTerminated(self):
        return self.passes >= self.laps

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if abs(state[7]) > .5 or abs(state[8]) > .5:       # tipped over (~29 deg)
            return True
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {"passes": self.passes, "success": self.passes >= self.laps,
                "is_success": self.passes >= self.laps,       # read by SB3's EvalCallback
                "net_contact": bool(self._ev["net"]), "rim_contact": bool(self._ev["rim"]),
                "net_contacts": self.net_contacts, "rim_contacts": self.rim_contacts,
                "wrong_way": self.wrong_way, "phase": self.phase,
                "hoop_centre": self.ring_c.tolist(), "hoop_normal": (self.ring_n * self._dir).tolist(),
                "hoop_radius": float(self.inner_radius),
                "wind": float(self.wind * self._gust), "push_offset": float(np.linalg.norm(self.push_off))}
