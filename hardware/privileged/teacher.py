"""
The privileged teacher: flies the simulator from its TRUE state, so it can say what the right
action is in any state a camera-only student ends up in (DAgger, Loquercio et al. 2021).

    from privileged.teacher import PrivilegedTeacher
    act = PrivilegedTeacher()(env)          # env = onboard.gym_env.OnboardHoopEnv

Why not onboard/expert.py: that expert only sees the 37 camera numbers and runs a phase machine,
so it cannot label an arbitrary state, and on the measured dynamics it passes 3 % (its small yaw
corrections fall inside the real yaw dead zone). This one:

  * reads only the true state: drone position, velocity and heading; the ring's (swinging) centre
    and normal; the lap bookkeeping the reward already keeps (gym_env: _dir, _armed, _returned).
    The only "memory" it touches is the last executed action, which the student also sees (the
    action history in its observation), used to limit how fast the sticks move;
  * aims for the same points the reward's potential does (gym_env._potential): the approach point
    in front of the ring, then the exit point behind it; round the outside for same-side laps;
  * never sends a stick the drone would ignore: commands inside a dead zone are either zero or
    lifted just past it (the yaw dead zone is ~6 rc on the real Tello);
  * damps with the TRUE velocity, so delay and lag (unknown exactly) cost accuracy, not stability.

Gains are module constants, picked once on the measured profile by gates.py and then frozen.
"""
import math
from dataclasses import dataclass

import numpy as np

from onboard import spec
from onboard.gym_env import APPROACH_M, RETURN_SIDE_M
from onboard.sim import CAGE_XY

NOMINAL_MPS = spec.RC_MAX / 125.0          # sim.RC_PER_MPS: m/s at full stick, nominally


@dataclass
class Gains:
    kp: float = 0.8               # 1/s: desired speed per metre of position error
    kd: float = 0.6               # lag compensation on the velocity error
    v_far: float = 0.22           # m/s cruise
    v_near: float = 0.12          # m/s within NEAR_M of the ring plane
    near_m: float = 0.6
    kp_z: float = 1.5
    yaw_kp: float = 1.0 / 40.0    # action per degree of heading error
    yaw_tol_deg: float = 4.0      # inside this, no yaw command at all
    yaw_dead: float = 0.20        # the real Tello ignores |yaw| below ~6 rc = 0.2
    lift_margin: float = 0.04         # 0.20 + 0.04 = 0.24: past the dead zone, inside one 0.25 step
    dead: float = 0.08            # the shared dead zone of every stick (sim Profile.deadband)
    max_step: float = 0.25        # per-tick change of any stick
    side_clear_m: float = 0.10    # "on the wrong side" once this far past the plane
    kp_lat: float = 1.5           # 1/s: sideways speed per metre off the axis, in the funnel
    v_lat_max: float = 0.10
    funnel0: float = 0.025        # m off-axis allowed at the plane ...
    funnel_slope: float = 0.25    # ... plus this per metre in front of it
    funnel_len: float = 0.6       # the cone starts this far in front of the plane
    funnel_tol: float = 0.02      # entering it, a little looser than the forward-motion limit
    funnel_yaw_deg: float = 20.0  # and only facing within this of the way through
    through_len: float = 0.45     # keep pushing through until this far past the plane
    clear_after_m: float = 0.35   # laps: after a pass, fly out this far before going round
    out_first_m: float = 0.20     # laps: first straight out of the opening this far
    cage_margin_m: float = 0.20   # waypoints stay this far inside the net
    retreat_band_m: float = 0.20  # back away from the plane first when this close to it ...
    retreat_off_m: float = 0.45   # ... and this close to the ring's axis
    return_side_m: float = 0.55   # laps: how far beside the ring the way round goes (gym_env RETURN_SIDE_M)
    v_back: float = 0.04          # m/s backing off when too far off-axis right at the plane


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def _lift(a, dead, margin):
    """A command the drone would ignore becomes either nothing or just enough."""
    if a == 0.0:
        return 0.0
    if abs(a) < dead * 0.5:
        return 0.0
    if abs(a) < dead + margin:
        return math.copysign(dead + margin, a)
    return a


def _snap(v, prev, dead, step):
    """After the rate limit: a value inside a dead zone (which the drone would ignore) moves to 0
    or to the dead zone's edge, whichever is reachable within one step and nearest to v."""
    if v == 0.0 or abs(v) >= dead:
        return v
    options = [c for c in (0.0, math.copysign(dead, v), -math.copysign(dead, v)) if abs(c - prev) <= step + 1e-9]
    return min(options, key=lambda c: abs(c - v)) if options else v


class PrivilegedTeacher:
    def __init__(self, gains: Gains = None):
        self.g = gains or Gains()

    def reset(self):
        """Nothing to clear: the teacher is stateless. Here so it fits wherever a policy does."""

    # -- where to go -------------------------------------------------------------------------
    def frame(self, env):
        """Ring frame from the true state: centre c, pass direction p (unit, the way the next pass
        must go), signed distance along p (< 0 = in front of the plane), sideways offset vector."""
        sim = env.sim
        c, n = sim.ring.centre, sim.ring.normal
        axial, _ = env._axial_radial()
        p = n * env._leg_dir(axial)
        o = sim.pos - c
        a = float(np.dot(o, p))
        lat = o - np.dot(o, n) * n
        return c, p, a, lat

    def phase(self, env):
        g, sim = self.g, env.sim
        c, p, a, lat = self.frame(env)
        if env.lap_dir == "same" and not env._returned:
            side = env._return_wp()
            if np.linalg.norm((sim.pos - side)[:2]) > 0.15 and a >= -0.05:
                return "return"
            return "approach"
        if a > g.side_clear_m:
            # past the plane but still in the opening's footprint: the prop wash tilts the ring
            # AWAY as the drone nears, so the geometry says "past" before any crossing happened.
            # Turning round here swerves into the noodle; keep pushing straight through instead.
            if a < g.through_len and float(np.linalg.norm(lat)) < sim.ring.radius:
                return "funnel"
            # Only the drone's OWN motion across the ring's current plane counts as a crossing,
            # and with the plane closed (front runway) any crossing outside the opening is a miss.
            # Getting here without a crossing means the ring swung / was blown past the drone, so
            # The only legal way back is backwards through the opening (a "blind pass" costs
            # nothing but time), so: the funnel in mirror image, then approach again.
            return "around" if sim.open_plane else "reverse"
        # the funnel: a cone round the axis, narrow at the plane; enter it only aligned with the
        # axis and facing the way through. Anywhere else, the approach point 0.5 m in front of the
        # ring (always a move AWAY from the plane, so never a diagonal pass at the noodle)
        off = float(np.linalg.norm(lat))
        inside = ((a > 0.0 and off < sim.ring.radius)
                  or (-g.funnel_len < a <= 0.0 and off <= g.funnel0 + g.funnel_slope * (-a) + g.funnel_tol))
        if env._armed and inside and abs(self.heading_error(env, phase="funnel")) < g.funnel_yaw_deg:
            return "funnel"
        return "approach"

    # -- how to get there ----------------------------------------------------------------------
    def desired_velocity(self, env, phase):
        g, sim = self.g, env.sim
        c, p, a, lat = self.frame(env)

        def go_to(pt, vmax):
            pt = np.array(pt, dtype=float)
            pt[:2] = np.clip(pt[:2], -(CAGE_XY - g.cage_margin_m), CAGE_XY - g.cage_margin_m)   # never into the net
            v = g.kp * (pt - sim.pos)
            sp = float(np.linalg.norm(v))
            return v * (vmax / sp) if sp > vmax else v

        if phase == "return":
            # round the outside in legs that each move AWAY from the noodle: diagonally out to a
            # point behind and beside the ring, then forward to the point beside it in its plane
            # (crossing the plane there, clear of the noodle, is what arms the next lap), then on
            # to the approach point. Heading straight for the in-plane point from just behind the
            # opening slides along the plane into the noodle.
            u = env._side_axis() * env._ret_side
            off = float(np.linalg.norm(lat))
            if a < g.out_first_m and off < g.return_side_m - 0.10:
                return p * g.v_near                        # 1. straight out of the opening
            if off < g.return_side_m - 0.10:               # 2. diagonally out and beside
                behind = c + p * g.clear_after_m + u * g.return_side_m
                behind[2] = c[2]
                return go_to(behind, g.v_far)
            return go_to(env._return_wp(), g.v_far)
        if phase == "around":
            u = env._side_axis()
            sgn = self._roomy_side(env, c, u)
            side = c + sgn * RETURN_SIDE_M * u
            side[2] = c[2]
            if np.linalg.norm((sim.pos - side)[:2]) > 0.15:
                return go_to(side, g.v_far)
            return go_to(c - p * 0.5, g.v_far)
        if phase == "approach":
            if -g.retreat_band_m < a <= 0.05 and float(np.linalg.norm(lat)) < g.retreat_off_m:
                # beside the ring and nearly in its plane (just came round the outside): back
                # straight away from the plane first. Heading for the approach point from here
                # slides ALONG the plane into the noodle while the lag catches up.
                return -p * g.v_near
            return go_to(c - p * 0.5, g.v_far)
        # funnel: slide onto the axis (sideways AND height: lat is 3-D) while moving through only
        # as fast as the alignment allows; the 8 cm margin leaves no room for a diagonal pass.
        # "reverse" is the same funnel run backwards, from behind the ring.
        if phase == "reverse":
            p, a = -p, -a
        off = float(np.linalg.norm(lat))
        v_lat = -g.kp_lat * lat
        sp = float(np.linalg.norm(v_lat))
        if sp > g.v_lat_max:
            v_lat *= g.v_lat_max / sp
        allow = g.funnel0 + g.funnel_slope * max(0.0, -a)
        if a > 0.0:
            v_ax = g.v_near                                    # through the plane: keep going
        elif off <= allow:
            v_ax = g.v_near * (1.0 - 0.5 * off / max(allow, 1e-6))
        else:
            v_ax = 0.0                     # too far off-axis for this distance: line up first
                                           # (never back off: that crossed the plane backwards)
        return v_lat + p * v_ax

    def __call__(self, env, obs=None):
        g, sim = self.g, env.sim
        phase = self.phase(env)
        v_des = self.desired_velocity(env, phase)
        # velocity setpoint with lag compensation from the TRUE velocity
        v_cmd = v_des + g.kd * (v_des - sim.vel)
        fb = float(np.dot(v_cmd, sim.forward)) / NOMINAL_MPS
        lr = float(np.dot(v_cmd, sim.right)) / NOMINAL_MPS
        ud = float(v_cmd[2]) / NOMINAL_MPS

        e = self.heading_error(env)
        yaw = 0.0 if abs(e) < g.yaw_tol_deg else float(np.clip(-g.yaw_kp * e, -1.0, 1.0))
        yaw = _lift(yaw, g.yaw_dead, g.lift_margin)

        a = np.clip(np.array([lr, fb, ud, yaw]), -1.0, 1.0)
        for i in (spec.ACT_LR, spec.ACT_FB, spec.ACT_UD):
            a[i] = _lift(float(a[i]), g.dead, 0.02)
        # smooth: the last executed action is part of the observation the student sees too
        prev = np.asarray(env.builder._acts[-1], dtype=float)
        a = prev + np.clip(a - prev, -g.max_step, g.max_step)
        # a rate-limited step can land back inside a dead zone: snap to 0 or its edge, within reach
        a[spec.ACT_YAW] = _snap(float(a[spec.ACT_YAW]), float(prev[spec.ACT_YAW]), g.yaw_dead, g.max_step)
        for i in (spec.ACT_LR, spec.ACT_FB, spec.ACT_UD):
            a[i] = _snap(float(a[i]), float(prev[i]), g.dead, g.max_step)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _roomy_side(env, c, u):
        """+1 / -1: the side of the ring to go round, the drone's own side unless that way runs
        into the net (the same rule as gym_env._pick_return_side)."""
        side = 1.0 if float(np.dot(env.sim.pos - c, u)) >= 0 else -1.0
        lim = CAGE_XY - 0.17
        for s_ in (side, -side):
            wp = c + s_ * RETURN_SIDE_M * u
            if abs(wp[0]) <= lim and abs(wp[1]) <= lim:
                return s_
        return side

    def heading_error(self, env, phase=None):
        """Degrees to turn (+ = counter-clockwise) to look along the ring's axis, the way through:
        the camera then keeps the ring in view on the way in and goes through camera first.
        Looking at the ring's CENTRE instead swings wildly right next to it (a first version
        crossed backwards that way)."""
        c, p, a, lat = self.frame(env)
        sim = env.sim
        look = c if (phase or self.phase(env)) == "return" else c + p * 1.5
        to = (look - sim.pos)[:2]
        if np.linalg.norm(to) < 1e-6:
            return 0.0
        return _wrap(math.degrees(math.atan2(to[1], to[0])) - sim.yaw_deg)

    def explain(self, env):
        """(phase, heading error deg, axial m, off-axis m) for printing."""
        c, p, a, lat = self.frame(env)
        return self.phase(env), self.heading_error(env), a, float(np.linalg.norm(lat))
