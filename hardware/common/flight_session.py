"""
FlightSession: one 30 Hz control tick that wires everything together, for both the real Tello
and FakeTello. Control scripts (keyboard_teleop, run_policy, calibration) only decide *what
action to request*; takeoff gating, warm-up, geofence, safety, landing escalation, logging and
the observation action-buffer all happen here, in one place, so no script can skip them.

Phases:  ground -> takeoff -> warmup -> flying -> landing -> landed        (or killed)

Tick order mirrors a simulator step: build obs from current state (buffer holds *previous*
executed actions) -> choose action -> geofence -> safety -> map to rc -> send -> push the
executed action into the buffer.
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from state_estimation.estimator import StateEstimator
from .action_mapper import ActionMapper, RcCommand
from .constants import CTRL_DT, REACH_RADIUS
from .geofence import Geofence, action_to_velocity
from .logger import FlightLogger
from .obs_builder import ObsBuilder
from .safety import (LandingSequence, Level, SafetyDecision, SafetyInputs, SafetyMonitor,
                     preflight_problems)


@dataclass
class TickInfo:
    phase: str
    est: object
    obs: Optional[np.ndarray]
    policy_action: Optional[np.ndarray]
    executed_action: np.ndarray
    rc: RcCommand
    decision: SafetyDecision
    source: str            # 'policy' | 'manual' | 'hover' | 'none'
    reached: bool = False


class FlightSession:
    def __init__(self, cfg, backend, camera=None, hoop=None, logger: Optional[FlightLogger] = None,
                 speed_scale=1.0, real=None, require_calibrated_axes=True):
        """require_calibrated_axes: real takeoffs are refused until calibrate_axes.py has set
        conventions.axes_calibrated. The calibration scripts themselves pass False (they have to fly in
        order to measure the axes; they use raw rc hops and stay under the safety monitor)."""
        self.cfg = cfg
        self.backend = backend
        self.camera = camera
        self.real = backend.is_real if real is None else real
        self.require_calibrated_axes = require_calibrated_axes
        self.speed_scale = float(speed_scale)
        self.logger = logger or FlightLogger(enabled=False)
        self.est_ctor = StateEstimator(cfg, t0=backend.now())
        self.mapper = ActionMapper(cfg)
        self.fence = Geofence(cfg["fence"])
        self.monitor = SafetyMonitor(cfg)
        self.landing = LandingSequence(cfg)
        self.obs = ObsBuilder(hoop if hoop is not None else np.zeros(3), cfg["estimator"]["angvel_lpf_alpha"])
        self.hoop = None if hoop is None else np.asarray(hoop, dtype=float)
        self.phase = "ground"
        self._t_phase = backend.now()
        self._t_flight0 = None
        self._t_policy0 = None
        self._last_tick = None
        self._bat_samples = []
        self._reached_since = None
        self.last_est = None
        self.last_executed = np.zeros(4)
        self.problems = []

    # -- human commands -------------------------------------------------------------
    def request_takeoff(self) -> list:
        est = self.last_est or self.est_ctor.estimate(self.backend.now())
        if self.monitor.latched >= Level.LAND:
            self.problems = ["a LAND/KILL fault is latched this session; restart the script"]
            return self.problems
        self.problems = preflight_problems(self.cfg, est, self.real and self.require_calibrated_axes, self._bat_samples)
        if self.problems:
            self.logger.event("takeoff_refused", problems=self.problems)
            return self.problems
        self.logger.event("takeoff")
        self.backend.takeoff()
        self._set_phase("takeoff")
        self.obs.reset()
        self.last_executed = np.zeros(4)
        return []

    def request_land(self, reason="human_land"):
        self.monitor.request(Level.LAND, reason)

    def request_kill(self, reason="human_kill"):
        self.monitor.request(Level.KILL, reason)

    def toggle_hold(self):
        self.monitor.request(Level.HOLD, "human_hold")

    # -- internals ----------------------------------------------------------------------
    def _set_phase(self, p):
        self.phase = p
        self._t_phase = self.backend.now()
        self.logger.event("phase", phase=p)

    def _send(self, rc: RcCommand):
        self.backend.send_rc(*rc.as_tuple())

    def _begin_landing(self, now, reason):
        if self.phase in ("landing", "landed", "killed"):
            return
        self._set_phase("landing")
        self.logger.event("landing", reason=reason)
        for a in self.landing.start(now):
            getattr(self.backend, a)()

    # -- the tick -------------------------------------------------------------------------
    def tick(self, manual_action=None, policy_fn: Optional[Callable] = None, raw_rc=None) -> TickInfo:
        """raw_rc=(lr, fb, ud, yaw): calibration only. Bypasses the action mapper (so the mapper's
        assumed axis conventions are not in the loop), still clamped to rc_max and still supervised
        by the safety monitor. Ignored outside the flying phase."""
        be = self.backend
        be.advance(CTRL_DT)                        # simulated time moves here; real time moves by itself
        now = be.now()
        loop_dt = None if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        est_in = self.est_ctor
        raws = be.poll_telemetry()
        est_in.on_raw_telemetry(raws, now)
        dets = []
        if self.camera is not None:
            hint = est_in.height_hint()
            if hasattr(self.camera, "set_height_hint"):
                self.camera.set_height_hint(hint)
            dets = self.camera.poll(now)
            for det in dets:
                est_in.on_camera(det, now)
        est = est_in.estimate(now)
        self.last_est = est
        bat = est.fields.get("bat") if est.fields else None
        if bat is not None and self.phase == "ground":
            self._bat_samples.append(bat)
            self._bat_samples = self._bat_samples[-10:]

        airborne = self.phase in ("takeoff", "warmup", "flying", "landing")
        obs = None
        policy_action = None
        source = "none"
        executed = np.zeros(4)
        fence_res = None

        if self.phase == "takeoff":
            if est.pos[2] > 0.5 or now - self._t_phase > 10.0:
                self._set_phase("warmup")
                self.obs.reset()
        if self.phase == "warmup" and now - self._t_phase >= self.cfg["control"]["warmup_s"]:
            self._set_phase("flying")
            self._t_flight0 = self._t_flight0 or now
            self._t_policy0 = now

        if self.phase in ("takeoff", "warmup", "flying") and self._t_flight0 is None and self.phase != "takeoff":
            self._t_flight0 = now
        if self.phase == "takeoff" and self._t_flight0 is None:
            self._t_flight0 = now

        reached = False
        if self.phase == "flying" and est.ready:
            obs = self.obs.build(est.pos, est.rpy, est.vel, now)
            if manual_action is not None:
                source = "manual"
                act = np.asarray(manual_action, dtype=float).reshape(-1)[:4]
            elif policy_fn is not None:
                source = "policy"
                policy_action = np.asarray(policy_fn(obs), dtype=float).reshape(-1)[:4]
                act = policy_action
            else:
                source = "hover"
                act = np.zeros(4)
            finite = bool(np.all(np.isfinite(act)))
            act = np.nan_to_num(act)
            act = np.clip(act, -1.0, 1.0)
            ramp = 1.0 if self._t_policy0 is None else min(1.0, (now - self._t_policy0) / max(self.cfg["control"]["ramp_s"], 1e-3))
            act = act.copy()
            act[3] = act[3] * self.speed_scale * (ramp if source == "policy" else 1.0)
            fence_res = self.fence.apply(act, est.pos)
            executed = fence_res.action
            if self.hoop is not None and np.linalg.norm(self.hoop - est.pos) < REACH_RADIUS:
                reached = True
        else:
            finite = True

        decision = self.monitor.evaluate(SafetyInputs(
            now=now, est=est, airborne=airborne,
            flight_elapsed=0.0 if self._t_flight0 is None else now - self._t_flight0,
            fence_hard=bool(fence_res.hard) if fence_res else False,
            fence_clamped=bool(fence_res.clamped) if fence_res else False,
            action_finite=finite, loop_dt=loop_dt,
            axis_mismatch=(est_in.axis.update(now, action_to_velocity(executed), est_in.latency.cam_velocity)
                           if self.phase == "flying" else False),
            policy_active=(source == "policy")))

        # reaching the hoop = success: hover 1 s, then land (not a fault)
        if reached and source == "policy":
            self._reached_since = self._reached_since or now
            if now - self._reached_since > 1.0:
                self.request_land("reached_hoop")
        else:
            self._reached_since = None

        rc = RcCommand.zero()
        if decision.level >= Level.KILL and self.phase != "killed":
            self.logger.event("KILL", reasons=decision.reasons)
            be.emergency()
            self._set_phase("killed")
        elif decision.level >= Level.LAND and self.phase in ("takeoff", "warmup", "flying"):
            self._begin_landing(now, ",".join(decision.reasons))
        elif decision.level >= Level.LAND and self.phase == "ground":
            pass
        elif decision.level == Level.HOLD and self.phase == "flying":
            executed = np.zeros(4)
            source = "hover"

        if self.phase == "landing":
            vz = float(est.vel[2])
            for a in self.landing.update(now, est.z_smooth, vz):
                self.logger.event("landing_action", action=a, h=est.z_smooth)
                getattr(be, a)()
                if a == "emergency":
                    self._set_phase("killed")
            if self.landing.landed:
                self._set_phase("landed")
        elif self.phase in ("warmup", "flying", "takeoff"):
            if self.phase != "takeoff":
                if raw_rc is not None and self.phase == "flying" and decision.level == Level.NONE:
                    m = self.mapper.rc_max
                    rc = RcCommand(*[int(max(-m, min(m, c))) for c in raw_rc])
                    source = "raw_rc"
                else:
                    rc = self.mapper.to_rc(executed, est.yaw, est.yaw_rate,
                                           speed_scale=1.0)       # speed_scale already folded into the action
                self._send(rc)
                if self.phase == "flying":
                    self.obs.push_action(executed)
                    self.last_executed = executed

        info = TickInfo(self.phase, est, obs, policy_action, executed, rc, decision, source, reached)
        self.logger.tick(t=now, phase=self.phase, state=est.vector(), pos_std=est.pos_std,
                         pos_health=est.pos_health, yaw_health=est.yaw_health,
                         cam_age=est.cam_age, tele_age=est.tele_age, obs=None if obs is None else obs.reshape(-1),
                         policy_action=policy_action, executed=executed, rc=rc.as_tuple(), source=source,
                         level=int(decision.level), reasons=decision.reasons,
                         latency=est_in.latency.latency, bat=bat,
                         tele=raws, cam=[(d.t, d.x, d.y, d.yaw, d.quality) for d in dets])
        return info

    def close(self):
        try:
            self.backend.send_rc(0, 0, 0, 0)
        except Exception:
            pass
        self.logger.close()


def run_realtime(session: FlightSession, step: Callable[[], None], hz=30.0, stop: Callable[[], bool] = lambda: False):
    """Drive `step()` at `hz` against a monotonic clock (real hardware). `step` calls session.tick()."""
    dt = 1.0 / hz
    nxt = time.monotonic()
    while not stop():
        step()
        nxt += dt
        sleep = nxt - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            nxt = time.monotonic()                 # overran: do not try to catch up in a burst
