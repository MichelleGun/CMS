"""
FakeTello: a Tello stand-in for offline tests and `--dry-run`, plus a fake overhead camera.

It deliberately models the things that make the real drone differ from the simulator's
PID-controlled Crazyflie, so a policy that only works on `ideal` shows up before flight:

  - rc deadband, per-axis time constants, asymmetric accel vs decel (it brakes slower
    than it starts), saturation and an acceleration limit
  - command latency with jitter, occasional dropped rc commands
  - velocity measurement bias and integer quantisation; height quantised to whole cm
    with extra noise near the ground
  - slow yaw drift and a random-walk wind-like velocity drift
  - autonomous takeoff to ~0.8 m (not the sim's 0.1 m), land / emergency behaviour
  - battery that sags under load and steps down, temperature creep
  - telemetry at 10 Hz with delivery delay, bursts, duplicates and drops

Profiles: `ideal`, `typical`, `harsh`. Numbers are plausible placeholders: once
calibrate_speed.py has been run on a real drone, put the measured values into a profile.

Conventions implemented here are the same unverified ones as config.yaml (rc: lr right+,
fb forward+, ud up+, yaw clockwise+; body frame FLU internally). Set
`profile.axis_flip=True` to emulate a drone whose axes disagree with the config, which is
what the runtime axis-mismatch fault exists to catch.
"""
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import List

import numpy as np

from state_estimation.est_types import CameraDetection


class SimClock:
    def __init__(self, t0=1000.0):
        self.t = t0

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@dataclass
class FakeTelloProfile:
    name: str = "typical"
    true_rc_per_mps: float = 125.0
    rc_deadband: int = 3
    tau_xy_up: float = 0.35
    tau_xy_down: float = 0.55           # slower to stop than to start
    tau_z_up: float = 0.25
    tau_z_down: float = 0.30
    max_accel: float = 2.0
    cmd_latency: float = 0.10
    cmd_latency_jitter: float = 0.03
    cmd_loss_prob: float = 0.01
    vel_bias: float = 0.01              # m/s
    vel_noise: float = 0.015
    vel_unit_mps: float = 0.1           # raw velocity quantum -> m/s (matches config vel_to_mps)
    yaw_drift_dps: float = 0.3
    wind_rw: float = 0.01               # m/s per sqrt(s) random walk on velocity
    h_noise_cm: float = 1.0
    h_noise_ground_cm: float = 4.0
    takeoff_h: float = 0.8
    battery_start: float = 85.0
    battery_drain_pct_per_s: float = 0.02
    battery_sag_pct: float = 3.0
    tele_hz: float = 10.0
    tele_delay: float = 0.04
    tele_delay_jitter: float = 0.03
    tele_burst_prob: float = 0.06
    tele_dup_prob: float = 0.04
    tele_drop_prob: float = 0.03
    axis_flip: bool = False
    # overhead camera
    cam_hz: float = 30.0
    cam_latency: float = 0.08
    cam_latency_jitter: float = 0.01
    cam_noise_xy: float = 0.015
    cam_yaw_noise_deg: float = 2.0
    cam_dropout_prob: float = 0.03
    cam_occlusions: tuple = ()          # ((t_start, t_end), ...) in seconds since sim start


PROFILES = {
    "ideal": FakeTelloProfile(
        name="ideal", rc_deadband=0, tau_xy_up=0.2, tau_xy_down=0.2, tau_z_up=0.2, tau_z_down=0.2,
        cmd_latency=0.03, cmd_latency_jitter=0.0, cmd_loss_prob=0.0, vel_bias=0.0, vel_noise=0.0,
        yaw_drift_dps=0.0, wind_rw=0.0, h_noise_cm=0.0, h_noise_ground_cm=0.0, tele_delay=0.02,
        tele_delay_jitter=0.0, tele_burst_prob=0.0, tele_dup_prob=0.0, tele_drop_prob=0.0,
        cam_noise_xy=0.005, cam_yaw_noise_deg=0.5, cam_dropout_prob=0.0, true_rc_per_mps=128.0,
        battery_sag_pct=0.0, battery_drain_pct_per_s=0.0),
    "typical": FakeTelloProfile(name="typical"),
    "harsh": FakeTelloProfile(
        name="harsh", true_rc_per_mps=110.0, rc_deadband=5, tau_xy_up=0.5, tau_xy_down=0.9, tau_z_up=0.35,
        tau_z_down=0.5, cmd_latency=0.16, cmd_latency_jitter=0.06, cmd_loss_prob=0.04, vel_bias=0.03,
        vel_noise=0.04, yaw_drift_dps=1.5, wind_rw=0.04, h_noise_cm=2.0, h_noise_ground_cm=8.0,
        battery_sag_pct=8.0, tele_delay=0.07, tele_delay_jitter=0.06, tele_burst_prob=0.15,
        tele_dup_prob=0.1, tele_drop_prob=0.08, cam_noise_xy=0.03, cam_yaw_noise_deg=5.0,
        cam_dropout_prob=0.12, cam_latency=0.11, cam_latency_jitter=0.025),
}


def get_profile(name_or_profile, **overrides) -> FakeTelloProfile:
    p = PROFILES[name_or_profile] if isinstance(name_or_profile, str) else name_or_profile
    return replace(p, **overrides) if overrides else p


class FakeTello:
    """Simulated Tello backend. Stepped explicitly with advance(dt) on a SimClock."""
    is_real = False

    def __init__(self, cfg: dict, profile="typical", seed=0, clock: SimClock = None,
                 start_xy=(0.0, 0.0), **profile_overrides):
        self.cfg = cfg
        self.p = get_profile(profile, **profile_overrides)
        self.rng = np.random.default_rng(seed)
        self.clock = clock or SimClock()
        self.t_start = self.clock.now()
        conv = cfg["conventions"]
        self.vel_signs = np.array(conv["vel_signs"], dtype=float)
        self.att = conv["att_signs"]
        # true state (cage frame)
        self.pos = np.array([start_xy[0], start_xy[1], 0.0])
        self.v_body = np.zeros(3)             # FLU
        self.yaw = 0.0                        # rad CCW, cage frame (true heading)
        self.yaw_rate = 0.0
        self.tello_yaw0 = 0.0                 # Tello reports yaw relative to its takeoff heading
        self.wind = np.zeros(3)
        self.flying = False
        self.motors_off = True
        self.phase = "ground"                 # ground | takeoff | hover | landing | falling
        self._rc_queue = deque()              # (t_apply, rc tuple)
        self._rc = (0, 0, 0, 0)
        self._tele_out = deque()              # (t_deliver, packet)
        self._next_tele = self.clock.now()
        self._cmd_queue = deque()             # (t_apply, name)
        self._hist = deque(maxlen=400)
        self.battery = self.p.battery_start
        self.temp = 55.0
        self.motor_time = 0.0
        self.log = []                         # commands received, for tests
        self.connected = False
        self.crashed = False

    # -- backend API -----------------------------------------------------------
    def connect(self):
        self.connected = True

    def close(self):
        pass

    def now(self):
        return self.clock.now()

    def send_rc(self, lr, fb, ud, yaw):
        self.log.append(("rc", (lr, fb, ud, yaw)))
        if self.rng.random() < self.p.cmd_loss_prob:
            return
        lat = max(0.0, self.p.cmd_latency + self.rng.normal(0, self.p.cmd_latency_jitter))
        self._rc_queue.append((self.clock.now() + lat, (lr, fb, ud, yaw)))

    def takeoff(self):
        self.log.append(("takeoff", None))
        self._cmd_queue.append((self.clock.now() + 0.3, "takeoff"))

    def land(self):
        self.log.append(("land", None))
        self._cmd_queue.append((self.clock.now() + 0.2, "land"))

    def emergency(self):
        self.log.append(("emergency", None))
        self._cmd_queue.append((self.clock.now() + 0.05, "emergency"))

    def poll_telemetry(self) -> List[dict]:
        out = []
        now = self.clock.now()
        while self._tele_out and self._tele_out[0][0] <= now:
            out.append(self._tele_out.popleft()[1])
        return out

    def advance(self, dt):
        """Advance physics by dt seconds in 5 ms sub-steps."""
        n = max(1, int(round(dt / 0.005)))
        h = dt / n
        for _ in range(n):
            self._step(h)
            self.clock.advance(h)
            self._hist.append((self.clock.now(), self.pos.copy(), self.yaw))
            self._telemetry()

    # -- physics ------------------------------------------------------------------
    def _apply_commands(self):
        now = self.clock.now()
        while self._rc_queue and self._rc_queue[0][0] <= now:
            self._rc = self._rc_queue.popleft()[1]
        while self._cmd_queue and self._cmd_queue[0][0] <= now:
            _, name = self._cmd_queue.popleft()
            if name == "takeoff" and self.phase == "ground":
                self.phase, self.motors_off, self.flying = "takeoff", False, True
                self.tello_yaw0 = self.yaw
            elif name == "land" and self.phase in ("hover", "takeoff"):
                self.phase = "landing"
            elif name == "emergency":
                self.phase, self.motors_off = "falling", True

    def _step(self, dt):
        p = self.p
        self._apply_commands()
        g = 9.81
        if self.phase == "falling":
            self.v_body[2] -= g * dt
            self.pos += self._to_world(self.v_body) * dt
            if self.pos[2] <= 0:
                self.pos[2], self.v_body[:] = 0.0, 0.0
                self.phase, self.flying = "ground", False
            return
        if self.phase == "ground":
            return
        self.motor_time += dt
        self.battery -= p.battery_drain_pct_per_s * dt
        self.temp += 0.01 * dt
        target = np.zeros(3)
        yaw_cmd = 0.0
        if self.phase == "takeoff":
            target[2] = 0.5
            if self.pos[2] >= p.takeoff_h:
                self.phase = "hover"
            lr = fb = ud = 0
        elif self.phase == "landing":
            target[2] = -0.3
            if self.pos[2] <= 0.0:
                self.pos[2], self.v_body[:] = 0.0, 0.0
                self.phase, self.flying, self.motors_off = "ground", False, True
                return
        else:
            lr, fb, ud, yw = [0 if abs(c) < p.rc_deadband else c for c in self._rc]
            flip = -1.0 if p.axis_flip else 1.0
            target = np.array([fb, -lr * flip, ud], dtype=float) / p.true_rc_per_mps   # FLU
            yaw_cmd = -yw * math.radians(1.2)             # rc yaw is clockwise-positive; internal CCW
        # first-order response, slower when braking, with an acceleration limit
        for i in range(3):
            braking = abs(target[i]) < abs(self.v_body[i])
            if i == 2:
                tau = p.tau_z_down if braking else p.tau_z_up
            else:
                tau = p.tau_xy_down if braking else p.tau_xy_up
            a = (target[i] - self.v_body[i]) / max(tau, 1e-3)
            a = max(-p.max_accel, min(p.max_accel, a))
            self.v_body[i] += a * dt
        self.yaw_rate += (yaw_cmd - self.yaw_rate) / 0.3 * dt
        self.yaw += self.yaw_rate * dt
        self.wind += self.rng.normal(0, p.wind_rw * math.sqrt(dt), 3) * np.array([1, 1, 0.3])
        self.wind *= (1 - 0.2 * dt)
        v_world = self._to_world(self.v_body) + self.wind
        self.pos += v_world * dt
        if self.pos[2] < 0:
            self.pos[2] = 0.0
            self.v_body[2] = max(0.0, self.v_body[2])

    def _to_world(self, vb):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * vb[0] - s * vb[1], s * vb[0] + c * vb[1], vb[2]])

    # -- telemetry ------------------------------------------------------------------
    def _packet(self):
        p = self.p
        speed = float(np.linalg.norm(self.v_body))
        v = self.v_body + p.vel_bias + self.rng.normal(0, p.vel_noise, 3)
        raw = np.round(v / (self.vel_signs * p.vel_unit_mps))             # Tello raw, integer units
        z_cm = self.pos[2] * 100.0
        noise = p.h_noise_ground_cm if self.pos[2] < 0.3 else p.h_noise_cm
        h = int(round(z_cm + self.rng.normal(0, noise)))
        drift = math.radians(p.yaw_drift_dps) * (self.clock.now() - self.t_start)
        yaw_tello = (self.yaw - self.tello_yaw0 + drift)
        sag = p.battery_sag_pct * min(1.0, speed / 0.3)
        bat = int(max(0, min(100, round(self.battery - sag + self.rng.normal(0, 0.4)))))
        pitch = math.degrees(-self.v_body[0] * 0.05) * self.att["pitch"]  # small, plausible attitude
        roll = math.degrees(-self.v_body[1] * 0.05) * self.att["roll"]
        return {
            "pitch": pitch, "roll": roll,
            "yaw": math.degrees(yaw_tello) / self.att["yaw"] if self.att["yaw"] else 0.0,
            "vgx": float(raw[0]), "vgy": float(raw[1]), "vgz": float(raw[2]),
            "h": float(max(0, h)), "tof": float(max(0, h)), "bat": float(bat),
            "templ": 55.0, "temph": float(round(self.temp)), "baro": 0.0,
            "time": float(int(self.motor_time)), "agx": 0.0, "agy": 0.0, "agz": -1000.0,
        }

    def _telemetry(self):
        p = self.p
        now = self.clock.now()
        if self.motors_off and self.phase == "ground" and not self.connected:
            return
        if now < self._next_tele:
            return
        self._next_tele += 1.0 / p.tele_hz
        if self.rng.random() < p.tele_drop_prob:
            return
        delay = max(0.0, p.tele_delay + abs(self.rng.normal(0, p.tele_delay_jitter)))
        if self.rng.random() < p.tele_burst_prob:
            delay += 0.25                                 # held up in the network, arrives with its neighbours
        pkt = self._packet()
        self._tele_out.append((now + delay, pkt))
        if self.rng.random() < p.tele_dup_prob:
            self._tele_out.append((now + delay + 0.005, dict(pkt)))
        self._tele_out = deque(sorted(self._tele_out, key=lambda x: x[0]))

    # -- helpers for tests ---------------------------------------------------------------
    def truth_at(self, t):
        if not self._hist:
            return self.pos.copy(), self.yaw
        ts = [h[0] for h in self._hist]
        i = int(np.searchsorted(ts, t))
        i = min(max(i, 0), len(ts) - 1)
        return self._hist[i][1].copy(), self._hist[i][2]


class FakeOverheadCamera:
    """Produces CameraDetections from FakeTello's truth with latency, noise and dropouts."""

    def __init__(self, fake: FakeTello, seed=1):
        self.f = fake
        self.rng = np.random.default_rng(seed)
        self._next = fake.clock.now()
        self.p = fake.p

    def poll(self, now) -> List[CameraDetection]:
        out = []
        p = self.p
        while self._next <= now:
            t = self._next
            self._next += 1.0 / p.cam_hz
            if not self.f.flying and self.f.pos[2] < 0.02 and self.f.phase == "ground":
                pass                                       # marker is visible on the ground too
            rel = t - self.f.t_start
            if any(a <= rel <= b for a, b in p.cam_occlusions):
                continue
            if self.rng.random() < p.cam_dropout_prob:
                continue
            lat = max(0.0, p.cam_latency + self.rng.normal(0, p.cam_latency_jitter))
            pos, yaw = self.f.truth_at(t - lat)
            quality = float(np.clip(1.0 - 0.15 * pos[2] + self.rng.normal(0, 0.05), 0.3, 1.0))
            out.append(CameraDetection(
                t=t,
                x=float(pos[0] + self.rng.normal(0, p.cam_noise_xy / quality)),
                y=float(pos[1] + self.rng.normal(0, p.cam_noise_xy / quality)),
                yaw=float(yaw + math.radians(self.rng.normal(0, p.cam_yaw_noise_deg))),
                quality=quality))
        return out
