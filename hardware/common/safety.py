"""
Safety layer. Every control script goes through it; the keyboard layer cannot bypass it.

Levels (a fault asks for at least this much):
    HOLD  zero rc, hover. Released automatically if the cause clears (or by the human)
    LAND  safe land. Latched for the rest of the flight
    KILL  emergency motor cut. Latched. The drone falls. Only a human triggers this above
          ~0.5 m; the automatic escalation in LandingSequence is height-gated.

A HOLD-level fault that lasts `hold_to_land_s` is promoted to LAND, so a stuck
hover can never be the end state.

SafetyMonitor is pure (inputs in, decision out, caller supplies the clock), so the
whole table of triggers is unit-tested without a drone.
"""
import queue
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np

from .smoothing import BatteryFilter, HoldTimer, RollingFraction, SmoothedSignal


class Level(IntEnum):
    NONE = 0
    HOLD = 1
    LAND = 2
    KILL = 3


@dataclass
class SafetyInputs:
    now: float
    est: object                       # StateEstimate
    airborne: bool
    flight_elapsed: float = 0.0
    fence_hard: bool = False
    fence_clamped: bool = False
    action_finite: bool = True
    loop_dt: Optional[float] = None
    axis_mismatch: bool = False
    policy_active: bool = False


@dataclass
class SafetyDecision:
    level: Level
    reasons: List[str] = field(default_factory=list)
    latched: bool = False


class SafetyMonitor:
    def __init__(self, cfg: dict):
        s = cfg["safety"]
        f = cfg["fence"]
        self.s = s
        self.max_flight_s = float(cfg["control"]["max_flight_s"])
        self.battery = BatteryFilter(7)
        self.temp = SmoothedSignal(5, 0.3)
        self._bat_hold = HoldTimer(s["battery_land_hold_s"])
        self._temp_hold = HoldTimer(s["temp_hold_s"])
        self._crit_count = 0
        self._overruns = 0
        self._fight = RollingFraction(f["fight_window_s"])
        self._fight_frac = float(f["fight_fraction"])
        self._fight_win = float(f["fight_window_s"])
        self._hold_since = {}
        self.latched = Level.NONE
        self.latched_reasons: List[str] = []
        self._human_hold = False
        self._human_events = queue.SimpleQueue()

    # -- human inputs (thread-safe: hotkey listener threads call these) ---------
    def request(self, level: Level, reason: str):
        self._human_events.put((level, reason))

    def release_human_hold(self):
        self._human_hold = False

    def _drain_human(self):
        faults = []
        while True:
            try:
                level, reason = self._human_events.get_nowait()
            except queue.Empty:
                break
            if level >= Level.LAND:
                self._latch(level, reason)
            elif level == Level.HOLD:
                self._human_hold = not self._human_hold      # SPACE toggles hold
        if self._human_hold:
            faults.append((Level.HOLD, "human_hold"))
        return faults

    def _latch(self, level, reason):
        if level > self.latched:
            self.latched = level
        if reason not in self.latched_reasons:
            self.latched_reasons.append(reason)

    # -- evaluation ---------------------------------------------------------------
    def evaluate(self, inp: SafetyInputs) -> SafetyDecision:
        now, est, s = inp.now, inp.est, self.s
        faults = self._drain_human()          # (level, reason)

        if inp.airborne:
            if est.tele_age > s["telemetry_stale_land_s"]:
                faults.append((Level.LAND, "telemetry_lost"))
            elif est.tele_age > s["telemetry_stale_hold_s"]:
                faults.append((Level.HOLD, "telemetry_stale"))
            if est.pos_health == "LOST":
                faults.append((Level.HOLD, "estimator_lost"))
            if est.yaw_health == "LOST":
                faults.append((Level.HOLD, "yaw_lost"))
            if inp.fence_hard:
                faults.append((Level.LAND, "fence_hard"))
            if max(abs(est.roll), abs(est.pitch)) > s["tilt_limit_rad"]:
                faults.append((Level.LAND, "tilt"))
            if inp.flight_elapsed > self.max_flight_s:
                faults.append((Level.LAND, "time_cap"))
            if not inp.action_finite:
                faults.append((Level.HOLD, "bad_action"))
            if inp.axis_mismatch:
                faults.append((Level.HOLD, "axis_mismatch"))
            if inp.loop_dt is not None:
                self._overruns = self._overruns + 1 if inp.loop_dt > s["overrun_s"] else 0
                if self._overruns >= s["overrun_count"]:
                    faults.append((Level.HOLD, "loop_overrun"))
            if inp.policy_active:
                frac = self._fight.update(inp.fence_clamped, now)
                if self._fight.span >= 0.8 * self._fight_win and frac > self._fight_frac:
                    faults.append((Level.HOLD, "policy_fights_fence"))
            else:
                self._fight.reset()

        # battery / temperature are checked on the ground too, but only ever *land* a flying drone
        bat_raw = est.fields.get("bat") if est.fields else None
        bat_s = self.battery.update(bat_raw)
        if bat_raw is not None:
            self._crit_count = self._crit_count + 1 if bat_raw < s["battery_critical_pct"] else 0
        if inp.airborne:
            if self._bat_hold.update(bat_s is not None and bat_s < s["battery_land_pct"], now):
                faults.append((Level.LAND, "battery_low"))
            if self._crit_count >= 2:
                faults.append((Level.LAND, "battery_critical"))
            temp_raw = est.fields.get("temph") if est.fields else None
            temp_s = self.temp.update(temp_raw)
            if self._temp_hold.update(temp_s is not None and temp_s > s["temp_land_c"], now):
                faults.append((Level.LAND, "overheating"))

        # promote persistent HOLD faults to LAND
        active_hold = {r for l, r in faults if l == Level.HOLD and r != "human_hold"}
        for r in list(self._hold_since):
            if r not in active_hold:
                del self._hold_since[r]
        for r in active_hold:
            self._hold_since.setdefault(r, now)
            if now - self._hold_since[r] >= s["hold_to_land_s"]:
                faults.append((Level.LAND, f"{r}_persisted"))

        for lvl, reason in faults:
            if lvl >= Level.LAND:
                self._latch(lvl, reason)

        level = max([self.latched] + [l for l, _ in faults], default=Level.NONE)
        reasons = list(self.latched_reasons) + [r for l, r in faults if l == Level.HOLD]
        return SafetyDecision(level=Level(level), reasons=reasons, latched=self.latched >= Level.LAND)


class LandingSequence:
    """Safe landing with height smoothing and a time-based fallback.

    t=0       land
    every land_reissue_s while still airborne: land again
    landed    : smoothed height < landed_h_m and flat for landed_window_s (held landed_hold_s)
    emergency (motor cut), only ever in these cases:
        height reliable:   t >= emergency_low_s and h < emergency_low_m and the descent has STALLED
                           (a drone still coming down is never cut), or
                           t >= emergency_max_s and h < emergency_max_m
        height unreliable: t >= emergency_blind_s (time alone governs)
    Above emergency_max_m the sequence never cuts motors on its own; only a human KILL can.
    """

    def __init__(self, cfg: dict):
        s = cfg["safety"]
        self.reissue = s["land_reissue_s"]
        self.low_s, self.low_m = s["land_emergency_low_s"], s["emergency_low_m"]
        self.max_s, self.max_m = s["land_emergency_max_s"], s["emergency_max_m"]
        self.blind_s = s["land_emergency_blind_s"]
        self.landed_h, self.landed_hold = s["landed_h_m"], s["landed_hold_s"]
        self.landed_range, self.landed_window = s["landed_range_m"], s["landed_window_s"]
        self._landed_timer = HoldTimer(self.landed_hold)
        self._h_hist = deque()
        self.t0 = None
        self._last_land = None
        self.landed = False
        self.emergency_sent = False

    @property
    def active(self):
        return self.t0 is not None

    def start(self, now) -> List[str]:
        if self.t0 is not None:
            return []
        self.t0 = now
        self._last_land = now
        return ["land"]

    def _observe(self, now, h):
        """Feed the smoothed height every tick; returns (flat, stalled).

        flat    : max-min of the smoothed height over landed_window_s is small. Used (with h < landed_h) to
                  declare 'landed'. Height, not the filter's vertical velocity: that velocity is noisy exactly
                  where it matters, near the ground.
        stalled : the drone has not descended 3 cm over the last second, so it is not making progress.
        """
        self._h_hist.append((now, h))
        while self._h_hist and now - self._h_hist[0][0] > 1.0:
            self._h_hist.popleft()
        hist = list(self._h_hist)
        flat = stalled = False
        if now - hist[0][0] >= 0.9 * self.landed_window:
            win = [x for t_, x in hist if now - t_ <= self.landed_window]
            flat = (max(win) - min(win)) < self.landed_range
        if now - hist[0][0] >= 0.9:
            stalled = (hist[0][1] - hist[-1][1]) < 0.03
        return flat, stalled

    def update(self, now, h_smooth: Optional[float], vz: float = 0.0) -> List[str]:
        """`vz` is accepted for API compatibility and deliberately unused (see _observe)."""
        if self.t0 is None or self.landed or self.emergency_sent:
            return []
        t = now - self.t0
        reliable = h_smooth is not None and np.isfinite(h_smooth)
        flat = stalled = False
        if reliable:
            flat, stalled = self._observe(now, h_smooth)
        else:
            self._h_hist.clear()
            self._landed_timer.reset()
        if reliable and self._landed_timer.update(h_smooth < self.landed_h and flat, now):
            self.landed = True
            return []
        acts = []
        if reliable:
            low_and_stuck = t >= self.low_s and h_smooth < self.low_m and stalled
            very_late = t >= self.max_s and h_smooth < self.max_m
            if low_and_stuck or very_late:
                self.emergency_sent = True
                return ["emergency"]
        elif t >= self.blind_s:
            self.emergency_sent = True
            return ["emergency"]
        if now - self._last_land >= self.reissue:
            self._last_land = now
            acts.append("land")
        return acts


def preflight_problems(cfg: dict, est, real_hardware: bool, battery_samples: List[float]) -> List[str]:
    """Reasons a takeoff must be refused. Empty list = OK."""
    probs = []
    if real_hardware and not cfg["conventions"].get("axes_calibrated", False):
        probs.append("axes not calibrated: run calibration/calibrate_axes.py (sets conventions.axes_calibrated)")
    if not est.ready:
        probs.append("state estimate not ready (need telemetry + overhead camera + yaw reference)")
    if est.pos_health != "OK":
        probs.append(f"position health {est.pos_health}")
    if est.yaw_health != "OK":
        probs.append(f"yaw health {est.yaw_health}")
    if len(battery_samples) < 5:
        probs.append("need 5 battery samples before takeoff")
    elif float(np.mean(battery_samples[-5:])) < cfg["control"]["min_takeoff_battery"]:
        probs.append(f"battery below {cfg['control']['min_takeoff_battery']} %")
    return probs


class GlobalKillSwitch:
    """System-wide hotkeys (work when the pygame window has lost focus). Needs `pynput`.

        Ctrl+Shift+X  EMERGENCY motor cut
        Ctrl+Shift+L  safe land
        Ctrl+Shift+H  toggle hold
    If pynput is unavailable the in-window keys still work; a warning is printed.
    """

    def __init__(self, monitor: SafetyMonitor):
        self.monitor = monitor
        self.listener = None
        self.active = False

    def start(self):
        try:
            from pynput import keyboard
        except Exception:
            print("[safety] WARNING: pynput not installed, global kill hotkey disabled "
                  "(window keys still work). pip install pynput")
            return False
        self.listener = keyboard.GlobalHotKeys({
            "<ctrl>+<shift>+x": lambda: self.monitor.request(Level.KILL, "hotkey_kill"),
            "<ctrl>+<shift>+l": lambda: self.monitor.request(Level.LAND, "hotkey_land"),
            "<ctrl>+<shift>+h": lambda: self.monitor.request(Level.HOLD, "hotkey_hold"),
        })
        self.listener.daemon = True
        self.listener.start()
        self.active = True
        return True

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.active = False
