"""
Telemetry timestamp correction and packet sanity checks.

Tello state packets (UDP, ~10 Hz) carry no sequence number and arrive bursty,
late, and occasionally duplicated. This gate turns raw (arrival time, payload)
pairs into validated samples with a monotonic measurement time:

- only time.monotonic()-style clocks are used, never wall-clock time;
- exact duplicates (same payload, arrived within `dup_window_s`) are dropped;
- packets that arrive together in one poll (a burst) are spread backwards onto
  the nominal 100 ms grid ending at the arrival time, because the newest one
  was just measured and the others were queued in the network;
- measurement time is arrival time minus a fixed `delay_s`, forced strictly
  increasing, and never later than the arrival time;
- each packet is sanity-checked (parse, ranges, implied acceleration) and
  rejected packets are counted per reason, never silently used.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

NOMINAL_DT = 0.1
MIN_SPACING = 0.005          # s, smallest allowed gap between accepted samples


@dataclass
class TelemetrySample:
    t: float                  # corrected measurement time (monotonic clock)
    t_recv: float             # arrival time (monotonic clock)
    f: Dict[str, float]       # parsed numeric fields (raw Tello units)


@dataclass
class GateStats:
    accepted: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)

    def reject(self, why):
        self.rejected[why] = self.rejected.get(why, 0) + 1


REQUIRED = ("pitch", "roll", "yaw", "vgx", "vgy", "vgz", "h", "bat")
OPTIONAL = ("tof", "templ", "temph", "baro", "time", "agx", "agy", "agz")


def parse_fields(raw: dict) -> Optional[Dict[str, float]]:
    """Coerce a raw state dict (djitellopy's get_current_state or a parsed packet) to floats."""
    out = {}
    try:
        for k in REQUIRED:
            out[k] = float(raw[k])
        for k in OPTIONAL:
            if k in raw and raw[k] is not None:
                out[k] = float(raw[k])
    except (KeyError, TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for v in out.values()):
        return None
    return out


class TelemetryGate:
    def __init__(self, delay_s=0.05, vel_to_mps=0.1, dup_window_s=0.05,
                 max_speed_mps=3.0, max_accel_mps2=20.0, max_yaw_rate_dps=400.0):
        self.delay_s = delay_s
        self.vel_to_mps = vel_to_mps
        self.dup_window_s = dup_window_s
        self.max_speed = max_speed_mps
        self.max_accel = max_accel_mps2
        self.max_yaw_rate = max_yaw_rate_dps
        self.stats = GateStats()
        self._last_payload = None
        self._last_recv = None
        self._last_t = None
        self._last_ok: Optional[TelemetrySample] = None

    # ------------------------------------------------------------------
    def _sane(self, f, t) -> Optional[str]:
        if not (0 <= f["bat"] <= 100):
            return "battery_range"
        if abs(f["pitch"]) > 90 or abs(f["roll"]) > 90:
            return "attitude_range"
        if abs(f["yaw"]) > 360:
            return "yaw_range"
        if not (-50 <= f["h"] <= 1500):
            return "height_range"
        v = [f[k] * self.vel_to_mps for k in ("vgx", "vgy", "vgz")]
        if math.sqrt(sum(x * x for x in v)) > self.max_speed:
            return "speed_range"
        prev = self._last_ok
        if prev is not None:
            dt = t - prev.t
            if dt > 1e-3:
                pv = [prev.f[k] * self.vel_to_mps for k in ("vgx", "vgy", "vgz")]
                acc = math.sqrt(sum((a - b) ** 2 for a, b in zip(v, pv))) / dt
                if acc > self.max_accel:
                    return "accel_jump"
                dyaw = abs(((f["yaw"] - prev.f["yaw"]) + 180) % 360 - 180) / dt
                if dyaw > self.max_yaw_rate:
                    return "yaw_jump"
        return None

    # ------------------------------------------------------------------
    def process_batch(self, raws: List[dict], t_recv: float) -> List[TelemetrySample]:
        """Packets returned by a single poll, all arriving at `t_recv`."""
        out: List[TelemetrySample] = []
        n = len(raws)
        for i, raw in enumerate(raws):
            f = parse_fields(raw) if isinstance(raw, dict) else None
            if f is None:
                self.stats.reject("parse")
                continue
            payload = tuple(sorted(f.items()))
            if (payload == self._last_payload and self._last_recv is not None
                    and t_recv - self._last_recv < self.dup_window_s):
                self.stats.reject("duplicate")
                continue
            self._last_payload = payload
            self._last_recv = t_recv

            # spread a burst backwards over the nominal grid; newest = arrival
            t = t_recv - (n - 1 - i) * NOMINAL_DT - self.delay_s
            t = min(t, t_recv)
            if self._last_t is not None:
                if t < self._last_t + MIN_SPACING:
                    # enforce a real gap: a float-epsilon spacing would blow up any d/dt downstream
                    t = self._last_t + MIN_SPACING
                    if t > t_recv:           # would put the sample in the future
                        self.stats.reject("out_of_order")
                        continue
            why = self._sane(f, t)
            if why:
                self.stats.reject(why)
                continue
            s = TelemetrySample(t=t, t_recv=t_recv, f=f)
            self._last_t = t
            self._last_ok = s
            self.stats.accepted += 1
            out.append(s)
        return out

    def age(self, now: float) -> float:
        """Seconds since the last accepted sample arrived; inf if none yet."""
        return float("inf") if self._last_ok is None else now - self._last_ok.t_recv

    @property
    def last(self) -> Optional[TelemetrySample]:
        return self._last_ok
