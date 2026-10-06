"""Small signal helpers used by the safety triggers and the estimator."""
import math
from collections import deque

import numpy as np


def wrap_pi(a):
    """Wrap an angle (rad) to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class MedianFilter:
    """Median of the last n valid samples. NaN / None samples are ignored."""

    def __init__(self, n=5):
        self.buf = deque(maxlen=n)

    def update(self, x):
        if x is not None and np.isfinite(x):
            self.buf.append(float(x))
        return self.value

    @property
    def value(self):
        return float(np.median(self.buf)) if self.buf else None

    def reset(self):
        self.buf.clear()


class EMA:
    def __init__(self, alpha=0.2, init=None):
        self.alpha = alpha
        self.value = init

    def update(self, x):
        if x is None or not np.isfinite(x):
            return self.value
        self.value = float(x) if self.value is None else self.value + self.alpha * (float(x) - self.value)
        return self.value

    def reset(self, init=None):
        self.value = init


class SmoothedSignal:
    """Median-of-n followed by an EMA. Used for height, temperature."""

    def __init__(self, n=5, alpha=0.3):
        self.med = MedianFilter(n)
        self.ema = EMA(alpha)

    def update(self, x):
        m = self.med.update(x)
        return self.ema.update(m) if m is not None else self.ema.value

    @property
    def value(self):
        return self.ema.value

    def reset(self):
        self.med.reset()
        self.ema.reset()


class BatteryFilter:
    """Tello battery readings jump and sag under load.

    Median of n, then treated as monotonically non-increasing while flying
    (running minimum): an upward spike is noise, a sag that recovers never
    lowers the estimate more than the true minimum of the medians.
    """

    def __init__(self, n=7):
        self.med = MedianFilter(n)
        self.value = None

    def update(self, x):
        m = self.med.update(x)
        if m is None:
            return self.value
        self.value = m if self.value is None else min(self.value, m)
        return self.value

    def reset(self):
        self.med.reset()
        self.value = None


class HoldTimer:
    """True once `cond` has been continuously true for `hold_s` seconds."""

    def __init__(self, hold_s):
        self.hold_s = hold_s
        self.t_start = None

    def update(self, cond, now):
        if not cond:
            self.t_start = None
            return False
        if self.t_start is None:
            self.t_start = now
        return (now - self.t_start) >= self.hold_s

    @property
    def active_since(self):
        return self.t_start

    def reset(self):
        self.t_start = None


class RollingFraction:
    """Fraction of True values within the last `window_s` seconds."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.buf = deque()

    def update(self, flag, now):
        self.buf.append((now, bool(flag)))
        while self.buf and now - self.buf[0][0] > self.window_s:
            self.buf.popleft()
        return self.fraction

    @property
    def fraction(self):
        if not self.buf:
            return 0.0
        return sum(f for _, f in self.buf) / len(self.buf)

    @property
    def span(self):
        return (self.buf[-1][0] - self.buf[0][0]) if len(self.buf) > 1 else 0.0

    def reset(self):
        self.buf.clear()
