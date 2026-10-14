"""
HISTORY: the fix below now lives in onboard/sim.py (2026-10-13 05:00); fixed_crossing() is a no-op.

A proposed fix for a crossing bug in onboard/sim.py, applied to THIS PROCESS ONLY (a context
manager that patches OnboardSim and restores it), so it can be measured without touching sim.py
while training runs that loaded the original are still going.

The bug (found 2026-10-13 while testing the privileged teacher): _check_crossing measures the
drone's position before AND after the step against the ring's NEW pose. That was meant to stop "a
ring swinging past a still drone" reading as a pass. But at the moment of a real pass the prop wash
eases off and the ring swings back towards the drone by a few mm per tick; measured against the
moved ring, the "before" position lands on the same side as the "after" one, and a clean, centred
pass is never recorded. The drone then flies on behind the ring.

The fix: "before" is measured against the ring as it was before the step, "after" against the ring
as it is now (the relative crossing), and it counts only if the drone itself moved in the crossing
direction. A ring sweeping past a still or retreating drone is still not a crossing, as intended.
"""
from contextlib import contextmanager

import numpy as np

from onboard import sim as onboard_sim
from onboard.sim import OnboardSim, Outcome


def _update_ring_keep_previous(self):
    self._ring_before = self.ring
    _ORIG_UPDATE(self)


def _check_crossing_relative(self, prev_pos):
    ring_before = getattr(self, "_ring_before", None) or self.ring
    side = self._plane_side()
    prev_side = float(np.dot(prev_pos - ring_before.centre, ring_before.normal))
    moved = float(np.dot(self.pos - prev_pos, self.ring.normal))
    if (self.outcome.done or side == 0.0 or np.sign(side) == np.sign(prev_side)
            or np.sign(moved) != np.sign(side)):
        self._prev_side = side
        return
    denom = side - prev_side
    f = 0.0 if abs(denom) < 1e-12 else -prev_side / denom
    hit = prev_pos + f * (self.pos - prev_pos)
    radial = float(np.linalg.norm(
        (hit - self.ring.centre) - np.dot(hit - self.ring.centre, self.ring.normal) * self.ring.normal))
    if self.open_plane and radial > (self.ring.radius + self.ring.noodle_m + self.drone_radius_m):
        self._prev_side = side
        self.clear_cross = 1 if side > 0 else -1
        return
    r = self.drone_radius_m
    inner = max(self.ring.radius - r, 0.0)
    outer = self.ring.radius + self.ring.noodle_m + r
    self.outcome = Outcome(crossed=True, radial_m=radial, passed=radial <= inner,
                           struck=inner < radial <= outer, missed=radial > outer,
                           direction=1 if side > 0 else -1)
    self._prev_side = side


_ORIG_UPDATE = OnboardSim._update_ring
_ORIG_CHECK = OnboardSim._check_crossing


@contextmanager
def fixed_crossing(on: bool = True):
    """Since 2026-10-13 05:00 the fix is in onboard/sim.py itself, so this changes nothing; kept
    so older scripts (gates.py's "as is" / "fixed crossing" rows) still run. Both rows now agree."""
    yield


__all__ = ["fixed_crossing", "onboard_sim"]
