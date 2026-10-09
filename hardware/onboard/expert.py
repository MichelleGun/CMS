"""
The scripted visual-servo controller: a hand-written expert that flies through the ring.

It serves three purposes at once:
  1. it is the thing that actually flies at stages 2-4;
  2. it generates the demonstrations that behaviour cloning trains on;
  3. it is the baseline a learned policy has to beat.

Phases:  SEARCH -> ALIGN -> APPROACH -> COMMIT -> DONE        (ABORT on repeated loss)

Two control ideas drive the design, and both are easy to get wrong.

1. One camera cannot tell "my nose is pointed off to the side of the ring" from "I am
   level with the ring but standing off to one side of its axis". Both show up as the
   same horizontal offset cx. They need different corrections -- yaw for the first,
   strafe for the second -- so cx alone cannot drive both. The resolution is to use a
   second, independent measurement: a circle seen off its own axis projects to an
   ellipse, so `axis_ratio` (minor/major) says how far off the ring's axis the drone is,
   while cx says where the nose is pointing. Hence:
       yaw  <- cx          (point the nose at the ring)
       lr   <- axis_ratio  (sidestep until the ring looks circular, i.e. on its axis)
   The ellipse's shape gives no sign, only magnitude -- it cannot say WHICH side the
   drone is on -- so the lateral correction is extremum seeking: strafe one way, and if
   axis_ratio got worse, reverse. Crude, but it converges and needs no extra sensor.

2. COMMIT is open-loop and that is unavoidable. The detector cannot see the ring once it
   fills the frame (its outline leaves the field of view, so too little arc remains to
   fit), so the last half metre is flown blind on a fixed forward push. Everything
   before COMMIT exists to make that blind push start from a good enough position. Its
   duration is computed from the last measured range and an assumed forward speed, not
   hardcoded, because a fixed tick count silently becomes wrong the moment the rc-to-
   speed scale is recalibrated.
"""
import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import spec
from .ring_detect import RingObs

SEARCH, ALIGN, APPROACH, COMMIT, DONE, ABORT = (
    "search", "align", "approach", "commit", "done", "abort")


@dataclass
class ExpertConfig:
    # --- gains (action units per unit of normalised error) ---
    k_yaw: float = 0.90          # cx -> yaw: point the nose at the ring
    k_ud: float = 0.80           # cy -> up/down (cy is +down, so this is negated)
    k_fb: float = 2.50           # range error -> forward. High because the usable range
                                 # error is small: the measured cage allows only ~1.1 m
                                 # of standoff and commit starts at 0.42 m, so a low gain
                                 # crawls and never arrives.
    lr_seek: float = 0.45        # strafe amplitude while seeking the ring's axis

    # --- channel limits ---
    yaw_max: float = 0.70
    lr_max: float = 0.55
    ud_max: float = 0.55
    fb_max: float = 0.80

    # --- phase thresholds, in METRES of real offset at the measured range ---
    # Image offsets alone are not thresholds: cx = 0.08 is 2.2 cm of lateral error at
    # 0.42 m but 9 cm at 1.6 m. Since what matters is "will the drone fit through a 30 cm
    # hole", these are metres and spec.offset_metres does the conversion. Expressed in
    # image units instead, the commit gate was so strict (2.8 cm) that the approach sat
    # on its knife edge -- which also made the expert almost impossible to clone, because
    # behaviour cloning averaged across the discontinuity instead of picking a side.
    #
    # These two are PHASE LABELS ONLY. __call__ picks the APPROACH/ALIGN name from
    # _aligned() and then runs _servo() either way, so they do not gate translation; the
    # gates that do are forward_misalign_m, commit_* and approach_target_m. They are kept
    # a little LOOSER than the commit gate so the phase ladder actually has three rungs:
    # ALIGN -> APPROACH (within 5 cm) -> COMMIT (within 4 cm). Set tighter than the commit
    # gate they would be reached only once the drone could already commit, and the
    # APPROACH label would never be observed -- which is what the demos and the flight
    # logs are read as reporting.
    align_lateral_m: float = 0.05    # under this counts as aligned
    align_vertical_m: float = 0.05
    seek_axis_ratio: float = 0.80   # below this, sidestep to get onto the ring's axis
    seek_period: int = 6         # ticks between re-deciding the sidestep direction
    seek_improve: float = 0.004  # axis_ratio gain per period that counts as progress

    # --- approach ---
    # The forward setpoint is deliberately CLOSER than commit_range_m. If the two were
    # equal, the proportional forward term would decay to zero exactly at the commit
    # threshold and the drone would asymptote just outside it forever, committing only
    # when noise happened to nudge the estimate across. (Measured: that bug scored 0/30
    # on the noise-free profile and 14/30 on the noisy one -- better sensors made it
    # worse, which is the signature of this mistake.)
    approach_target_m: float = 0.28

    # Over the last this-many metres before commit_range_m, the forward command is
    # blended smoothly up to commit_fb, so crossing into COMMIT is not a step change.
    # Two reasons, and the second is the one that forced it:
    #   * flight quality -- a sudden throttle jump at the commit boundary overshoots the
    #     alignment it just achieved;
    #   * learnability -- the commit ENTRY state occurs exactly once per episode (about
    #     40 samples in 6000), while the continuation occurs ~45 times per episode. A
    #     cloned policy therefore learned to continue a push but never to start one, and
    #     stalled at ~0.5 m every time. Measured: 1/12 traversals before this ramp. A
    #     ramp puts the same information in every tick of the approach instead.
    approach_blend_m: float = 0.22

    # --- commit ---
    # Start the blind push from here. NOT set to where the ring stops being visible
    # (0.40 m: at the measured 42 cm outer diameter and a 70 deg HFOV, that is exactly
    # where the ring becomes tangent to a 4:3 frame, top and bottom). Committing at the
    # edge of detectability means deciding from the one measurement that is just starting
    # to clip; measured, 0.42 m beats both 0.40 and 0.45 because the ring is still whole
    # when the decision is made.
    commit_range_m: float = 0.42
    # Must be this well lined up to commit, in metres. The MEASURED 30 cm opening has a
    # 15 cm radius and an unguarded Tello is about 14 cm wide, so the clearance from the
    # drone's centre to the noodle is only 8 cm (spec.RING_CLEARANCE_M) -- less than the
    # 10 cm this gate used to allow, i.e. the old value committed to a push it could not
    # complete. Half the clearance is spent here and half left for drift during the blind
    # push. Not tighter: at 3 cm the drone cannot converge inside the gate at all, loses
    # lock and aborts (measured 6/30 with 16 aborts on the harsh profile, against 19/30
    # with 2 aborts at 4 cm). The failure mode of over-tightening is an abort, not a
    # strike, so it is safe -- but it looks like a broken controller.
    commit_lateral_m: float = 0.04
    commit_vertical_m: float = 0.04
    commit_axis_ratio: float = 0.55   # and this square-on to the ring's plane
    commit_fb: float = 1.00
    commit_speed_mps: float = 0.24   # MEASURE at stage 3: ground speed at commit_fb.
                                     # Default assumes rc 30 at ~125 rc per m/s.
    commit_margin_m: float = 0.20    # extra distance, to be sure the plane is crossed.
                                     # 0.20 not 0.30: the far side of the measured cage
                                     # has only 90 cm of runway, and 0.20 scored
                                     # identically in simulation.
    commit_ticks_min: int = 20
    commit_ticks_max: int = 75        # 5 s cap: a blind push must be bounded
    settle_ticks: int = 8             # hover after the push before declaring DONE

    # --- search ---
    search_yaw: float = 0.45
    search_ud: float = 0.25       # gentle climb while scanning (ring is above takeoff height)
    search_sweep_ticks: int = 45  # ticks per half-sweep (~3 s each side at 15 Hz)
    search_max_ticks: int = 150   # 10 s of scanning before giving up

    # --- robustness ---
    max_aborts: int = 3           # losses of lock tolerated before ABORT
    forward_misalign_m: float = 0.10      # lateral error above this zeroes forward, and
                                          # forward is scaled down linearly approaching
                                          # it. This IS a functional gate (unlike
                                          # align_lateral_m) and it is set to the 8 cm
                                          # clearance plus a little, so the drone never
                                          # drives in from outside the error budget.
    forward_axis_ratio_min: float = 0.40  # nor drive in while this far off-axis

    # --- staged bring-up (see the plan's stage table) ---
    allow_forward: bool = True    # stage 2 sets False: align only, never translate
    allow_commit: bool = True     # stage 3 sets False: approach but never push through

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class ExpertInfo:
    phase: str = SEARCH
    reason: str = ""
    commit_tick: int = 0
    commit_target: int = 0
    aborts: int = 0
    range_m: float = float("inf")
    seek_dir: float = 0.0


class Expert:
    """Stateful. One instance per flight; call reset() before each attempt."""

    def __init__(self, cfg: Optional[ExpertConfig] = None,
                 hfov_deg: float = spec.CAMERA_HFOV_DEG):
        self.cfg = cfg or ExpertConfig()
        self.hfov_deg = float(hfov_deg)
        self.reset()

    def reset(self):
        self.phase = SEARCH
        self._search_ticks = 0
        self._commit_tick = 0
        self._commit_target = 0
        self._settle_tick = 0
        self._aborts = 0
        self._lost = 0
        self._search_dir = 1.0
        self._search_sweep_tick = 0
        self._search_leg = self.cfg.search_sweep_ticks   # first leg only; see __call__
        self._seek_dir = 1.0
        self._seek_tick = 0
        self._seek_ref = None
        self._last_range = float("inf")
        self.commit_clamped = False
        self.reason = ""

    # -- helpers ----------------------------------------------------------------------
    def _errors_m(self, ring: RingObs):
        """(lateral, vertical) offset of the ring from straight ahead, in metres."""
        return (spec.offset_metres(ring.cx, ring.range_m, self.hfov_deg),
                spec.offset_metres(ring.cy, ring.range_m, self.hfov_deg))

    def _aligned(self, ring: RingObs) -> bool:
        c = self.cfg
        lat, vert = self._errors_m(ring)
        return lat <= c.align_lateral_m and vert <= c.align_vertical_m

    def _can_commit(self, ring: RingObs) -> bool:
        c = self.cfg
        if not c.allow_commit or not c.allow_forward:
            return False
        if not np.isfinite(ring.range_m) or ring.range_m > c.commit_range_m:
            return False
        lat, vert = self._errors_m(ring)
        return (lat <= c.commit_lateral_m and vert <= c.commit_vertical_m
                and ring.axis_ratio >= c.commit_axis_ratio)

    def _commit_ticks_for(self, range_m: float) -> int:
        """How long to push blind, from the last measured range.

        Derived rather than hardcoded so that recalibrating commit_speed_mps (which is
        what stage 3 measures) automatically corrects the push instead of silently
        leaving it wrong.
        """
        c = self.cfg
        per_tick = max(c.commit_speed_mps * spec.CTRL_DT, 1e-4)
        want = (range_m + c.commit_margin_m) / per_tick
        ticks = int(np.clip(round(want), c.commit_ticks_min, c.commit_ticks_max))
        # If the cap binds, the push provably cannot cover the distance and the drone
        # will stop short. Worth saying out loud, because the cause is a commit_speed_mps
        # / commit_range_m combination that does not fit inside the 5 s bound, which
        # looks from the outside like a controller that simply misses.
        self.commit_clamped = want > c.commit_ticks_max
        if self.commit_clamped:
            print(f"[expert] commit push clamped to {ticks} ticks "
                  f"({ticks * spec.CTRL_DT:.1f} s): covering {range_m:.2f} m at "
                  f"{c.commit_speed_mps:.2f} m/s would need {want * spec.CTRL_DT:.1f} s. "
                  f"It will stop short.")
        return ticks

    def _seek_lr(self, ring: RingObs) -> float:
        """Extremum seeking on axis_ratio: sidestep onto the ring's axis.

        The ellipse gives no sign, so the direction is found by trying one and reversing
        if axis_ratio did not improve over a whole period. Only the direction is
        searched; the amplitude is fixed.
        """
        c = self.cfg
        if ring.axis_ratio >= c.seek_axis_ratio:
            self._seek_ref = None
            return 0.0
        self._seek_tick += 1
        if self._seek_ref is None:
            self._seek_ref = ring.axis_ratio
            self._seek_tick = 0
        elif self._seek_tick >= c.seek_period:
            if ring.axis_ratio - self._seek_ref < c.seek_improve:
                self._seek_dir *= -1.0
            self._seek_ref = ring.axis_ratio
            self._seek_tick = 0
        return float(np.clip(c.lr_seek * self._seek_dir, -c.lr_max, c.lr_max))

    def _servo(self, ring: RingObs) -> np.ndarray:
        c = self.cfg
        a = np.zeros(spec.ACT_DIM)
        a[spec.ACT_YAW] = float(np.clip(c.k_yaw * ring.cx, -c.yaw_max, c.yaw_max))
        # cy is positive DOWN in image coordinates, so a ring below centre needs a
        # downward command.
        a[spec.ACT_UD] = float(np.clip(-c.k_ud * ring.cy, -c.ud_max, c.ud_max))
        a[spec.ACT_LR] = self._seek_lr(ring)

        if c.allow_forward:
            rng = ring.range_m if np.isfinite(ring.range_m) else spec.RANGE_NORM
            fb = float(np.clip(c.k_fb * (rng - c.approach_target_m), 0.0, c.fb_max))
            # Ramp up to the commit throttle over the last approach_blend_m, so entering
            # COMMIT is continuous rather than a one-tick jump (see approach_blend_m).
            if c.allow_commit and c.approach_blend_m > 0:
                blend = np.clip((c.commit_range_m + c.approach_blend_m - rng)
                                / c.approach_blend_m, 0.0, 1.0)
                fb = max(fb, c.commit_fb * float(blend))
            # Never drive forward while badly misaligned or far off the ring's axis:
            # arriving fast and off-centre is the one state COMMIT cannot recover from.
            lat, _vert = self._errors_m(ring)
            if lat > c.forward_misalign_m or ring.axis_ratio < c.forward_axis_ratio_min:
                fb = 0.0
            else:
                fb *= float(np.clip(1.0 - lat / c.forward_misalign_m, 0.0, 1.0))
            a[spec.ACT_FB] = fb
        return a

    # -- the controller ---------------------------------------------------------------
    def __call__(self, ring: RingObs, state: Optional[dict] = None):
        """-> (action (4,), ExpertInfo). `state` is accepted for signature parity with a
        policy and is not currently used."""
        c = self.cfg
        a = np.zeros(spec.ACT_DIM)

        if self.phase in (DONE, ABORT):
            return a, self._info()

        # COMMIT is deliberately blind: no detection input at all. Re-entering the servo
        # on a spurious late detection would steer mid-push.
        if self.phase == COMMIT:
            self._commit_tick += 1
            if self._commit_tick <= self._commit_target:
                a[spec.ACT_FB] = c.commit_fb
                return a, self._info()
            self._settle_tick += 1
            if self._settle_tick >= c.settle_ticks:
                self.phase = DONE
                self.reason = "commit complete"
            return a, self._info()

        if not ring.found:
            self._lost += 1
            if self.phase in (ALIGN, APPROACH) and self._lost > spec.LOCK_MAX_AGE:
                self._aborts += 1
                self.phase = SEARCH
                self._search_ticks = 0
                self._search_dir *= -1.0           # scan back the other way
                self.reason = f"lost lock ({self._aborts})"
                if self._aborts > c.max_aborts:
                    self.phase = ABORT
                    self.reason = "lost lock too many times"
            if self.phase == SEARCH:
                self._search_ticks += 1
                if self._search_ticks > c.search_max_ticks:
                    self.phase = ABORT
                    self.reason = "ring not found while searching"
                else:
                    # Sweep yaw to BOTH sides of the start heading: one leg out, then
                    # double-length legs, so it swings -A..+A rather than 0..+A (equal
                    # legs from the start heading would never look to the other side).
                    # Once the ring appears _servo() takes over height.
                    self._search_sweep_tick += 1
                    if self._search_sweep_tick >= self._search_leg:
                        self._search_dir *= -1.0
                        self._search_sweep_tick = 0
                        self._search_leg = 2 * c.search_sweep_ticks
                    a[spec.ACT_YAW] = c.search_yaw * self._search_dir
                    a[spec.ACT_UD] = c.search_ud
            return a, self._info(ring)

        # We have a lock.
        self._lost = 0
        if np.isfinite(ring.range_m):
            self._last_range = ring.range_m
        if self.phase == SEARCH:
            self.phase = ALIGN
            self.reason = "acquired"

        if self._can_commit(ring):
            self.phase = COMMIT
            self._commit_tick = 0
            self._settle_tick = 0
            self._commit_target = self._commit_ticks_for(self._last_range)
            self.reason = (f"commit at {self._last_range:.2f} m "
                           f"({self._commit_target} ticks)")
            a[spec.ACT_FB] = c.commit_fb
            return a, self._info(ring)

        self.phase = APPROACH if self._aligned(ring) else ALIGN
        return self._servo(ring), self._info(ring)

    def _info(self, ring: Optional[RingObs] = None) -> ExpertInfo:
        return ExpertInfo(phase=self.phase, reason=self.reason,
                          commit_tick=self._commit_tick,
                          commit_target=self._commit_target, aborts=self._aborts,
                          range_m=(ring.range_m if ring is not None else self._last_range),
                          seek_dir=self._seek_dir)

    @property
    def finished(self) -> bool:
        return self.phase in (DONE, ABORT)


class ExpertPolicy:
    """The scripted Expert behind a learned policy's interface: obs -> action.

    Makes the expert a drop-in wherever a policy is expected, which is what allows the
    two to be scored on identical episodes (gym_env.rollout) and the expert to generate
    demonstrations through the same path a policy would fly. `reset()` is required
    because the Expert runs a phase machine that ends an attempt in DONE or ABORT.
    """

    def __init__(self, cfg: Optional[ExpertConfig] = None):
        self.expert = Expert(cfg)

    def reset(self):
        self.expert.reset()

    def __call__(self, obs) -> np.ndarray:
        from .obs import ring_from_obs
        action, self.info = self.expert(ring_from_obs(obs), None)
        return np.asarray(action, dtype=float)

    @property
    def phase(self) -> str:
        return self.expert.phase

    @property
    def finished(self) -> bool:
        return self.expert.finished
