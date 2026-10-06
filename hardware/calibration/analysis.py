"""
Pure analysis for the calibration scripts (no drone, no camera): turns recorded trials into config values.

Frames: cage frame x/y (z up), body frame FLU (x forward, y left, z up). Camera positions are cage-frame
metres from the overhead camera ONLY (never the fused estimate, whose velocity input depends on the very
conventions being tested).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

UNITS = (0.01, 0.1, 1.0)           # plausible Tello velocity units -> m/s


@dataclass
class Trial:
    name: str
    rc: tuple                                   # (lr, fb, ud, yaw) sent
    t: np.ndarray                               # tick times
    cam: np.ndarray                             # (N, 4): t, x, y, yaw_cam   (camera detections during the trial)
    fields: List[dict]                          # raw Tello fields per tick (vgx, vgy, vgz, yaw, h, ...)
    t_cmd: float = 0.0                          # when the rc command started
    t_end: float = 0.0                          # when it stopped
    heading_cage: float = 0.0                   # cage-frame heading at the start (rad, from camera)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _fit_line(t, p):
    """Least-squares position slope (velocity) and the fit's displacement over the span."""
    t = np.asarray(t) - np.mean(t)
    A = np.stack([t, np.ones_like(t)], axis=1)
    sol, *_ = np.linalg.lstsq(A, np.asarray(p), rcond=None)
    return sol[0]


def cam_velocity_body(trial: Trial, t_lo, t_hi) -> Optional[np.ndarray]:
    """Mean camera velocity in the body frame (forward, left) over [t_lo, t_hi], using the trial's heading."""
    c = trial.cam
    if len(c) < 5:
        return None
    m = (c[:, 0] >= t_lo) & (c[:, 0] <= t_hi)
    if m.sum() < 5:
        return None
    vx = _fit_line(c[m, 0], c[m, 1])
    vy = _fit_line(c[m, 0], c[m, 2])
    h = trial.heading_cage
    return np.array([math.cos(h) * vx + math.sin(h) * vy, -math.sin(h) * vx + math.cos(h) * vy])


def snap_unit(ratio: float) -> Optional[float]:
    for u in UNITS:
        if abs(ratio - u) / u < 0.4:
            return u
    return None


def mean_field(trial: Trial, key, t_lo, t_hi):
    vals = [f[key] for tt, f in zip(trial.t, trial.fields) if t_lo <= tt <= t_hi and key in f]
    return float(np.mean(vals)) if vals else None


@dataclass
class AxesResult:
    conventions: dict = field(default_factory=dict)      # updates for config conventions
    problems: List[str] = field(default_factory=list)    # unresolved issues: do NOT mark calibrated
    notes: List[str] = field(default_factory=list)
    ok: bool = False


def analyze_axes(trials: Dict[str, Trial], current: dict) -> AxesResult:
    """trials: 'fb', 'lr', 'ud', 'yaw'. current: the config 'conventions' dict."""
    res = AxesResult()
    rc_signs = dict(current["rc_signs"])
    vel_signs = list(current["vel_signs"])
    att_signs = dict(current["att_signs"])
    unit_estimates = []

    # forward / backward
    tr = trials.get("fb")
    if tr is not None:
        win = (tr.t_cmd + 0.6, tr.t_end)
        v = cam_velocity_body(tr, *win)
        if v is None:
            res.problems.append("fb: no camera data during the forward hop")
        else:
            fwd, left = v
            sign_cmd = math.copysign(1, tr.rc[1])
            if abs(left) > abs(fwd):
                res.problems.append(f"fb: drone moved sideways (forward {fwd:+.2f}, left {left:+.2f} m/s): rc channels look swapped")
            else:
                if fwd * sign_cmd < 0:
                    rc_signs["fb"] = -rc_signs["fb"]
                    res.notes.append("fb: positive rc moved the drone BACKWARD -> flipping rc_signs.fb")
                vgx = mean_field(tr, "vgx", *win)
                if vgx is not None and abs(vgx) > 1e-6:
                    sgn = math.copysign(1, fwd / vgx)
                    vel_signs[0] = int(sgn)
                    u = snap_unit(abs(fwd / vgx))
                    if u:
                        unit_estimates.append(u)
                    else:
                        res.problems.append(f"fb: telemetry vgx unit unclear (camera {abs(fwd):.2f} m/s vs raw {abs(vgx):.1f})")
                else:
                    res.problems.append("fb: vgx stayed ~0 during motion: cannot determine its sign")

    # left / right
    tr = trials.get("lr")
    if tr is not None:
        win = (tr.t_cmd + 0.6, tr.t_end)
        v = cam_velocity_body(tr, *win)
        if v is None:
            res.problems.append("lr: no camera data during the sideways hop")
        else:
            fwd, left = v
            sign_cmd = math.copysign(1, tr.rc[0])            # positive rc lr must mean RIGHT = negative left
            if abs(fwd) > abs(left):
                res.problems.append(f"lr: drone moved forward/back (forward {fwd:+.2f}, left {left:+.2f} m/s): rc channels look swapped")
            else:
                if left * sign_cmd > 0:
                    rc_signs["lr"] = -rc_signs["lr"]
                    res.notes.append("lr: positive rc moved the drone LEFT -> flipping rc_signs.lr")
                vgy = mean_field(tr, "vgy", *win)
                if vgy is not None and abs(vgy) > 1e-6:
                    vel_signs[1] = int(math.copysign(1, left / vgy))
                    u = snap_unit(abs(left / vgy))
                    if u:
                        unit_estimates.append(u)
                else:
                    res.problems.append("lr: vgy stayed ~0 during motion: cannot determine its sign")

    # up / down (the overhead camera cannot see height: use Tello's own height reading)
    tr = trials.get("ud")
    if tr is not None:
        h0 = mean_field(tr, "h", tr.t_cmd - 0.5, tr.t_cmd)
        h1 = mean_field(tr, "h", tr.t_end - 0.1, tr.t_end + 0.6)
        if h0 is None or h1 is None:
            res.problems.append("ud: no height telemetry")
        else:
            dz = (h1 - h0) / 100.0
            if abs(dz) < 0.03:
                res.problems.append(f"ud: height barely changed ({dz:+.3f} m); cannot verify")
            else:
                if dz * math.copysign(1, tr.rc[2]) < 0:
                    rc_signs["ud"] = -rc_signs["ud"]
                    res.notes.append("ud: positive rc moved the drone DOWN -> flipping rc_signs.ud")
                vgz = mean_field(tr, "vgz", tr.t_cmd + 0.4, tr.t_end)
                dur = max(tr.t_end - tr.t_cmd, 1e-3)
                if vgz is not None and abs(vgz) > 1e-6:
                    vel_signs[2] = int(math.copysign(1, (dz / dur) / vgz))
                else:
                    res.problems.append("ud: vgz stayed ~0 during climb: cannot determine its sign")

    # yaw
    tr = trials.get("yaw")
    if tr is not None:
        c = tr.cam
        m0 = (c[:, 0] >= tr.t_cmd - 0.4) & (c[:, 0] <= tr.t_cmd)
        m1 = (c[:, 0] >= tr.t_end - 0.1) & (c[:, 0] <= tr.t_end + 0.5)
        y0, y1 = mean_field(tr, "yaw", tr.t_cmd - 0.4, tr.t_cmd), mean_field(tr, "yaw", tr.t_end - 0.1, tr.t_end + 0.5)
        if m0.sum() < 3 or m1.sum() < 3 or y0 is None or y1 is None:
            res.problems.append("yaw: not enough camera/telemetry data")
        else:
            d_cam = math.degrees(_wrap(np.mean(c[m1, 3]) - np.mean(c[m0, 3])))      # CCW positive, cage frame
            d_tel = (y1 - y0 + 180) % 360 - 180
            if abs(d_cam) < 8:
                res.problems.append(f"yaw: heading barely changed ({d_cam:+.1f} deg); cannot verify")
            else:
                if d_cam * math.copysign(1, tr.rc[3]) > 0:         # positive rc yaw must be clockwise = negative CCW
                    rc_signs["yaw"] = -rc_signs["yaw"]
                    res.notes.append("yaw: positive rc turned the drone counter-clockwise -> flipping rc_signs.yaw")
                if abs(d_tel) > 3:
                    att_signs["yaw"] = int(math.copysign(1, d_cam / d_tel))
                else:
                    res.problems.append("yaw: Tello yaw reading did not change")

    if unit_estimates:
        res.conventions["vel_to_mps"] = float(np.median(unit_estimates))
        if len(set(unit_estimates)) > 1:
            res.problems.append(f"fb/lr disagree on the velocity unit: {unit_estimates}")
    res.conventions.update(rc_signs=rc_signs, vel_signs=vel_signs, att_signs=att_signs)
    missing = [k for k in ("fb", "lr", "ud", "yaw") if k not in trials]
    if missing:
        res.problems.append("missing trials: " + ", ".join(missing))
    res.ok = not res.problems
    return res


# -- speed ----------------------------------------------------------------------------------------
def fit_rc_per_mps(levels, speeds) -> float:
    """Least squares through the origin: rc = k * speed."""
    r, v = np.asarray(levels, dtype=float), np.asarray(speeds, dtype=float)
    return float(np.sum(r * v) / np.sum(v * v))


def steady_speed_and_tau(trial: Trial, settle=1.0):
    """Steady forward speed (m/s, camera) and time to reach 63 % of it after the command."""
    c = trial.cam
    m = (c[:, 0] >= trial.t_cmd) & (c[:, 0] <= trial.t_end + 0.8)
    if m.sum() < 8:
        return None, None
    t = c[m, 0]
    h = trial.heading_cage
    s = np.cos(h) * c[m, 1] + np.sin(h) * c[m, 2]            # position along the initial heading
    steady = cam_velocity_body(trial, trial.t_cmd + settle, trial.t_end)
    if steady is None:
        return None, None
    v_ss = abs(steady[0])
    # speed history by finite differences over 0.3 s, then first time it crosses 63 %
    tau = None
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] - 0.3)
        if t[i] - t[j] >= 0.2:
            if abs((s[i] - s[j]) / (t[i] - t[j])) >= 0.63 * v_ss:
                tau = float(t[i] - trial.t_cmd)
                break
    return float(v_ss), tau
