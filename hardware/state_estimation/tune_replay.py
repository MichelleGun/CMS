"""
Tune the fusion filter's noise settings offline, from a logged flight.

    python hardware/state_estimation/tune_replay.py --log hardware/flights/teleop_<ts>.jsonl
    python hardware/state_estimation/tune_replay.py --log ... --grid wide

Every flight log stores the raw telemetry packets and camera detections, so the estimator can be re-run
with different Q/R as often as you like. Scoring: every 4th camera detection is WITHHELD from the filter;
we measure how far the filter's prediction at that instant is from it. Lower is better. The floor is
about the camera's own noise (it is also a noisy measurement), so compare candidates against each other
and against the current config, not against zero.

How to read the result (see TUNING.md): if the best sigma_cam_xy is much larger than the config value,
your camera is noisier than assumed (light, motion blur, small marker); if the best q_acc is large, the
constant-velocity model lags your real manoeuvres; if the best sigma_vel is large, Tello's velocity
readings are biased/noisy and the filter should lean on the camera more.
"""
import argparse
import copy
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW))
from common.config import load_effective_config      # noqa: E402
from state_estimation.est_types import CameraDetection   # noqa: E402
from state_estimation.estimator import StateEstimator   # noqa: E402

GRIDS = {
    "small": dict(q_acc=[0.4, 0.8, 1.6], sigma_cam_xy=[0.02, 0.04], sigma_vel=[0.04, 0.08]),
    "wide": dict(q_acc=[0.2, 0.4, 0.8, 1.6, 3.2], sigma_cam_xy=[0.01, 0.02, 0.04, 0.08],
                 sigma_vel=[0.02, 0.04, 0.08, 0.16]),
}


def load_ticks(path):
    ticks = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("type") == "tick":
            ticks.append(r)
    if not ticks:
        raise ValueError("no tick records in log")
    return ticks


def replay_score(ticks, cfg, holdout_every=4):
    """RMS (m) between filter prediction and the withheld camera detections."""
    est = StateEstimator(cfg, t0=ticks[0]["t"])
    errs, n_cam = [], 0
    for r in ticks:
        now = r["t"]
        est.on_raw_telemetry(r.get("tele") or [], now)
        for (t, x, y, yaw, q) in (r.get("cam") or []):
            n_cam += 1
            det = CameraDetection(t=t, x=x, y=y, yaw=yaw, quality=q)
            if n_cam % holdout_every == 0:
                if est.kf.xy_initialized:
                    xp, _ = est.kf.peek(t - est.latency.latency)
                    errs.append(math.hypot(xp[0] - x, xp[1] - y))
            else:
                est.on_camera(det, now)
    if len(errs) < 10:
        return float("nan"), len(errs)
    return float(np.sqrt(np.mean(np.square(errs)))), len(errs)


def tune(ticks, base_cfg, grid):
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = copy.deepcopy(base_cfg)
        for k, v in zip(keys, combo):
            cfg["estimator"][k] = v
        rms, n = replay_score(ticks, cfg)
        rows.append((rms, dict(zip(keys, combo)), n))
    rows.sort(key=lambda r: (np.isnan(r[0]), r[0]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--grid", choices=list(GRIDS), default="small")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    cfg = load_effective_config()
    ticks = load_ticks(a.log)
    cur, n = replay_score(ticks, cfg)
    e = cfg["estimator"]
    print(f"[tune] {len(ticks)} ticks, {n} withheld detections")
    print(f"[tune] current config (q_acc={e['q_acc']}, sigma_cam_xy={e['sigma_cam_xy']}, sigma_vel={e['sigma_vel']}): "
          f"RMS {cur * 100:.2f} cm")
    rows = tune(ticks, cfg, GRIDS[a.grid])
    print(f"[tune] best {a.top} of {len(rows)}:")
    for rms, p, _ in rows[:a.top]:
        print(f"   RMS {rms * 100:5.2f} cm   {p}")
    best = rows[0]
    if best[0] < 0.97 * cur:
        print("[tune] a different setting is clearly better: copy these into config.yaml `estimator:`")
    else:
        print("[tune] the current config is already about as good as this grid gets")
    return rows, cur


if __name__ == "__main__":
    main()
