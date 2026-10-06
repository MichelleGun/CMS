"""
How far apart are the simulator and the real flight? Replay the actions a flight actually executed,
open-loop, in HoopFindAviary and compare trajectories.

    python hardware/sim_check/sim_vs_real.py --log hardware/flights/policy_<ts>.jsonl

Uses the logged `executed` actions (the exact sim-format actions that went into the observation buffer) and
the logged fused position as 'real'. The sim starts at the flight's first flying position with zero
velocity. A growing gap means the real drone responds differently from the sim's PID-controlled
Crazyflie to the same command: that gap is the sim-to-real mismatch to close (calibrate rc_per_mps,
add latency / action noise to the sim's training, or fine-tune on hardware data).
Writes a PNG next to the log. Needs pybullet + gym_pybullet_drones (the `drones` env).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
for p in (HW, HW.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_flight(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ticks = [r for r in rows if r.get("type") == "tick" and r.get("phase") == "flying" and r.get("obs")]
    if len(ticks) < 20:
        raise ValueError("not enough flying ticks with observations in this log")
    t = np.array([r["t"] for r in ticks])
    pos = np.array([r["state"][0:3] for r in ticks])
    act = np.array([r["executed"] for r in ticks])
    hoop = pos[0] + np.array(ticks[0]["obs"][72:75])
    return t - t[0], pos, act, hoop


def replay_in_sim(pos0, hoop, actions):
    from hoop_env import HoopFindAviary
    start = np.array([pos0[0], pos0[1], max(float(pos0[2]), 0.1)])
    env = HoopFindAviary(gui=False, start_pos=start, target_pos=np.asarray(hoop, dtype=float))
    out = []
    try:
        env.reset(seed=0)
        raw = env.unwrapped
        for a in actions:
            out.append(raw._getDroneStateVector(0)[0:3].copy())
            _, _, te, tr, _ = env.step(np.asarray(a, dtype=np.float32).reshape(1, 4))
            if te or tr:
                break
        out = np.array(out)
    finally:
        env.close()
    return out


def compare(path, plot=True):
    t, pos, act, hoop = load_flight(path)
    sim = replay_in_sim(pos[0], hoop, act)
    n = len(sim)
    err = np.linalg.norm(sim - pos[:n], axis=1)
    horizons = {h: float(err[min(int(h * 30), n - 1)]) for h in (1, 3, 5, 10) if int(h * 30) < n + 30}
    res = dict(steps=n, rms_m=float(np.sqrt(np.mean(err ** 2))), final_m=float(err[-1]), at_s=horizons,
              hoop=hoop.tolist())
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(pos[:n, 0], pos[:n, 1], label="real (fused)")
        ax[0].plot(sim[:, 0], sim[:, 1], "--", label="sim, same actions")
        ax[0].plot(*hoop[:2], "g*", ms=14, label="hoop")
        ax[0].set_aspect("equal")
        ax[0].set_title("x-y path")
        ax[0].legend()
        ax[1].plot(t[:n], pos[:n, 2], label="real z")
        ax[1].plot(t[:n], sim[:, 2], "--", label="sim z")
        ax[1].set_title("height")
        ax[1].legend()
        ax[2].plot(t[:n], err)
        ax[2].set_title("position gap (m)")
        for a in ax:
            a.set_xlabel("s") if a is not ax[0] else None
        out = Path(path).with_suffix(".sim_vs_real.png")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        res["plot"] = str(out)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    r = compare(a.log, not a.no_plot)
    print(f"[sim_vs_real] {r['steps']} steps replayed; RMS gap {r['rms_m']:.3f} m, final gap {r['final_m']:.3f} m")
    print("              gap at " + ", ".join(f"{h}s: {v:.2f} m" for h, v in r["at_s"].items()))
    if "plot" in r:
        print("              plot:", r["plot"])
    return r


if __name__ == "__main__":
    main()
