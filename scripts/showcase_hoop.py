"""
Play ONE episode of a run's current policy, for the dashboard's "Watch current
policy" button. Unlike the training env the dashboard normally shows (mostly exploration), this is
the policy acting deterministically, so you can see what it has actually learned so far.

By default it runs as fast as the simulator goes (a 20 s episode plays in a few seconds);
--realtime paces it to real speed. Telemetry and camera frames are written per *simulated* time
(15 Hz and 6 Hz), so the dashboard gets the same trail and POV either way.

Runs headless in its own simulator (it can run while training continues) and writes, under the
run directory:
    showcase.jsonl     telemetry records (same format as TelemetryCallback, plus
                       "source": "showcase"), paced to real time, ending with one
                       {"summary": true, ...} record
    showcase_cam.jpg   the drone's onboard camera, ~8 Hz

Usage (the backend launches this; see webapp/backend/run_manager.start_showcase):
    python showcase_hoop.py --run-dir ../results/hoop/runs/<run_id> [--variant latest|best] [--seed N] [--realtime]
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hoop_env import HoopFindAviary  # noqa: E402
from hoop_net_env import HoopNetAviary  # noqa: E402
from discrete_action_wrapper import HoopFindDiscreteVel  # noqa: E402
from camera_capture import get_onboard_frame  # noqa: E402

ALGO_CLASSES = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN}
TELEMETRY_HZ = 15.0
CAMERA_HZ = 6.0


def pick_model(run_dir: Path, variant: str) -> Path:
    order = ["latest_model.zip", "best_model.zip", "final_model.zip"]
    if variant == "best":
        order = ["best_model.zip", "latest_model.zip", "final_model.zip"]
    for name in order:
        if (run_dir / name).is_file():
            return run_dir / name
    raise FileNotFoundError(f"no model in {run_dir}")


def write_jpg(path: Path, rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    try:
        os.replace(tmp, path)
    except PermissionError:          # OneDrive / the backend reading it: skip this frame
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=["latest", "best"], default="latest")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sway-deg", type=float, default=None,
                    help="net env: largest ring swing about its rope knot (deg; default: the run's own)")
    ap.add_argument("--realtime", action="store_true", help="pace to real speed (default: full speed)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    model_path = pick_model(run_dir, args.variant)
    model = ALGO_CLASSES.get(meta.get("algo", "ppo"), PPO).load(str(model_path))
    use_net = meta.get("env") == "net"
    seed = args.seed if args.seed is not None else int(time.time()) % 100000

    sway = args.sway_deg if args.sway_deg is not None else meta.get("sway_deg")
    # same episode mix as the run trained with (older runs: always front -> back, usual lean)
    mix = {k: meta[k] for k in ("reverse_prob", "lean_flip_prob") if meta.get(k) is not None}
    if not mix:
        mix = {"reverse_prob": 0.0, "lean_flip_prob": 0.0}
    env = HoopNetAviary(seed=seed, sway_deg=sway, **mix) if use_net else HoopFindAviary(seed=seed)
    raw = env
    if meta.get("env_variant") == "discrete_vel":
        env = HoopFindDiscreteVel(env)
    out = run_dir / "showcase.jsonl"
    cam = run_dir / "showcase_cam.jpg"
    print(f"[showcase] {model_path.name} ({meta.get('algo')}, env={'net' if use_net else 'legacy'}), seed {seed}")

    obs, _ = env.reset(seed=seed)
    dt = raw.CTRL_TIMESTEP
    t0 = time.time()
    tel_every = max(1, round(1 / (TELEMETRY_HZ * dt)))      # control steps between records
    # camera renders dominate playback time; at full speed take a few fewer (4 Hz of sim time)
    cam_every = max(1, round(1 / ((CAMERA_HZ if args.realtime else 4.0) * dt)))
    net_hits = rim_hits = 0
    hit_since = {"net": False, "rim": False}
    info, k, term, trunc = {}, 0, False, False
    with open(out, "a", newline="\n") as f:
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(act)
            k += 1
            net_hits += bool(info.get("net_contact"))
            rim_hits += bool(info.get("rim_contact"))
            hit_since["net"] |= bool(info.get("net_contact"))
            hit_since["rim"] |= bool(info.get("rim_contact"))
            now = time.time()
            end = term or trunc
            if end or k % tel_every == 0:
                o = np.asarray(obs).reshape(-1)
                pos, rel = o[0:3], o[-3:]
                rec = {"source": "showcase", "ts": now, "step": k, "pos": pos.tolist(),
                       "rpy": o[3:6].tolist(), "target": (pos + rel).tolist(),
                       "net_contact": hit_since["net"], "rim_contact": hit_since["rim"],
                       "passes": int(info.get("passes", 0))}
                if use_net:
                    rec["hoop_normal"] = info.get("hoop_normal")
                    rec["hoop_radius"] = info.get("hoop_radius")
                hit_since = {"net": False, "rim": False}
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if (k - 1) % cam_every == 0 or end:
                try:
                    write_jpg(cam, get_onboard_frame(raw, nth_drone=0, res=(200, 150)))
                except Exception as e:      # a dropped frame must not end the showcase
                    print(f"[showcase] camera frame failed: {e}")
            if end:
                break
            if args.realtime:                 # pace to real time
                lag = t0 + k * dt - time.time()
                if lag > 0:
                    time.sleep(lag)

        success = bool(info.get("is_success", info.get("success", False))) if use_net else bool(term)
        summary = {"source": "showcase", "summary": True, "success": success,
                   "passes": int(info.get("passes", 0)), "net_hits": net_hits, "rim_hits": rim_hits,
                   "duration_s": round(k * dt, 2), "ended": "passed" if success else
                   ("timeout" if trunc and k * dt >= getattr(raw, "EPISODE_LEN_SEC", 1e9) else
                    "tipped over" if trunc else "ended"),
                   "model": model_path.name, "realtime": bool(args.realtime),
                   "reversed": bool(info.get("reversed", False)),
                   "lean_flipped": bool(info.get("lean_flipped", False))}
        f.write(json.dumps(summary) + "\n")
    print(f"[showcase] done: {summary}")
    env.close()


if __name__ == "__main__":
    main()
