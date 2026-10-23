"""
Play ONE episode of an onboard-camera run's saved policy, for the dashboard's "Watch policy"
button: the onboard counterpart of scripts/showcase_hoop.py (which is PyBullet-only).

The policy is loaded with policy.load_policy, the loader fly_hoop.py --mode policy uses, and
flown on the laps task export_policy.py scores with (endless same-side laps from a random start
zone, the run's own profile / randomisation), deterministically. So what you watch is what the
exported LOOP_*.zip does, not the exploring training env.

Writes, under the run directory, the same files showcase_hoop.py does:
    showcase.jsonl     telemetry records (TelemetryCallback's format plus "source": "showcase"),
                       ending with one {"summary": true, ...} record
    showcase_cam.jpg   the drone's onboard camera

Usage (the backend launches this; see webapp/backend/run_manager.start_showcase):
    python hardware/onboard/showcase_onboard.py --run-dir results/hoop/runs/<run_id> [--variant best|latest] [--seed N] [--realtime]
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

# flake8: noqa: E402
from onboard import spec
from onboard.export_policy import env_kwargs
from onboard.gym_env import OnboardHoopEnv
from onboard.policy import load_policy

CAMERA_HZ = 6.0


def pick_model(run_dir: Path, variant: str) -> Path:
    order = ["latest_model.zip", "best_model.zip", "final_model.zip"]
    if variant == "best":
        order = ["best_model.zip", "final_model.zip", "latest_model.zip"]
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
    ap.add_argument("--variant", choices=["latest", "best"], default="best")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--realtime", action="store_true", help="pace to real speed (default: full speed)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    model_path = pick_model(run_dir, args.variant)
    fn = load_policy(model_path)
    tracker = fn.obs_width == spec.TRACKER_OBS_DIM
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    env = OnboardHoopEnv(seed=seed, **env_kwargs(meta, tracker))
    if env.observation_space.shape[0] != fn.obs_width:
        sys.exit(f"[showcase] {model_path.name} reads {fn.obs_width} inputs but the env makes "
                 f"{env.observation_space.shape[0]}")
    out = run_dir / "showcase.jsonl"
    cam = run_dir / "showcase_cam.jpg"
    print(f"[showcase] {model_path.name} (onboard, {meta.get('profile')}), seed {seed}")

    obs, _ = env.reset(seed=seed)
    if hasattr(fn, "reset"):            # a memory policy clears its frame stack
        fn.reset()
    dt = spec.CTRL_DT
    cam_every = max(1, round(1 / ((CAMERA_HZ if args.realtime else 4.0) * dt)))
    t0 = time.time()
    rim_hits = 0
    info, k, term, trunc = {}, 0, False, False
    with open(out, "a", newline="\n") as f:
        while True:
            obs, _r, term, trunc, info = env.step(fn(obs))
            k += 1
            contact = bool(info.get("contact"))
            rim_hits += contact
            end = term or trunc
            # one record per control step: the env runs at 15 Hz, the dashboard's telemetry rate
            rec = {"source": "showcase", "ts": time.time(), "step": k, "t": round(k * dt, 2),
                   **env.telemetry(), "net_contact": False, "rim_contact": contact}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k - 1) % cam_every == 0 or end:
                try:
                    write_jpg(cam, env.onboard_frame())
                except Exception as e:      # a dropped frame must not end the showcase
                    print(f"[showcase] camera frame failed: {e}")
            if end:
                break
            if args.realtime:
                lag = t0 + k * dt - time.time()
                if lag > 0:
                    time.sleep(lag)

        passes = int(info.get("passes", 0))
        phase = info.get("phase", "ended")
        ended = ("stalled" if info.get("stalled") else "time up" if trunc else
                 {"struck": "hit the ring", "missed": "missed", "out_of_cage": "left the cage",
                  "wrong_way": "went the wrong way"}.get(phase, phase))
        summary = {"source": "showcase", "summary": True, "success": passes >= 1, "passes": passes,
                   "net_hits": 0, "rim_hits": rim_hits, "duration_s": round(k * dt, 2),
                   "ended": ended, "start": info.get("start_type"),
                   "model": model_path.name, "realtime": bool(args.realtime), "reversed": False}
        f.write(json.dumps(summary) + "\n")
    print(f"[showcase] done: {summary}")
    env.close()


if __name__ == "__main__":
    main()
