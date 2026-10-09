"""
Generate a replay video for a run the dashboard already trained: a
third-person (chase-cam) mp4 always, and optionally a second onboard "drone
POV" mp4 of the same flight.

Reads meta.json from the run directory to know which algorithm (and so which
action-space variant, continuous vs the DQN discrete wrapper) the saved model
expects, so the replay env matches what it was trained with.

MP4 recording needs a visible PyBullet window (gui=True); record=True with
gui=False only writes numbered PNG frames, no automatic video, confirmed by
reading gym_pybullet_drones/envs/BaseAviary.py. So this always opens a
(brief) window on the desktop. ffmpeg must be on PATH, both for PyBullet's
own third-person recorder and, when --pov is given, for this script's own
PNG-to-mp4 stitching of the onboard frames.

Two more things confirmed by reading BaseAviary, not assumed:
- It writes to a relative "results/video-<timestamp>.mp4" (BaseRLAviary,
  which HoopFindAviary is built on, does not expose an output_folder
  parameter to override this), so this script temporarily chdirs into the
  run's replay/ directory before creating the env, then moves the file out
  and restores the original working directory, rather than widening the
  dependency's own API.
- _startVideoRecording() runs again on every reset(), starting a NEW
  p.startStateLogging() without stopping the previous one, so a multi-episode
  replay can leave more than one logger writing to the same PyBullet client
  at once (an upstream quirk, not something this script works around).
  Episodes therefore default to 1: a single clean clip. Passing more than 1
  is supported but not recommended for now.

Usage:
    conda activate drones
    python replay_hoop.py --run-dir ../results/hoop/runs/ppo_20260930-143210 \
        --variant best --replay-id best_20260930-160000
    python replay_hoop.py --run-dir ../results/hoop/runs/ppo_20260930-143210 \
        --variant current --replay-id current_20260930-160500 --pov --episodes 2
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pybullet as p
from stable_baselines3 import PPO, A2C, SAC, TD3, DQN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import court as court_mod  # noqa: E402  (local module, next to this file)
from hoop_env import HoopFindAviary  # noqa: E402
from hoop_net_env import HoopNetAviary  # noqa: E402
from discrete_action_wrapper import HoopFindDiscreteVel  # noqa: E402
from camera_capture import get_onboard_frame  # noqa: E402

ALGO_CLASSES = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN}


def set_camera(client, room):
    # BaseAviary only disables the RGB/depth/segmentation preview panels, not
    # the Explorer/Params side panels or top menu, which otherwise show up in
    # the recorded video.
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.resetDebugVisualizerCamera(
        cameraDistance=max(room) * 1.6, cameraYaw=40, cameraPitch=-30,
        cameraTargetPosition=[0, 0, room[2] / 2], physicsClientId=client,
    )


def resolve_model_path(run_dir: Path, variant: str) -> Path:
    if variant == "best":
        return run_dir / "best_model.zip"
    final = run_dir / "final_model.zip"
    return final if final.is_file() else run_dir / "best_model.zip"


def stitch_pov_video(frame_dir: Path, out_path: Path, fps: int):
    if shutil.which("ffmpeg") is None:
        print("[replay_hoop] WARNING: ffmpeg not found on PATH, skipping POV video stitch")
        return False
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-pix_fmt", "yuv420p", str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[replay_hoop] ffmpeg failed (code {result.returncode}):\n{result.stderr[-2000:]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=["current", "best"], default="current")
    ap.add_argument("--replay-id", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=1,
                     help="1 recommended (see module docstring: BaseAviary restarts its "
                          "video logger on every reset without stopping the previous one)")
    ap.add_argument("--pov", action="store_true")
    ap.add_argument("--sway-deg", type=float, default=None,
                    help="net env: largest ring swing about its rope knot (deg; default: the run's own)")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    meta = json.loads((run_dir / "meta.json").read_text())
    algo = meta.get("algo", "ppo")
    needs_discrete = meta.get("env_variant") == "discrete_vel"
    use_net = meta.get("env") == "net"          # runs from before the net env have no "env" key

    model_path = resolve_model_path(run_dir, args.variant)
    if not model_path.is_file():
        print(f"[replay_hoop] ERROR: no model at {model_path}")
        sys.exit(1)
    model = ALGO_CLASSES.get(algo, PPO).load(str(model_path))

    replay_dir = run_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    out_video = replay_dir / f"{args.replay_id}.mp4"
    out_pov = replay_dir / f"{args.replay_id}_pov.mp4"

    # HoopFindAviary (via BaseRLAviary) does not expose output_folder, so its
    # recording always lands in a relative "results/video-*.mp4". Rather than
    # widening the dependency's API, chdir into replay_dir for the duration
    # of the recording and move the file out afterward.
    original_cwd = Path.cwd()
    os.chdir(replay_dir)
    pov_frame_dir = None
    frame_idx = 0
    env = None
    try:
        sway = args.sway_deg if args.sway_deg is not None else meta.get("sway_deg")
        env = (HoopNetAviary(gui=True, record=True, sway_deg=sway) if use_net
               else HoopFindAviary(gui=True, record=True))
        if needs_discrete:
            env = HoopFindDiscreteVel(env)
        raw_env = env.unwrapped
        client = raw_env.getPyBulletClient()
        set_camera(client, raw_env.ROOM)

        if args.pov:
            pov_frame_dir = Path(tempfile.mkdtemp(prefix=f"pov_{args.replay_id}_"))

        def rebuild_scene():
            # env.reset() calls p.resetSimulation() internally, wiping every
            # body (drone, ground plane, our floor/hoop), so the whole scene
            # has to be rebuilt after every reset, not just the first one,
            # mirroring fly_hoop_demo.py's own rebuild_scene().
            if use_net:      # HoopNetAviary builds its own net + ring on every reset
                return
            p.changeVisualShape(raw_env.PLANE_ID, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=client)
            court_mod.build_room(client, room=tuple(raw_env.ROOM))
            court_mod.build_hoop(client, hoop_pos=tuple(raw_env.TARGET_POS))

        from gym_pybullet_drones.utils.utils import sync

        ep = 0
        obs, _ = env.reset(seed=1000 + ep)
        rebuild_scene()
        t0 = time.time()
        step = 0
        while ep < args.episodes:
            act, _ = model.predict(obs, deterministic=True)
            obs, _reward, term, trunc, _info = env.step(act)
            env.render()
            if pov_frame_dir is not None:
                frame = get_onboard_frame(raw_env, nth_drone=0)
                from PIL import Image
                Image.fromarray(frame).save(pov_frame_dir / f"frame_{frame_idx:06d}.png")
                frame_idx += 1
            sync(step, t0, raw_env.CTRL_TIMESTEP)
            step += 1
            if term or trunc:
                ep += 1
                if ep >= args.episodes:
                    break
                obs, _ = env.reset(seed=1000 + ep)
                rebuild_scene()
                t0 = time.time()
                step = 0
        env.close()  # stops PyBullet's own STATE_LOGGING_VIDEO_MP4, writing the video's content
    finally:
        os.chdir(original_cwd)

    # BaseAviary wrote to <replay_dir>/results/video-<its own timestamp>.mp4
    # (see module docstring); find that file and move it to the path the
    # dashboard was told to expect.
    candidates = sorted((replay_dir / "results").glob("video-*.mp4"),
                         key=lambda f: f.stat().st_mtime) if (replay_dir / "results").is_dir() else []
    if candidates:
        candidates[-1].replace(out_video)
        shutil.rmtree(replay_dir / "results", ignore_errors=True)
    else:
        print("[replay_hoop] WARNING: no third-person video was produced "
              "(check that ffmpeg is on PATH)")

    if pov_frame_dir is not None and frame_idx > 0:
        fps = int(round(raw_env.CTRL_FREQ))
        stitch_pov_video(pov_frame_dir, out_pov, fps)
        shutil.rmtree(pov_frame_dir, ignore_errors=True)

    print(f"[replay_hoop] done: {out_video}" + (f", {out_pov}" if args.pov else ""))


if __name__ == "__main__":
    main()
