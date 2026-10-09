"""
Fly the trained hoop-finding policy inside the 2x2x2 m hardware-demo cage.

The hoop -- and the drone's start position -- jump to a new random spot every
episode. That's the point: proving the policy generalises to "the hoop could
be anywhere in the cage", not just wherever it happened to sit during a fixed
demo run.

Usage:
    conda activate drones
    python fly_hoop_demo.py                    # GUI, loops episodes until Ctrl+C
    python fly_hoop_demo.py --episodes 5
    python fly_hoop_demo.py --record            # save results/video-*.mp4
    python fly_hoop_demo.py --model ../results/hoop/best_model.zip
"""
import argparse
import sys
import time
from pathlib import Path

import pybullet as p
from stable_baselines3 import PPO

from gym_pybullet_drones.utils.utils import sync

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE))
import court as court_mod            # noqa: E402  (local module, next to this file)
from hoop_env import HoopFindAviary  # noqa: E402
from hoop_net_env import HoopNetAviary  # noqa: E402


def set_camera(client, room):
    p.resetDebugVisualizerCamera(
        cameraDistance=max(room) * 1.6, cameraYaw=40, cameraPitch=-30,
        cameraTargetPosition=[0, 0, room[2] / 2], physicsClientId=client,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=RESULTS / "hoop" / "best_model.zip")
    ap.add_argument("--env", choices=["net", "legacy"], default="net",
                    help="net = 2x2x2 net + ring (hoop_net_env.py); legacy = original point-reaching env")
    ap.add_argument("--episodes", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--record", action="store_true", help="save an mp4 into results/")
    ap.add_argument("--sway-deg", type=float, default=None,
                    help="net env: largest ring swing about its rope knot (deg; default: HoopSpec.sway_tilt_deg)")
    args = ap.parse_args()

    model = PPO.load(str(args.model))
    net = args.env == "net"
    env = (HoopNetAviary(gui=True, record=args.record, sway_deg=args.sway_deg) if net
           else HoopFindAviary(gui=True, record=args.record))
    client = env.getPyBulletClient()
    set_camera(client, env.ROOM)

    # env.reset() calls p.resetSimulation() internally, wiping every body
    # (drone, ground plane, our floor/hoop) -- so the whole scene, not just
    # the hoop, has to be rebuilt after every reset, not just the first one.
    def rebuild_scene():
        if net:          # HoopNetAviary builds its own net + ring on every reset
            return
        p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=client)
        court_mod.build_room(client, room=tuple(env.ROOM))
        court_mod.build_hoop(client, hoop_pos=tuple(env.TARGET_POS))

    ep = 0
    obs, _ = env.reset(seed=ep)
    rebuild_scene()
    t0 = time.time()
    step = 0
    while args.episodes == 0 or ep < args.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(act)
        env.render()
        sync(step, t0, env.CTRL_TIMESTEP)
        step += 1
        if term or trunc:
            print(f"[ep {ep}] {'passed through the ring' if (term and net) else 'reached hoop' if term else 'timed out / tipped over'}"
                  f" at target {env.TARGET_POS.round(2)}")
            ep += 1
            obs, _ = env.reset(seed=ep)
            rebuild_scene()
            t0 = time.time()
            step = 0
    env.close()
    if args.record:
        print(f"[fly] video saved under {RESULTS}/video-*.mp4")


if __name__ == "__main__":
    main()
