"""
Fly a Crazyflie inside a virtual basketball court.

Two modes:

  (default) PID waypoint tour -- a classic position controller (DSLPIDControl)
            flies the drone on a lap of the court: centre -> wings -> up to each
            hoop -> back. This actually moves around, so it's the good demo.

  --rl      drop the TRAINED RL hover policy (results/best_model.zip) into the
            court. Note: it was trained with the 1-D-thrust action space, so it
            can only hold altitude -- it hovers at centre court, it doesn't fly
            around. Flying the RL agent around needs retraining with a position
            or velocity action space (see PROGRESS.md).

Usage:
    conda activate drones
    python fly_in_court.py                 # PID tour, GUI
    python fly_in_court.py --record        # + save results/video-*.mp4
    python fly_in_court.py --scale 0.4     # smaller court
    python fly_in_court.py --rl            # RL hover policy in the court
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.utils.utils import sync

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE))
import court as court_mod  # noqa: E402  (local module, next to this file)


def set_camera(client, info):
    p.resetDebugVisualizerCamera(
        cameraDistance=info["length"] * 0.85,
        cameraYaw=50, cameraPitch=-35,
        cameraTargetPosition=[0, 0, info["rim_height"] * 0.4],
        physicsClientId=client,
    )


def court_waypoints(info):
    """A lap of the court, in metres. Columns: x, y, z."""
    rx, rh, Wd = info["rim_x"], info["rim_height"], info["width"]
    z = 1.0
    return np.array([
        [0, 0, 0.15], [0, 0, z],                    # take off, hold centre
        [rx * 0.35,  Wd * 0.28, z],                 # right wing
        [rx * 0.80,  Wd * 0.40, z],                 # right corner three
        [rx - 0.7,   0.0, rh],                      # up to the right hoop
        [rx * 0.30,  0.0, z + 0.3],                 # pull back out
        [0, 0, z + 0.4],                            # centre, high
        [-rx * 0.35, -Wd * 0.28, z],                # left wing
        [-rx * 0.80, -Wd * 0.40, z],               # left corner three
        [-rx + 0.7,  0.0, rh],                      # up to the left hoop
        [-rx * 0.30, 0.0, z + 0.3],
        [0, 0, z],                                  # back to centre
        [0, 0, 0.12],                               # land
    ])


def run_pid(args):
    start = np.array([[0.0, 0.0, 0.10]])
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=start, pyb_freq=240, ctrl_freq=48,
                     gui=True, record=args.record, obstacles=False,
                     user_debug_gui=False, output_folder=str(RESULTS))
    client = env.getPyBulletClient()
    info = court_mod.build_court(client, scale=args.scale)
    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=client)  # hide default ground
    set_camera(client, info)

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    wps = court_waypoints(info)
    speed = 1.1                                   # m/s target-cursor speed
    tol = 0.16
    cursor = start[0].copy()
    wp_i = 0
    dt = env.CTRL_TIMESTEP

    action = np.zeros((1, 4))
    t0 = time.time()
    max_steps = int(args.max_seconds * env.CTRL_FREQ)
    for i in range(max_steps):
        obs, *_ = env.step(action)
        pos = obs[0][0:3]

        goal = wps[wp_i]
        to_goal = goal - cursor
        d = np.linalg.norm(to_goal)
        cursor += to_goal / d * min(speed * dt, d) if d > 1e-6 else 0.0

        if np.linalg.norm(pos - goal) < tol and d < 0.05:
            wp_i = min(wp_i + 1, len(wps) - 1)

        yaw = np.arctan2(goal[1] - pos[1], goal[0] - pos[0]) if wp_i > 1 else 0.0
        action[0], _, _ = ctrl.computeControlFromState(
            control_timestep=dt, state=obs[0],
            target_pos=cursor, target_rpy=[0, 0, yaw])

        env.render()
        sync(i, t0, dt)
        if wp_i == len(wps) - 1 and np.linalg.norm(pos - wps[-1]) < 0.15:
            break
    env.close()
    if args.record:
        print(f"[fly] video saved under {RESULTS}/video-*.mp4")


def run_rl(args):
    from stable_baselines3 import PPO
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.utils.enums import ObservationType, ActionType

    model = PPO.load(str(RESULTS / "best_model.zip"))
    env = HoverAviary(gui=True, obs=ObservationType.KIN, act=ActionType.ONE_D_RPM,
                      record=args.record)
    client = env.getPyBulletClient()
    info = court_mod.build_court(client, scale=args.scale)
    p.changeVisualShape(env.PLANE_ID, -1, rgbaColor=[0, 0, 0, 0], physicsClientId=client)
    set_camera(client, info)

    obs, _ = env.reset(seed=0)
    t0 = time.time()
    for i in range((env.EPISODE_LEN_SEC + 6) * env.CTRL_FREQ):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(act)
        env.render()
        sync(i, t0, env.CTRL_TIMESTEP)
        if term or trunc:
            obs, _ = env.reset(seed=0)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", action="store_true", help="fly the trained RL hover policy instead of the PID tour")
    ap.add_argument("--scale", type=float, default=0.5, help="court size (0.5 = 14 x 7.5 m)")
    ap.add_argument("--record", action="store_true", help="save an mp4 into results/")
    ap.add_argument("--max_seconds", type=float, default=70, help="PID tour time budget")
    args = ap.parse_args()
    (run_rl if args.rl else run_pid)(args)


if __name__ == "__main__":
    main()
