"""
Quick-start RL training for gym-pybullet-drones.

Task:   HoverAviary  -> a single Crazyflie learns to take off and hold z = 1.0 m
Algo:   PPO (Stable-Baselines3), MLP policy
Action: 'one_d_rpm'  -> 1 scalar per step (same RPM on all 4 motors); easiest to learn
Obs:    'kin'        -> position, orientation, linear/angular velocity (12-dim)

Runs headless (no GUI) so it trains as fast as your CPU allows, early-stops once
the policy is good enough, then saves the model. Visualize afterwards with:

    python -m gym_pybullet_drones.examples.play --model_path ../results/best_model.zip

Usage:
    conda activate drones
    python quickstart_train.py
    python quickstart_train.py --timesteps 500000 --out my_run
    python quickstart_train.py --watch          # open the PyBullet window DURING training
                                                # (forces 1 env, ~5-10x slower, but you see
                                                #  the drone go from flailing -> hovering)
"""
import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

OBS = ObservationType.KIN
ACT = ActionType.ONE_D_RPM
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="max training timesteps (early-stops sooner if solved)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="parallel env copies (uses more CPU cores)")
    parser.add_argument("--out", type=Path, default=HERE.parent / "results",
                        help="where to write best_model.zip / evaluations.npz")
    parser.add_argument("--watch", action="store_true",
                        help="show the PyBullet GUI during training (forces --n_envs 1, much slower)")
    args = parser.parse_args()

    OUTPUT_DIR = str(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.watch and args.n_envs != 1:
        print("[INFO] --watch: forcing n_envs=1 (PyBullet allows one GUI window per process)")
        args.n_envs = 1

    # --- environments -------------------------------------------------------
    train_env = make_vec_env(
        HoverAviary,
        env_kwargs=dict(obs=OBS, act=ACT, gui=args.watch),
        n_envs=args.n_envs,
        seed=0,
    )
    eval_env = make_vec_env(HoverAviary, env_kwargs=dict(obs=OBS, act=ACT), n_envs=1, seed=1)

    print("[INFO] action space:     ", train_env.action_space)
    print("[INFO] observation space:", train_env.observation_space)

    # --- agent ------------------------------------------------------------
    # To get TensorBoard curves: `pip install tensorboard`, add
    # tensorboard_log=f"{OUTPUT_DIR}/tb/" below, then `tensorboard --logdir <out>/tb`
    model = PPO("MlpPolicy", train_env, verbose=1)

    # HoverAviary with one_d_rpm: ~474 is "solved" (see examples/learn.py)
    stop_when_good = StopTrainingOnRewardThreshold(reward_threshold=474.0, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_when_good,
        best_model_save_path=OUTPUT_DIR,
        log_path=OUTPUT_DIR,
        eval_freq=2000,          # per-env steps between evaluations
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # --- train ----------------------------------------------------------
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, log_interval=10)
    model.save(f"{OUTPUT_DIR}/final_model.zip")

    print(f"\n[DONE] trained {model.num_timesteps} steps in {time.time() - t0:.0f}s")
    print(f"[DONE] best model -> {OUTPUT_DIR}/best_model.zip")
    print("\nVisualize it with:")
    print(f"    python -m gym_pybullet_drones.examples.play --model_path {OUTPUT_DIR}/best_model.zip")


if __name__ == "__main__":
    main()
