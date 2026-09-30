"""
Train a policy to find a hoop that moves to a random spot in the 2x2x2 m
hardware-demo cage every episode -- see hoop_env.py for the environment and
why it needs 'vel' actions and a target-relative observation.

This is a harder task than quickstart_train.py's fixed-target hover (the
policy has to navigate, not just hold altitude), so it needs more timesteps
and there's no known "solved" reward threshold yet -- it just trains for the
given budget and keeps the best eval checkpoint.

Writes into results/hoop/ (not results/) so it doesn't overwrite the
hover-task artefacts already in results/best_model.zip.

Usage:
    conda activate drones
    python train_hoop_find.py
    python train_hoop_find.py --timesteps 2000000 --out ../results/hoop
    python train_hoop_find.py --watch          # open the PyBullet window DURING training
"""
import argparse
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from gym_pybullet_drones.utils.enums import ObservationType, ActionType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hoop_env import HoopFindAviary  # noqa: E402

OBS = ObservationType.KIN
ACT = ActionType.VEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="max training timesteps (harder task than hover, needs more)")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="parallel env copies (uses more CPU cores)")
    parser.add_argument("--out", type=Path, default=HERE.parent / "results" / "hoop",
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
        HoopFindAviary,
        env_kwargs=dict(obs=OBS, act=ACT, gui=args.watch),
        n_envs=args.n_envs,
        seed=0,
    )
    eval_env = make_vec_env(HoopFindAviary, env_kwargs=dict(obs=OBS, act=ACT), n_envs=1, seed=1)

    print("[INFO] action space:     ", train_env.action_space)
    print("[INFO] observation space:", train_env.observation_space)

    # --- agent ------------------------------------------------------------
    model = PPO("MlpPolicy", train_env, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=OUTPUT_DIR,
        log_path=OUTPUT_DIR,
        eval_freq=2000,          # per-env steps between evaluations
        n_eval_episodes=10,      # reward varies a lot with hoop placement, average more of them
        deterministic=True,
        render=False,
    )

    # --- train ----------------------------------------------------------
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, log_interval=10)
    model.save(f"{OUTPUT_DIR}/final_model.zip")

    print(f"\n[DONE] trained {model.num_timesteps} steps in {time.time() - t0:.0f}s")
    print(f"[DONE] best model -> {OUTPUT_DIR}/best_model.zip")
    print("\nWatch it find the (randomly placed) hoop with:")
    print(f"    python fly_hoop_demo.py --model {OUTPUT_DIR}/best_model.zip")


if __name__ == "__main__":
    main()
