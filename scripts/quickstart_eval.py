"""
Evaluate a trained hover policy headlessly and save a trajectory plot.

    conda activate drones
    python quickstart_eval.py                      # uses ../results/best_model.zip
    python quickstart_eval.py --model path/to.zip

For the live PyBullet GUI instead:
    python -m gym_pybullet_drones.examples.play --model_path ../results/best_model.zip
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write PNG, don't open a window
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "results" / "best_model.zip"
OBS, ACT = ObservationType.KIN, ActionType.ONE_D_RPM

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
parser.add_argument("--out", type=Path, default=HERE.parent / "results" / "hover_trajectory.png")
args = parser.parse_args()

model = PPO.load(str(args.model))

# --- quantitative score over 20 episodes -------------------------------
env = HoverAviary(obs=OBS, act=ACT)
mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=20)
print(f"[EVAL] mean reward over 20 episodes: {mean_r:.1f} +/- {std_r:.1f}  (474 ~ perfect hover)")

# --- one rollout, record altitude & horizontal drift -----------------
obs, _ = env.reset(seed=123)
zs, xy, ts = [], [], []
for i in range((env.EPISODE_LEN_SEC + 2) * env.CTRL_FREQ):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    s = obs.squeeze()
    zs.append(s[2])
    xy.append(np.hypot(s[0], s[1]))
    ts.append(i / env.CTRL_FREQ)
    if terminated or truncated:
        break
env.close()

fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax[0].axhline(1.0, ls="--", c="gray", label="target z = 1.0 m")
ax[0].plot(ts, zs, c="C0")
ax[0].set_ylabel("altitude z (m)"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ts, xy, c="C3")
ax[1].set_ylabel("horizontal drift (m)"); ax[1].set_xlabel("time (s)"); ax[1].grid(alpha=.3)
fig.suptitle(f"Trained PPO hover policy  (mean reward {mean_r:.0f}/474)")
fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=120)
print(f"[EVAL] saved plot -> {args.out}")
