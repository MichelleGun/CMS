# Drone RL from zero — a full walkthrough

This is the step-by-step version of everything in the project: from a clean
machine with nothing installed, to a PPO agent that hovers a simulated quadrotor,
to flying a drone around a virtual basketball court.

Every command was run on **Windows 11, an ASUS Zenbook 14 (no NVIDIA GPU)**.
Notes for macOS / Linux are inline where they differ.

**Contents**

1. [Pick a simulator: PyBullet vs MuJoCo](#1-pick-a-simulator)
2. [Install the prerequisites](#2-install-the-prerequisites)
3. [Create the environment and install `gym-pybullet-drones`](#3-create-the-environment)
4. [Sanity-check the install](#4-sanity-check)
5. [Understand the RL environment](#5-understand-the-rl-environment)
6. [Write and run the first training script](#6-first-training-script)
7. [Watch / evaluate the trained policy](#7-watch--evaluate)
8. [Read the results](#8-read-the-results)
9. [Build a virtual basketball court](#9-build-a-basketball-court)
10. [Fly the drone in the court](#10-fly-in-the-court)
11. [Where to go next](#11-where-to-go-next)

---

## 1. Pick a simulator

| | **PyBullet** (`gym-pybullet-drones`) | MuJoCo |
|---|---|---|
| Ready-made quadrotor env | Yes — Crazyflie model, Gymnasium API, PPO examples, PID baselines | No maintained drone package; build it yourself |
| Physics | Fine for rigid-body quad dynamics | Better contacts, a bit faster |
| Time to first trained agent | ~20 min including install | Hours of scaffolding first |

For a first RL-on-drones project, PyBullet wins because the environment already
exists. We use [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones)
(the `utiasDSL` / `learnsyslab` version, v2.2.0).

> **"But I want a real DJI drone."** Consumer DJI drones (Mini / Air / Mavic) give
> you *no* low-level motor or attitude control through their SDK, so a policy that
> outputs rotor commands has nothing to talk to. Real sim-to-real needs a
> DJI/Ryze **Tello** (high-level velocity commands only) or a custom
> **PX4 / Betaflight** build. Everything below is simulation.

---

## 2. Install the prerequisites

You need three things on the machine:

| tool | why | Windows | macOS | Linux |
|---|---|---|---|---|
| **Miniconda / Anaconda** | Python environment manager | [installer](https://docs.conda.io/en/latest/miniconda.html) | `brew install --cask miniconda` | [installer](https://docs.conda.io/en/latest/miniconda.html) |
| **Git** | clone the simulator | [git-scm.com](https://git-scm.com) | `brew install git` | `apt install git` |
| **A C++ compiler** | `pybullet` has no pre-built wheel for Python > 3.10, so `pip` compiles it | [VS Build Tools](https://visualstudio.microsoft.com/downloads/) → "Desktop development with C++" | Xcode CLT: `xcode-select --install` | `sudo apt install build-essential` |

Check they're visible from a terminal:

```bash
conda --version
git --version
```

(There's no simple one-liner to check the C++ compiler — you'll find out in step 3
if it's missing, with a `error: Microsoft Visual C++ 14.0 or greater is required`
type message.)

---

## 3. Create the environment

```bash
# clone the simulator
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones

# gym-pybullet-drones 2.2.0 needs Python >= 3.12.
# Using the conda-forge channel avoids Anaconda's Terms-of-Service prompt.
conda create -n drones -c conda-forge --override-channels python=3.12 -y
conda activate drones

# conda-forge's python=3.12 ships without pip, so add it
conda install -c conda-forge --override-channels pip setuptools wheel -y

# install the package + every dependency
# (compiles pybullet from source, downloads PyTorch ~124 MB — allow 10-20 min)
pip install -e .
```

What lands in the env (see `../requirements-lock.txt` for exact pins):

| package | version | role |
|---|---|---|
| `torch` | 2.14.0 **+cpu** | neural nets (CPU only — no NVIDIA GPU on this laptop) |
| `stable-baselines3` | 2.9.0 | the PPO implementation |
| `gymnasium` | 1.3.0 | the RL environment API |
| `pybullet` | 3.2.7 | the physics engine (built from source) |
| `gym-pybullet-drones` | 2.2.0 | the quadrotor environments |

> **`conda activate drones` says `EnvironmentNameNotFound`?** Your shell's `conda`
> and the env are from different installs (e.g. Anaconda vs Miniconda). Fix once
> with `conda config --append envs_dirs <path>\Miniconda3\envs`, or activate by
> full path, or just call that env's Python directly:
> `& "<path>\envs\drones\python.exe" script.py`.

---

## 4. Sanity check

```bash
python -c "import pybullet, torch, stable_baselines3, gymnasium; \
from gym_pybullet_drones.envs.HoverAviary import HoverAviary; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
e = HoverAviary(); e.reset(); print('env OK'); e.close()"
```

Expected:

```
torch 2.14.0+cpu cuda False
...
env OK
```

`cuda False` is correct here — training runs on CPU. That's fine: the policy
network is tiny.

---

## 5. Understand the RL environment

We use **`HoverAviary`**: one Crazyflie 2.X quadrotor whose job is to reach and
hold `(0, 0, 1.0)` metres for an 8-second episode.

```python
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM)
print(env.observation_space)   # Box(..., (1, 27), float32)
print(env.action_space)        # Box(-1.0, 1.0, (1, 1), float32)
```

- **Observation** (`kin`, 27 numbers): position, orientation (quaternion),
  linear & angular velocity, plus a short buffer of recent actions.
- **Action** (`one_d_rpm`, 1 number): a single scalar in `[-1, 1]`, applied as the
  *same* RPM to all four motors. This is the simplest possible action space — the
  drone can only go up or down, not translate — which makes hover very fast to
  learn. (Richer action spaces: `rpm` = 4 independent motors, `pid` = position
  setpoints, `vel` = velocity commands.)
- **Reward**: shaped so that ~474 over an episode ≈ sitting perfectly still on the
  target. The training script uses 474 as the "solved" threshold.

---

## 6. First training script

The full script is [`../scripts/quickstart_train.py`](../scripts/quickstart_train.py).
The important parts:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

OBS, ACT = ObservationType.KIN, ActionType.ONE_D_RPM

# 4 copies of the env so PPO collects experience faster (uses 4 CPU cores)
train_env = make_vec_env(HoverAviary, env_kwargs=dict(obs=OBS, act=ACT), n_envs=4, seed=0)
eval_env  = make_vec_env(HoverAviary, env_kwargs=dict(obs=OBS, act=ACT), n_envs=1, seed=1)

model = PPO("MlpPolicy", train_env, verbose=1)          # default hyperparameters

# stop as soon as the policy is good enough
stop = StopTrainingOnRewardThreshold(reward_threshold=474.0, verbose=1)
eval_cb = EvalCallback(eval_env, callback_on_new_best=stop,
                       best_model_save_path="../results", log_path="../results",
                       eval_freq=2000, n_eval_episodes=5, deterministic=True)

model.learn(total_timesteps=300_000, callback=eval_cb)
model.save("../results/final_model.zip")
```

Run it:

```bash
cd path/to/CMS/scripts
python quickstart_train.py            # headless, ~7 min, early-stops when solved
```

On this laptop it **solved hover in 160k steps / ~6.5 min of CPU** and stopped early.
`EvalCallback` writes `results/best_model.zip` every time it beats its own record,
plus `results/evaluations.npz` (reward vs. timesteps).

> Stable-Baselines3 buffers its console output, so the terminal can look frozen
> for a minute at a time — that's normal.

**Watch it learn live** instead:

```bash
python quickstart_train.py --watch    # opens a PyBullet window; 1 env, ~5-10x slower
```

---

## 7. Watch / evaluate

**Score the saved policy** (headless, ~30 s) —
[`../scripts/quickstart_eval.py`](../scripts/quickstart_eval.py):

```bash
python quickstart_eval.py
```

It runs 20 episodes, prints the mean reward, and saves
`results/hover_trajectory.png` (altitude and horizontal drift over one flight).

**Watch the trained drone fly** in the PyBullet GUI:

```bash
python -m gym_pybullet_drones.examples.play --model_path ../results/best_model.zip
```

---

## 8. Read the results

From `results/evaluations.npz`:

| training steps | mean reward |
|---:|---:|
| 8k | 336 |
| 16k | 441 |
| 40k | 219  ← PPO wobble, recovered |
| 104k | 466 |
| 160k | **474 → early stop** |

The dip near 40k is ordinary early-PPO variance with stock hyperparameters — the
official example just trains 10M steps to ride it out. Lowering PPO's learning
rate to `1e-4` smooths it.

Final policy: **474.1 ± 0.0** over 20 episodes — the drone climbs to 1.0 m and
holds it with millimetre-scale drift.

---

## 9. Build a basketball court

[`../scripts/court.py`](../scripts/court.py) does two things:

**(a) Draw the markings** into a texture with PIL — a top-down FIBA court: boundary,
halfway line, centre circle, both keys, free-throw circles, three-point arcs:

```python
from PIL import Image, ImageDraw
# ... draw lines in court-metre coordinates, save court_texture.png
```

**(b) Build the 3D scene** in a running PyBullet world, using the client id the
environment exposes via `env.getPyBulletClient()`:

```python
import pybullet as p

# textured floor slab
floor = p.createMultiBody(0, floor_col, floor_vis, [0, 0, -0.01])
tex = p.loadTexture("court_texture.png")
p.changeVisualShape(floor, -1, textureUniqueId=tex)

# each hoop = pole (cylinder) + backboard (box) + a ring of small orange spheres
```

Real courts are 28 × 15 m with a 3.05 m rim — too big next to a drone that hovers
at 1 m — so `build_court(client, scale=0.5)` shrinks everything: a 14 × 7.5 m
court with the rim at ~1.5 m, right in the drone's working range.

---

## 10. Fly in the court

[`../scripts/fly_in_court.py`](../scripts/fly_in_court.py) creates a `CtrlAviary`
(a plain simulation env, no RL), injects the court, and flies a **waypoint tour**
using `DSLPIDControl` — the classic PID position controller that ships with
`gym-pybullet-drones`:

```python
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

env  = CtrlAviary(num_drones=1, gui=True, obstacles=False)
info = build_court(env.getPyBulletClient(), scale=0.5)
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

for step in range(N):
    obs, *_ = env.step(action)
    action[0], _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP,
        state=obs[0],
        target_pos=current_waypoint,   # centre -> wing -> up to hoop -> ...
        target_rpy=[0, 0, yaw])
```

```bash
python fly_in_court.py            # PID tour, GUI
python fly_in_court.py --record   # + save results/video-*.mp4
python fly_in_court.py --rl       # the trained RL policy, in the court
```

**Important:** `--rl` loads `best_model.zip`, but that policy was trained with the
1-scalar-thrust action space, so it can only hold altitude — it hovers at centre
court, it does **not** fly the tour. Making the *RL* agent fly a course means
retraining with a position or velocity action space.

---

## 11. Where to go next

1. **Stabilise training** — PPO learning rate `1e-4`, seed sweep, TensorBoard
   (`pip install tensorboard`, uncomment the `tensorboard_log=` line).
2. **Richer action space** — retrain hover with `act="rpm"` (4 motors) or
   `act="pid"` (position setpoints); compare sample efficiency.
3. **New task** — RL waypoint following: a custom reward on top of `BaseRLAviary`
   that rewards tracking a path (e.g. the basketball-court tour). *This* is how you
   get an RL agent to fly the court.
4. **Domain randomisation** — randomise mass / motor gain / latency each reset;
   measure the hover-reward drop. Needed before any real-hardware thought.
5. **SAC vs PPO** — one-line change in Stable-Baselines3, same task.
6. **Write-up + video** — `fly_in_court.py --record`, plus the training curve and
   method notes for the assignment.

---

*Companion notebook: [`walkthrough.ipynb`](walkthrough.ipynb) runs the
headless-safe parts of this (install check, env inspection, a short training run,
evaluation, court build + render) cell by cell.*
