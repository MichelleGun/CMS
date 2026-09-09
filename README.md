# Drone RL — Assignment 3 (CMS)

Reinforcement learning for quadrotor control, built on
[`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) (PyBullet physics)
and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO.

**Status:** first milestone done — a PPO agent learns to take off and hold a hover
at 1.0 m altitude. Mean reward **474.1 / 474** ("perfect") over 20 evaluation episodes,
trained in **160k steps (~6.5 min on a laptop CPU)**.

![Trained hover policy](results/hover_trajectory.png)

---

## Why PyBullet and not MuJoCo

| | PyBullet (`gym-pybullet-drones`) | MuJoCo |
|---|---|---|
| Ready-made quadrotor env | **Yes** — Crazyflie model, Gymnasium API, SB3 examples, PID baselines | No maintained drone package; you build the env yourself |
| Physics | Good for rigid-body quad dynamics | Better contacts, generally faster |
| Time to first training run | Minutes | Hours of infrastructure first |

For a first RL-on-drones project the deciding factor is that PyBullet gives a working
hover / waypoint environment on day one.

### Note on "real DJI drone" transfer
Consumer DJI drones (Mini / Air / Mavic) expose **no low-level motor or attitude control**,
so a policy that outputs rotor thrusts has nothing to send them to. Real sim-to-real would
need a DJI/Ryze **Tello** (high-level velocity commands only) or a custom PX4 / Betaflight
build. That is out of scope for this assignment — we train in simulation.

---

## The task (`HoverAviary`)

- **Agent:** one Crazyflie 2.X quadrotor
- **Goal:** reach and hold position `(0, 0, 1.0)` m for an 8 s episode
- **Observation** (`kin`, 27-dim): position, orientation, linear & angular velocity, plus a short action history
- **Action** (`one_d_rpm`, 1-dim): a single scalar per step, applied as equal RPM to all four motors — the simplest action space, learns fastest
- **Reward:** shaped, ~474 ≈ a stationary hover exactly on target

---

## Repository layout

```
CMS/
├── README.md                 <- this file
├── SETUP.md                  <- exact environment setup (Windows + conda)
├── PROGRESS.md               <- what's done / what's next
├── requirements-lock.txt     <- exact package versions used
├── tutorial/
│   ├── TUTORIAL.md           <- full walkthrough: install PyBullet -> hover -> court
│   └── walkthrough.ipynb     <- runnable notebook of the headless-safe steps
├── scripts/
│   ├── quickstart_train.py   <- PPO training, early-stops when solved
│   ├── quickstart_eval.py    <- headless evaluation + trajectory plot
│   ├── court.py              <- builds a virtual basketball court in PyBullet
│   └── fly_in_court.py       <- fly the drone around the court (PID tour or RL policy)
├── assets/
│   └── court_texture.png     <- top-down court markings (auto-generated)
└── results/
    ├── best_model.zip        <- trained policy (load with PPO.load)
    ├── final_model.zip       <- policy at end of training
    ├── evaluations.npz       <- reward vs timesteps (EvalCallback log)
    └── hover_trajectory.png  <- altitude / drift of the trained policy
```

`gym-pybullet-drones` itself is **not** vendored here — it is installed as a dependency
(see `SETUP.md`). These scripts import it as a normal Python package.

---

## Reproduce

Once the `drones` conda environment exists (see [SETUP.md](SETUP.md)):

```bash
conda activate drones
cd path/to/CMS/scripts

# train headless (4 parallel envs, fast — ~7 min, early-stops when solved)
python quickstart_train.py

# evaluate the saved policy + regenerate the plot (headless, ~30 s)
python quickstart_eval.py

# watch the trained policy fly in the PyBullet GUI
python -m gym_pybullet_drones.examples.play --model_path ../results/best_model.zip
```

### Watching the simulation

**During training** — add `--watch`: a PyBullet window opens and you see the drone go
from flailing to a clean hover as PPO learns.

```bash
python quickstart_train.py --watch
```

This forces a single environment (PyBullet allows one GUI window per process), so it runs
roughly 5–10× slower than the headless 4-env run, and motion looks fast because there is no
real-time throttle. Good for a demo recording; use headless for real training.

**While a headless run is going** — open a *second* terminal and replay the current best
policy. `results/best_model.zip` is overwritten every time training reaches a new best, so
re-run this whenever you want a fresh snapshot:

```bash
python -m gym_pybullet_drones.examples.play --model_path ../results/best_model.zip
```

> If `conda activate drones` fails with `EnvironmentNameNotFound`, your shell's `conda` and
> the env are from different installs (e.g. Anaconda vs Miniconda). Either
> `conda config --append envs_dirs <path>\Miniconda3\envs` once, activate by full path
> (`conda activate <path>\envs\drones`), or just call that env's Python directly:
> `& "<path>\envs\drones\python.exe" quickstart_eval.py`.

---

## Flying in a basketball court

`fly_in_court.py` drops the drone into a virtual FIBA-style court (markings baked
into `assets/court_texture.png`, plus two 3D hoops). The court is scaled down
(`--scale 0.5` → 14 × 7.5 m, rim at ~1.5 m) so the drone's motion stays meaningful.

```bash
cd path/to/CMS/scripts

# PID waypoint tour: centre -> wings -> up to each hoop -> back  (this actually flies around)
python fly_in_court.py
python fly_in_court.py --record        # also save results/video-*.mp4
python fly_in_court.py --scale 0.4     # smaller court

# the trained RL policy, in the court
python fly_in_court.py --rl
```

The **PID tour** uses a classic position controller (`DSLPIDControl`) — it's the
good visual demo. The **`--rl`** mode loads `results/best_model.zip`; because that
policy was trained with the 1-D-thrust action space it can only hold altitude, so
it just hovers at centre court. Making the *RL* agent fly a course means retraining
with a position/velocity action space — see [PROGRESS.md](PROGRESS.md).

![court markings](assets/court_texture.png)

---

## Results so far

Training reward (from `results/evaluations.npz`):

| Steps | Mean reward |
|------:|------------:|
| 8k    | 336 |
| 16k   | 441 |
| 40k   | 219  *(PPO instability — recovered)* |
| 104k  | 466 |
| 152k  | 474 |
| 160k  | **474 → early stop** |

The dip around 40–56k is ordinary early-PPO variance with default hyperparameters;
lowering the learning rate to `1e-4` smooths it.

Final policy (`quickstart_eval.py`, 20 episodes): **474.1 ± 0.0** — the drone climbs to
1.0 m and holds it with millimetre-scale horizontal drift.

---

See [PROGRESS.md](PROGRESS.md) for next steps.
