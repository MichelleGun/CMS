# Drone RL, Assignment 3 (CMS)

A simulated quadrotor (PyBullet, via [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones),
trained with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)) that learns two
things: how to hover, and how to **take off from the floor, find a hoop and fly through it** inside
a 2x2x2 m net. The simulated cage mirrors the real one used for the hardware demo (a 2x2x2 m net
with a ring near its centre that sways a little in the wind), so a policy trained here can be flown
on a real DJI Tello.

Everything can be driven from a local **web dashboard**: set up and start training, watch the
drone fly in 3D (far view and drone camera), play back what the current policy has learned, and
compare the sim against the phone scan of the real cage.

---

## Quick start (for teammates)

You need **Git**, **Miniconda** and a C++ compiler (Windows: Visual Studio Build Tools, "Desktop
development with C++"). No GPU is needed. Total setup is about 20 minutes, mostly compiling PyBullet.

```bash
# 1. Get this repo (the 33 MB phone scan in assets/ comes with it)
git clone https://github.com/MichelleGun/CMS.git
cd CMS

# 2. Build the Python environment `drones` (simulator, PyBullet, PyTorch, Stable-Baselines3).
#    Follow SETUP.md steps 1-4; the short version is:
#       git clone https://github.com/utiasDSL/gym-pybullet-drones.git   (next to this repo, not inside it)
#       conda create -n drones -c conda-forge --override-channels python=3.12 -y
#       conda activate drones
#       conda install -c conda-forge --override-channels pip setuptools wheel -y
#       cd ../gym-pybullet-drones && pip install -e . && cd ../CMS

# 3. Add the dashboard's packages to the same environment
conda activate drones
pip install -r webapp/requirements.txt

# 4. Start the dashboard (run it from the CMS folder, inside the `drones` environment)
python -m uvicorn main:app --app-dir webapp/backend --port 8000
```

Then open **http://localhost:8000** in a browser (it needs an internet connection: the 3D
view loads three.js from a CDN). Stop the server with `Ctrl+C`. Never add `--host 0.0.0.0`: the
dashboard can fly a real drone and should only be reachable from your own computer.

**First things to try**

1. **Environment** tab: the simulated net and ring. *Randomise ring* shows the per-episode
   variation, *Wind sway* the swinging, *Play example flight* a take-off through the ring, and
   *Overlay phone scan* lays the real cage's scan over the sim net (fitted automatically).
2. **Simulator** tab: choose *Net + ring*, an algorithm and *Fixed* or *Until I click Stop*
   (for overnight runs), then **Start training**. The run view shows the far view with the drone's
   route, the drone's camera, and **Watch current policy** plays one episode with what it has
   learned so far. A *Real-time* checkbox slows everything to real speed so you can follow it.
   By default episodes are varied so the policy gets robust. About 70% start where the previous
   episode ended (as if the drone landed there), the rest anywhere on the floor. Half are flown back
   to front, and about 30% have the ring leaning the other way, as if the wind held it over. Each has a
   checkbox in the Training panel (and `--carry-start`, `--reverse-prob`, `--lean-flip-prob` on
   `scripts/train_hoop_multi.py`).
3. Training saves into `results/hoop/runs/<run id>/` (not committed to git). `latest_model.zip` is
   updated at every evaluation and `best_model.zip` on each new best, so an interrupted run keeps
   its progress.

**Good to know**

- The dashboard starts training in a separate process, so **keep the server window open** while a
  run trains (closing it ends the run). For overnight runs also keep the laptop plugged in; while an
  *Until I click Stop* run trains, Windows is kept from sleeping.
- Training speed depends on your CPU. A 1M-step run takes a few hours on a laptop; around 100-150
  simulation steps per second is typical. With the dashboard's 3D view open it is a little slower.
- If something looks stale after pulling new code, press **Ctrl+F5** in the browser and restart
  the server.
- Windows is what this was developed on. macOS/Linux should work (the same Python code), but the
  *Stop* button and sleep-prevention are only tested on Windows.
- Without a real Tello nothing in `hardware/` is needed. The **Drone** tab has a simulated drone for
  dry runs.

---

For a full, step-by-step guide to recreating this entire system from nothing, see
[tutorial/TUTORIAL.md](tutorial/TUTORIAL.md). For exact environment setup commands, see
[SETUP.md](SETUP.md). For the full chronological build log and open questions, see
[PROGRESS.md](PROGRESS.md).

![Trained hover policy](results/hover_trajectory.png)

---

## How to run it

This assumes the `drones` conda environment already exists and `pip install -r
webapp/requirements.txt` has been run in it (see the quick start above). If the environment doesn't
exist yet, start with [SETUP.md](SETUP.md) or [tutorial/TUTORIAL.md](tutorial/TUTORIAL.md).

### The web dashboard (easiest way to play with it)

A local website with a live 3D view of the drone and hoop, a choice of training algorithm,
start/stop controls, a reward chart, an onboard drone camera, and video replay. No coding needed
once it's running.

```bash
conda activate drones
cd path/to/CMS
python -m uvicorn main:app --app-dir webapp/backend --port 8000
```

Then open `http://localhost:8000` in a browser. Stop it with Ctrl+C.

If `uvicorn` is on your PATH, `uvicorn main:app --app-dir backend --port 8000` (or
`cd backend && uvicorn main:app --port 8000`) also works. If the plain `uvicorn` command isn't
found even after `conda activate drones`, use the `python -m uvicorn` form above instead, that's
the one confirmed to work regardless of PATH quirks.

Full feature list and notes: [webapp/README.md](webapp/README.md).

### From the command line

| Script | What it does |
|---|---|
| `scripts/quickstart_train.py` | Trains the hover policy (PPO, stops early once solved). |
| `scripts/quickstart_eval.py` | Evaluates the saved hover policy and regenerates the trajectory plot. |
| `scripts/fly_in_court.py` | Flies the drone around a virtual basketball court (PID tour, or the RL hover policy). |
| `scripts/train_hoop_find.py` | Trains the hoop-finding policy (PPO only, writes to `results/hoop/`). |
| `scripts/fly_hoop_demo.py` | Watches the trained hoop-finding policy; the hoop moves to a new spot every episode. |
| `scripts/train_hoop_multi.py` | Same hoop task, with a choice of algorithm and a run-scoped output folder. This is what the webapp itself calls, and it also works standalone from the terminal. Defaults to the net + ring environment (`--env net`); `--env legacy` trains the original point-reaching task. |

Run these from `scripts/` with `drones` activated. Full walkthroughs and explanations of each are
in [tutorial/TUTORIAL.md](tutorial/TUTORIAL.md).

### The notebook

`tutorial/walkthrough.ipynb` already has training curves, full rollout diagnostics, and embedded
video for both tasks baked in. Re-run it with
`jupyter nbconvert --to notebook --execute --inplace tutorial/walkthrough.ipynb` if you want fresh
numbers after retraining.

---

## Which file to edit for what

```
CMS/
├── scripts/       drone environments, training scripts, and shared helpers
├── webapp/        the local web dashboard (FastAPI backend + browser frontend)
├── hardware/      flying the policies on a real DJI Tello (see hardware/README.md)
├── tutorial/      TUTORIAL.md and the runnable notebook
├── assets/        the basketball court texture and the phone scan of the real cage (3D_net&hoop.glb)
└── results/       trained models, evaluation logs, videos, and webapp run data
```

| Want to change... | Edit... |
|---|---|
| Net size, ring size/position/facing and its per-episode jitter, net bounce, reward (the default training env) | `scripts/hoop_net_env.py` (`HoopSpec`) |
| The original point-reaching task (`--env legacy`) | `scripts/hoop_env.py` |
| Hover task behaviour | `scripts/quickstart_train.py` (the environment itself, `HoverAviary`, lives in the installed `gym-pybullet-drones` package, not this repo) |
| Basketball court geometry or markings | `scripts/court.py` |
| Add or change a training algorithm in the webapp | `webapp/backend/config.py` (the algorithm list) and `scripts/train_hoop_multi.py` (`ALGO_CLASSES`) |
| What gets logged during training (telemetry, camera, eval snapshots) | `scripts/telemetry_callback.py` |
| Onboard camera capture | `scripts/camera_capture.py` |
| Replay video generation | `scripts/replay_hoop.py` |
| Webapp API, subprocess start/stop | `webapp/backend/main.py`, `webapp/backend/run_manager.py` |
| Webapp interface (3D view, controls, chart) | `webapp/frontend/index.html`, `webapp/frontend/js/*.js` |
| 3D scene shared by the CLI and the webapp's replay generator | `scripts/court.py` (`build_room`, `build_hoop`) |
| Anything about the real Tello: safety limits, calibration, camera, keyboard flying, imitation, running a policy | `hardware/` (settings in `hardware/config.yaml`) |

---

## Real hardware (DJI Tello)

`hardware/` runs the simulator's policies on a Tello in the 2x2x2 m cage: keyboard flying, loading a trained policy,
recording demonstrations for behaviour cloning, an overhead-camera plus dead-reckoning position estimate, and
emergency-stop / safe-landing keys. It reproduces the simulator's 75-number observation exactly (checked against the
real environment) and is fully testable without a drone. **It has not flown a real Tello yet.**
[hardware/README.md](hardware/README.md) has three parts: what the folder is for, a step by step beginner
walkthrough (start here, it begins with steps that need no drone), and a full explanation of what is inside, why,
and how it was made.

```
python hardware/control/keyboard_teleop.py --dry-run          # fly a simulated Tello with the keyboard
python -m pytest hardware/tests -q                            # offline checks
```

---

## Results so far

### Hover

| Steps | Mean reward |
|------:|------------:|
| 8k    | 336 |
| 16k   | 441 |
| 40k   | 219 (PPO instability, recovered) |
| 104k  | 466 |
| 152k  | 474 |
| 160k  | 474, early stop |

Final policy (20 episodes): **474.1 ± 0.0**. The drone climbs to 1.0 m and holds it with
millimetre-scale drift.

### Hoop-finding (hardware-demo cage)

Evaluated over 20 episodes with fresh random hoop and start positions, not seen during training.

| | First attempt (`pid`, 400k steps) | Current (`vel`, 500k steps) |
|---|---:|---:|
| Reached the hoop | 0/20 | **9/20** |
| Mean final distance to hoop | 0.84 ± 0.34 m | **0.34 ± 0.22 m** |

Those numbers are for the legacy point-reaching task. 

### Net + ring environment (`scripts/hoop_net_env.py`)

This environment mirrors the real cage. It has a 2×2×2 m net with a vertical ring of 0.30 m opening near the centre. Every episode varies the ring's position, facing and size to cover wind swing and repositioning. The drone always starts on the floor. Success means flying through the opening in the required direction. Touching the net or the rim bounces the drone back with a penalty, and the episode continues. A scripted expert passes through on all test seeds (`scripts/tests/test_hoop_net_env.py`).

A first 300k-step PPO run (`results/hoop/runs/net_ppo_trial`) learned to take off and fly to the approach point in front of the ring, but has not yet passed through (0/10 evals). It needs longer runs or an imitation warm start.

You can preview it in the dashboard's **Environment** tab, which also overlays the phone scan.

The full diagnosis of the `pid` to `vel` fix is in
[tutorial/TUTORIAL.md](tutorial/TUTORIAL.md) and [PROGRESS.md](PROGRESS.md). Training curves,
full rollout diagnostics, and video for both policies are in `tutorial/walkthrough.ipynb`.

---

See [PROGRESS.md](PROGRESS.md) for what's next and what's still open.
