# Drone RL, Assignment 3 (CMS)

A simulated quadrotor (PyBullet, via [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones),
trained with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)) that learns two
things: how to hover, and how to find a hoop placed at a random spot inside a 2x2x2 m cage every
episode. The 2x2x2 m cage matches the footprint of a planned real-hardware demo, so the task is
built around that constraint from the start.

For a full, step-by-step guide to recreating this entire system from nothing, see
[tutorial/TUTORIAL.md](tutorial/TUTORIAL.md). For exact environment setup commands, see
[SETUP.md](SETUP.md). For the full chronological build log and open questions, see
[PROGRESS.md](PROGRESS.md).

![Trained hover policy](results/hover_trajectory.png)

---

## How to run it

This assumes the `drones` conda environment already exists. If it doesn't yet, start with
[SETUP.md](SETUP.md) or [tutorial/TUTORIAL.md](tutorial/TUTORIAL.md).

### The web dashboard (easiest way to play with it)

A local website with a live 3D view of the drone and hoop, a choice of training algorithm,
start/stop controls, a reward chart, an onboard drone camera, and video replay. No coding needed
once it's running.

```bash
conda activate drones
cd path/to/CMS/webapp
python -m uvicorn main:app --app-dir backend --port 8000
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
| `scripts/train_hoop_multi.py` | Same hoop task, with a choice of algorithm and a run-scoped output folder. This is what the webapp itself calls, and it also works standalone from the terminal. |

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
├── assets/        the basketball court texture
└── results/       trained models, evaluation logs, videos, and webapp run data
```

| Want to change... | Edit... |
|---|---|
| Room size, reward, random hoop/start position, action space | `scripts/hoop_env.py` |
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

Still improving, not plateaued yet. The full diagnosis of the `pid` to `vel` fix is in
[tutorial/TUTORIAL.md](tutorial/TUTORIAL.md) and [PROGRESS.md](PROGRESS.md). Training curves,
full rollout diagnostics, and video for both policies are in `tutorial/walkthrough.ipynb`.

---

See [PROGRESS.md](PROGRESS.md) for what's next and what's still open.
