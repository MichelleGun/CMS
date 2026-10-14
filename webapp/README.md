# Hoop finder dashboard

A local web interface for the whole project, in five tabs. Nothing needs a terminal once the
server is running.

| Tab | What you do there |
|---|---|
| **Environment** | See the simulated 2×2×2 m net and ring: randomise the ring, preview its wind sway, play an example flight, and overlay the phone scan of the real cage (auto-fitted to the sim net) |
| **Simulator** | Train RL policies in the PyBullet simulator: start positions, algorithm, reward curve, onboard camera, video replays |
| **Drone** | Fly the real Tello or a simulated one: the six Stage C steps as buttons, the drone camera, telemetry, Land / Emergency motor cut, **keyboard flight**, and saving the run as a demonstration |
| **Imitation** | List every demonstration (recorded flights and simulated), generate simulated demos, train a behaviour clone on the ones you tick, evaluate clones, and send one to the Drone tab |
| **History** | Every real flight, dry run, training run and job, with charts of range, height, travel and commands over time, the events, and the last camera frame |

## Run it

**First time only**, in the `drones` environment (see [../SETUP.md](../SETUP.md) to create it), from the
`CMS` folder:

```bash
conda activate drones
pip install -r webapp/requirements.txt
```

Then, from the `CMS` folder with `drones` activated:

```bash
python -m uvicorn main:app --app-dir webapp/backend --port 8000
```

Then open `http://localhost:8000` (three.js and Chart.js are bundled in
`frontend/vendor/`, so no internet connection is needed). Stop it with Ctrl+C. **Do not add `--host 0.0.0.0`**: the dashboard can take off a real
drone, and it should only be reachable from this computer.

If `python` is not the `drones` one (conda not activated), call that environment's interpreter
directly, e.g. on Windows `C:/Users/<you>/miniconda3/envs/drones/python.exe -m uvicorn ...`. The
server starts training as a child process using the same interpreter, so it must be the `drones` one.

`pygame` (keyboard flying of a real Tello) is only needed for the Drone tab with real hardware:
`pip install -r hardware/requirements-hardware.txt`.

**Training on the net + ring env (Simulator tab).** To train, set up the **Training** panel and press *Start training*.
- **Environment:** choose *Net + ring*, the default.
- **Ring moves / turns a little each episode:** gives the ring per-episode variation.
- **Wider ring at first:** starts with an easier opening that shrinks to real size.

While a run is live, the main view is a **far view** of training env 0. It shows the net, the ring and the drone, and draws the route as it flies. The last 3 episodes stay as faded trails. Red dots mark net bumps, orange dots mark rim hits and a green dot marks a pass. The **Drone POV** pane shows that drone's onboard camera, and it switches on automatically.
- **Watch current policy:** plays one real-time episode of the policy saved at the last evaluation. That episode has no exploration, so it shows what has actually been learned.
- **During that episode:** the drone turns green, and both views follow it.
- **When it ends:** it reports passed, or missed with the number of net bumps.
- **Speed:** training and *Watch current policy* both run at full speed. Tick **Real-time** in the run view to slow both to real speed, so you can follow the flight; you can switch it during a run.
- **Start-up time:** loading the policy takes 10–30 s while training is running.
- **Evaluations:** training pauses for a few seconds every 20k steps to evaluate the policy, and the run view says so.

**Phone scan overlay (Environment tab).** When loaded, the scan is fitted to the sim net automatically. It's scaled to the 2×2×2 m net, squared to its walls and centred on the floor. The sliders fine-tune that fit, and *Auto-fit again* resets them.

**Environment tab.** The *Environment* button in the header shows the training cage on its own. That covers the net, the ring with an arrow for the pass direction, and the drone on the floor. It has buttons to preview the ring's per-episode variation and play an example take-off-and-fly-through. It can also overlay the phone scan of the real cage, with scale and yaw sliders. Geometry comes from `/api/config` → `hoop`, which is the same `HoopSpec` the env uses.

**Scan viewer.** `http://localhost:8000/scan.html` shows the phone scan in `assets/` in 3D, with a 0.30 m reference ring, to check its scale and that the hoop is open.

**Continuing a run.** *Start from* → pick the run and keep its settings: the policy continues from
its latest save and the step count carries on.
- **The server owns the runs.** Training runs are child processes of this server, so restarting it
  ends them (they then show as "crashed").
- **Memory.** On 16 GB of RAM, two until-stopped runs at once (6 + 8 envs) ran out of memory.

**Troubleshooting.**
- **Startup is per part.** Each part of the page starts on its own. If one fails, a red banner under
  the header names it and the reason, and the tabs and run list still work.
- **WebGL unavailable.** The banner means the browser's graphics acceleration is off: turn it on at
  edge://settings/system (or chrome://settings/system), restart, and check edge://gpu.
- **Use a real browser.** Use Edge or Chrome directly, not VS Code's preview, and hard-refresh with
  Ctrl+F5 after updates.

## Flying from the Drone tab

1. Pick **Dry run** or **Real drone**. For a real flight, join the TELLO Wi-Fi first.
2. Pick a step (1 Hover test ... 6 Fly a trained policy). It fills in the options, which you can
   change.
3. Press **Take off**. A real flight asks you to confirm once, then counts down 3 s.
4. **Land now** (or **L** / **Esc**) and **Emergency motor cut** (or **X** / **Backspace**) are
   always on the right. The cut asks for confirmation when clicked, but not from the keyboard.

**Every flight starts the same way (since 2026-10-14):** takeoff → hover → climb to the
**Start height** (80 cm) → only then do the keys / the policy get control. A low takeoff no longer
aborts the flight. Defaults: **Max height 190 cm**, **Max airborne 150 s**, no travel cap. The camera
shows from the countdown on.

**Keyboard flight (step 5).** A small **pop-up window** opens on this computer (the same one a
terminal flight uses). **Click it**, then fly with W/S (forward/back), A/D (strafe), Up/Down (height)
and Left/Right (rotate); L/Esc lands, X/Backspace cuts the motors. The dashboard keeps showing the
camera and telemetry next to it. This is much smoother than typing into the browser, whose keys went
page → WebSocket → backend → UDP. That path is still available under **Keys from: this browser tab**:
there, the flight hovers if it hears nothing for 0.4 s (switching tab, closing the page). No key is ever
left stuck down.

**Fly a trained policy (step 6).** The picker groups the policies (**Recommended** first, then RL
policies, then clones / DAgger students). Each one shows its name, what it was trained for (same-side
laps, back-and-forth, single pass) and its simulator score from all 9 start zones. The line underneath
says how it was trained.

**After a run** the next **Start** is available as soon as the drone has landed (and, for a recorded
keyboard flight, the save question is answered). The ClearML upload now runs in a separate background
process, so it no longer blocks Start; untick **Upload to ClearML** to skip it.

**Saving a run.** After a recorded keyboard flight lands, the page shows **Went through: save as
pass**, **Missed: save as miss** and **Discard**. The flight waits up to 10 minutes on the ground
for the answer, then discards. A pass is saved as `demo.npz` and a miss as `demo_failed.npz` in that
flight's folder. You can change the label later in the Imitation tab.

## Safety rules the backend enforces

* A real takeoff needs `confirm: true` in the request (the confirm dialog) and must come from this
  computer. A request from any other address gets 403, even with `--host 0.0.0.0`.
* Only one flight at a time, including a flight started from a terminal.
* Land and kill are flag files (`land.flag`, `kill.flag`) that the flight checks every tick. If the
  backend dies, the flight's own safety layer still lands the drone.
* A stop request for a real flight only ever asks it to land. It never kills the process mid-air.

## The Config card: see a run's settings, edit them for the next run

On the Simulator tab, a run's header has a **Config** button; the History tab shows the same card
under a training run's chart. It has two sides for every setting of an onboard-camera run:

* **This run** (left, read-only): what the run was started with, from its `meta.json`. Groups:
  Run (steps, parallel envs, seed, evaluation), Environment (drone profile, randomisation, laps,
  start zones, streak, ring tracker), PPO (learning rate, gamma, GAE lambda, entropy, target KL,
  batch size, epochs, critic warm-up, log-std reset, gSDE), Imitation (demo-term strength and
  decay, human weight) and the 17 **reward weights** (with their defaults). Below them: the
  network's architecture (layer shapes, activation, log-std values, the PPO numbers saved in the model).
* **Next run** (right, editable): a copy you change. Changed rows are highlighted, "Reset to this
  run" undoes it, you pick whether the next run **continues from this run's model**, starts fresh, or
  uses the same parent, and which demos feed the behaviour-cloning term. The **command line** it
  becomes is shown live (and can be copied), and **Start next run** launches it.

Nothing on the card changes the run you are looking at. Net / legacy simulator runs show their
record read-only. A run started from the old Training panel behaves exactly as before.

API behind it: `GET /api/runs/{id}/config` (grouped values, the form description, architecture,
command), `POST /api/runs/command` (the command line for a set of values, nothing started), and the
new optional fields of `POST /api/runs` (`gamma`, `gae_lambda`, `ent_coef`, `target_kl`, `batch_size`,
`n_epochs`, `learning_rate`, `critic_warmup_steps`, `log_std_reset`, `sde`, `ring_tracker`,
`streak_cap`, `bc_coef_min`, `bc_decay_steps`, `reward_weights`, and profile `measured`). The fields
are described once in `backend/run_config.py`; the card renders its form from that description, so a
new setting added there appears in the card without touching the frontend. The architecture is read
by a short subprocess (`scripts/clearml_sync.model_architecture`), so the dashboard still never
imports torch; the first open of a run's card takes a few seconds.

Reward weights are per run: `train_onboard.py --reward-weights '{"r_passed": 25}'` (only the values
that differ from the defaults are sent), recorded in the run's `meta.json` as `reward` (effective
values) and `reward_overrides`.

## Jobs

Generating demos, training a clone and evaluating one run as **jobs**, one at a time. Each writes
its log and record to `hardware/flights/jobs/`, so the History tab shows them after a restart. The
running job's log streams into the Imitation tab, and **Stop** ends it.

## Tests

```powershell
& $PY -m pytest webapp/backend/tests -q
```

They use temporary folders and never start a real process.

## Layout

```
webapp/
  backend/     FastAPI app: REST API, WebSocket, serves the frontend and result videos
  frontend/    plain HTML/CSS/JS, Three.js and Chart.js loaded from a CDN, no build step
```

Backend modules: `run_manager.py` (simulator training), `run_config.py` (the Config card's field
description and command builder), `drone_manager.py` (flights),
`imitation_manager.py` (demos and clones), `jobs.py` (shared subprocess helpers and the job
registry), `drone_ws.py` / `telemetry_ws.py` (live streams). Frontend: one file per tab
(`drone.js`, `imitation.js`, `history.js`; the simulator is `app.js`), plus `tabs.js`,
`keys.js` and `runconfig.js` (the Config card). The backend reads what the scripts write under `hardware/flights/` and
`results/hoop/runs/`, and never imports djitellopy, OpenCV, torch or the flight code.

The backend never imports `stable_baselines3`/`pybullet`/`torch` itself. It only starts,
stops and polls the training and replay scripts in `../scripts/` as subprocesses, and reads
the small JSON/JSONL files they write under `../results/hoop/runs/<run_id>/`. That folder is
the whole database: there is no separate registry, restarting the backend just re-reads it.

## Notes

- One training run and one replay at a time. Replay is disabled while training is active
  (both are CPU-heavy simulation, no GPU on this machine).
- DQN uses a discrete 7-move wrapper around the same velocity controller the continuous
  algorithms use (SB3's DQN needs a `Discrete` action space), so it will likely need more
  timesteps and may cap out at a lower reward than PPO, A2C, SAC or TD3. The UI shows a note
  when DQN is selected.
- The live drone camera and replay video need real CPU time (rendering frames), the camera
  toggle is off by default for that reason. Video replay needs `ffmpeg` on `PATH`; the backend
  adds the `drones` conda env's own `Library/bin` to the subprocess `PATH` automatically, so a
  `conda install -c conda-forge ffmpeg` into that env is enough, no separate PATH setup needed.
- `results/hoop/runs/` is git-ignored already (the existing `.gitignore`'s `runs/` rule).
