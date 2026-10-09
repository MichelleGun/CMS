# Hoop finder dashboard

A local web interface for the whole project, in four tabs. Nothing needs a terminal once the
server is running.

| Tab | What you do there |
|---|---|
| **Simulator** | Train RL policies in the PyBullet simulator: start positions, algorithm, reward curve, onboard camera, video replays |
| **Drone** | Fly the real Tello or a simulated one: the six Stage C steps as buttons, the drone camera, telemetry, Land / Emergency motor cut, **keyboard flight**, and saving the run as a demonstration |
| **Imitation** | List every demonstration (recorded flights and simulated), generate simulated demos, train a behaviour clone on the ones you tick, evaluate clones, and send one to the Drone tab |
| **History** | Every real flight, dry run, training run and job, with charts of range, height, travel and commands over time, the events, and the last camera frame |

## Run it

Open PowerShell, then copy and paste:

```powershell
cd "C:\Users\miche\OneDrive\Documents\Michelle\UTS\SPR_26\RL\ASS3\CMS"
$PY = "C:/Users/miche/miniconda3/envs/drones/python.exe"
& $PY -m uvicorn main:app --app-dir webapp/backend --port 8000
```

Then open `http://localhost:8000`. Stop it with Ctrl+C. **Do not add `--host 0.0.0.0`**: the
dashboard can now take off a real drone, and it should only be reachable from this computer.

First time only:

```powershell
& $PY -m pip install fastapi "uvicorn[standard]" psutil pygame
```

## Flying from the Drone tab

1. Pick **Dry run** or **Real drone**. For a real flight, join the TELLO Wi-Fi first.
2. Pick a step (1 Hover test ... 6 Fly a trained policy). It fills in the options, which you can
   change.
3. Press **Take off**. A real flight asks you to confirm once, then counts down 3 s.
4. **Land now** (or **L** / **Esc**) and **Emergency motor cut** (or **X** / **Backspace**) are
   always on the right. The cut asks for confirmation when clicked, but not from the keyboard.

**Keyboard flight (step 5).** Keep the Drone tab in front and use W/S (forward/back), A/D (strafe),
Up/Down (height) and Left/Right (rotate). The keys go from the page to the backend over a
WebSocket, then to the flight over UDP on this computer. The flight hovers if it hears nothing for
0.4 s, which is what happens when you switch tab or window, close the page or the backend stops.
No key is ever left stuck down.

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

Backend modules: `run_manager.py` (simulator training), `drone_manager.py` (flights),
`imitation_manager.py` (demos and clones), `jobs.py` (shared subprocess helpers and the job
registry), `drone_ws.py` / `telemetry_ws.py` (live streams). Frontend: one file per tab
(`drone.js`, `imitation.js`, `history.js`; the simulator is `app.js`), plus `tabs.js` and
`keys.js`. The backend reads what the scripts write under `hardware/flights/` and
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
