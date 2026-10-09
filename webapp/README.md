# Hoop finder dashboard

A local web interface for the hoop-finding task: see the drone, hoop and 2x2x2 m cage in 3D,
set the start positions or randomise them with a seed, pick a training algorithm, start and
stop training, watch the reward curve and the drone's own onboard camera live, and generate
video replays afterward.

## Run it

```bash
conda activate drones
uvicorn main:app --app-dir path/to/CMS/webapp/backend --port 8000

# or:
# uvicorn main:app --app-dir "C:\Users\miche\OneDrive\Documents\Michelle\UTS\SPR_26\RL\ASS3\CMS\webapp\backend" --port 8000
```

Then open `http://localhost:8000`. Stop it with Ctrl+C.

This installs three extra packages into the `drones` env the first time (already done if you
followed this session's build): `fastapi`, `uvicorn[standard]`, `psutil`.

```bash
pip install fastapi "uvicorn[standard]" psutil
```

## The "Real drone" panel

The bottom panel watches a real Tello flying the onboard-camera hoop task
(`hardware/onboard/`, which has its own README). It shows battery, temperature, height,
ring lock, estimated range, forward travel and the live rc command, plus the drone's own
camera with the detected ring drawn on it.

**It cannot take off a real drone, by design.** `POST /api/drone/flights` accepts dry runs
only; a real flight is started from a terminal by someone standing next to the cage, and
the panel attaches to whichever flight is newest. A page left open in a forgotten tab, a
double click or a reconnecting socket must not be able to spin propellers in another
room. Every button in the panel either reads state or makes the drone safer: **Land now**
and **Emergency stop**.

Commands are flag files (`land.flag`, `kill.flag`, `camera_enabled.flag`) that the flight
script polls each tick, the same mechanism the training camera toggle already uses. That
is fail-safe in the direction that matters: if the backend dies no flag appears and the
flight's own safety still lands the drone.

The "Dry run" disclosure starts a simulated flight — no hardware is touched — which is
the way to see the panel working before there is anything to watch.

## Layout

```
webapp/
  backend/     FastAPI app: REST API, WebSocket, serves the frontend and result videos
  frontend/    plain HTML/CSS/JS, Three.js and Chart.js loaded from a CDN, no build step
```

The real-drone side mirrors the training side exactly: `drone_manager.py` next to
`run_manager.py`, `drone_ws.py` next to `telemetry_ws.py`, `frontend/js/drone.js` next to
the other modules. It reads what `hardware/onboard/fly_hoop.py` writes under
`hardware/flights/<flight_id>/` and never imports djitellopy, OpenCV or the flight code.

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
