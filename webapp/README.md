# Hoop finder dashboard

A local web interface for the hoop-finding task: see the drone, hoop and 2x2x2 m cage in 3D,
set the start positions or randomise them with a seed, pick a training algorithm, start and
stop training, watch the reward curve and the drone's own onboard camera live, and generate
video replays afterward.

## Run it

```bash
conda activate drones
uvicorn main:app --app-dir path/to/CMS/webapp/backend --port 8000
```

Then open `http://localhost:8000`. Stop it with Ctrl+C.

This installs three extra packages into the `drones` env the first time (already done if you
followed this session's build): `fastapi`, `uvicorn[standard]`, `psutil`.

```bash
pip install fastapi "uvicorn[standard]" psutil
```

## Layout

```
webapp/
  backend/     FastAPI app: REST API, WebSocket, serves the frontend and result videos
  frontend/    plain HTML/CSS/JS, Three.js and Chart.js loaded from a CDN, no build step
```

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
