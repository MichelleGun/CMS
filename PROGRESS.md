# Progress log

## Done (2026-09-30, later)

- **Local web dashboard for the hoop-finding task** (`webapp/`). FastAPI backend plus a plain
  HTML/JS/Three.js frontend, no build step. Open `http://localhost:8000` after starting it
  (see `webapp/README.md` for the exact command). Features: a live 3D view of the 2x2x2 m cage,
  hoop and drone, drone/hoop start position set with on-screen arrows or the keyboard (or
  randomised by seed) before a run, algorithm choice (PPO, A2C, SAC, TD3 on the existing
  continuous velocity action, DQN via a new discrete-move wrapper since SB3's DQN needs a
  Discrete action space), start/stop training from the page, a live reward chart (current run
  vs best run on disk), an onboard drone camera feed toggleable live during training (off by
  default, real per-frame cost otherwise), and generated video replay (third-person and drone
  POV) for the current or best checkpoint.
  New scripts: `hoop_env.py` gained optional `start_pos`/`target_pos` overrides (random-by-seed
  stays the default), `discrete_action_wrapper.py`, `train_hoop_multi.py`, `telemetry_callback.py`,
  `camera_capture.py`, `replay_hoop.py`. `train_hoop_find.py` is untouched and still works for
  manual PPO-only runs from the terminal.
  Two real bugs found and fixed while testing end to end: `Image.fromarray` needs uint8, not
  the int64 `p.getCameraImage` returns; and redirecting a GUI subprocess's stdout to an unread
  `subprocess.PIPE` stalls it once the OS pipe buffer fills, `replay_hoop.py`'s own verbose
  per-step printing was chatty enough to hit this, fixed by writing subprocess output to a log
  file on disk instead.

## Done (2026-09-30)

- **Hardware-demo cage (2x2x2 m)** — the real-world demo space is a 2x2x2 m netted volume,
  too small for a scaled-down basketball court. Added `hoop_env.py` (`HoopFindAviary`): one
  hoop, position randomised every episode, drone start position randomised too, so the
  policy has to learn to search rather than memorise a fixed layout. Target-relative vector
  added to the observation (otherwise unsolvable — hidden, moving goal). Action space is
  `vel` since `one_d_rpm` can't move laterally to reach an off-centre hoop (see next bullet
  for why not `pid`, which looked like the obvious first choice).
  `court.py` gained `build_room` / `build_hoop` for the cage + single hoop; `train_hoop_find.py`
  / `fly_hoop_demo.py` mirror the existing quickstart/fly_in_court scripts, writing to
  `results/hoop/` so they don't clobber the existing hover artefacts. Answers the "target
  hardware" open question below (answer: no hardware chosen yet, but the flight *volume* is
  fixed at 2x2x2 m).
- **Found and fixed why the first `HoopFindAviary` training run wouldn't converge** — 400k
  steps with `act="pid"` plateaued at eval reward ≈ -6 and never once reached the hoop (0/20
  eval episodes). Root cause: `pid`'s action is an absolute destination, and in a room this
  small the destination is very often more than the controller's 1 m per-step cap away from
  the drone, so an untrained policy commands ~1 m instantaneous setpoint jumps every 33 ms
  control step. `DSLPIDControl` reacts with an aggressive tilt, which tips the drone past the
  tilt-truncation threshold in ~8 control steps, every episode, before it has moved almost at
  all — the policy never lived long enough to learn anything. Switched to `act="vel"` (bounded
  velocity setpoints, ~0.25 m/s max for the CF2X) — verified with a hand-written
  fly-toward-target policy first (smooth flight, no tumbling, one test episode reached the
  hoop) before spending retraining time. Also loosened the tilt-truncation bound slightly
  (0.4 → 0.5 rad — real banking during fast translation, not loss of control), widened the
  hoop's height range now that it isn't fighting `pid`'s bound (`[0.3, 0.9]` → `[0.3, 1.6]` m),
  and gave episodes more time (`EPISODE_LEN_SEC` 10 → 15 s) to match the slower, honest speed
  limit. Retrained 500k steps on the fixed environment: **9/20 eval episodes reach the hoop**
  (0 before), mean final distance **0.34 ± 0.22 m** (0.84 m before). Eval reward was still
  trending down (better) at 500k, noisily, not clearly plateaued — more steps would likely
  help further; see `results/hoop/evaluations.npz`.
- Rebuilt `tutorial/walkthrough.ipynb` to cover both tasks end to end: training curves,
  full-parameter rollout diagnostics (position vs target, orientation, linear/angular
  velocity, actions), and embedded recorded video for each. Needed `ffmpeg` (PyBullet's GUI
  video logger shells out to it and silently no-ops without it — nothing in this repo's
  `--record` flags actually produced a video before this) and `jupyter`/`nbconvert`, neither
  of which were installed; added to the `drones` env.

## Done (2026-09-09)

- **Decided PyBullet over MuJoCo** — `gym-pybullet-drones` ships a ready-made
  quadrotor env; MuJoCo would need one built from scratch.
- **Environment set up** — `drones` conda env, Python 3.12, PyTorch 2.14 CPU,
  `gym-pybullet-drones` 2.2.0 (pybullet compiled from source). See `SETUP.md`.
- **First RL milestone — hover** — PPO on `HoverAviary`, `one_d_rpm` action.
  Solved (reward 474.1/474) in 160k steps / ~6.5 min CPU.
  Artefacts in `results/`.
- Scripts written: `quickstart_train.py`, `quickstart_eval.py`.
- **Virtual basketball court** (`court.py` + `fly_in_court.py`) — scaled FIBA court
  with textured markings and two hoops. PID waypoint tour flies a lap of the court;
  `--rl` drops the hover policy in (hovers at centre only, as expected for the
  1-D-thrust action space).

## Next

1. **Stabilise training** — drop PPO learning rate to `1e-4`, add a fixed seed sweep,
   confirm the 40k-step reward dip disappears. Log to TensorBoard
   (`pip install tensorboard`, uncomment the `tensorboard_log=` line in
   `quickstart_train.py`).
2. **Harder action space** — retrain hover with `act="rpm"` (4 independent motors)
   and `act="pid"` (position setpoints), compare sample efficiency.
3. **New task — trajectory following** — switch env to `FlyThruGateAviary` or
   write a custom circular-waypoint reward on top of `BaseRLAviary`.
4. **Robustness / domain randomisation** — subclass the env to randomise mass,
   motor gain, and control latency each reset; measure how much hover reward drops.
5. **Compare algorithms** — SAC vs PPO on the same task (SB3 makes this a one-line
   change).
6. **Report + short video** — record a GUI rollout (`play.py --record_video true`),
   write up method + results for the assignment.

## Open questions

- Which physical drone flies the 2x2x2 m cage demo? (Tello = velocity commands
  only; anything lower-level needs a PX4/Betaflight build.) Determines whether
  `HoopFindAviary`'s trained policy can ever run on the real hardware, or whether
  the demo has to be PID/scripted like `fly_in_court.py`'s tour.
- Single-agent only, or is multi-agent (`MultiHoverAviary`) in scope for ASS3?
- How long does `train_hoop_find.py` actually need to converge? Still open: 500k
  steps gets to 9/20 (45%) hoop-reach rate, eval reward still noisily improving,
  not plateaued — worth trying 1-2M steps next to see where it actually settles,
  and/or reward shaping (a denser bonus for closing distance, not just the
  terminal +10) to speed convergence up. No hover-474-equivalent "solved"
  number yet.

## Housekeeping notes

- A stray earlier install left ~25 dependency packages in **system Python 3.14**
  user-site (`%APPDATA%\Python\Python314\site-packages`). Harmless but can be removed:
  `"C:/Python314/python.exe" -m pip uninstall -y numpy scipy pybullet transforms3d matplotlib pillow sympy networkx cloudpickle colorama cycler farama-notifications filelock fonttools fsspec iniconfig kiwisolver MarkupSafe mpmath packaging pluggy pygments pyparsing setuptools six typing-extensions`
- The real simulator lives at `~/gym-pybullet-drones` (a git clone, not part of this
  folder). This folder only holds our own scripts, results, and docs.
