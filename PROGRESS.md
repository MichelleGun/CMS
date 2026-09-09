# Progress log

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

- Target hardware for any real-world demo? (Tello = velocity commands only;
  anything lower-level needs a PX4/Betaflight build.)
- Single-agent only, or is multi-agent (`MultiHoverAviary`) in scope for ASS3?

## Housekeeping notes

- A stray earlier install left ~25 dependency packages in **system Python 3.14**
  user-site (`%APPDATA%\Python\Python314\site-packages`). Harmless but can be removed:
  `"C:/Python314/python.exe" -m pip uninstall -y numpy scipy pybullet transforms3d matplotlib pillow sympy networkx cloudpickle colorama cycler farama-notifications filelock fonttools fsspec iniconfig kiwisolver MarkupSafe mpmath packaging pluggy pygments pyparsing setuptools six typing-extensions`
- The real simulator lives at `~/gym-pybullet-drones` (a git clone, not part of this
  folder). This folder only holds our own scripts, results, and docs.
