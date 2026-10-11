# Progress log

## Done (2026-10-11, late night): count passes, streaks, same side, every area (v5)

- **Why:** v4 still never made a second pass in PyBullet (best 0.7 passes per episode), and both
  policies passed mostly from one corner:
  - PyBullet: 70% of episodes started where the last one ended.
  - Onboard: 50% of episodes started on the front runway.
  - `best_model.zip` followed reward, and the dashboard's best run followed success rate.
- **Counting passes** (`scripts/telemetry_callback.py`, `webapp/`):
  - `best_model.zip` follows mean passes per evaluation episode (ties broken by reward).
  - Every evaluation goes to `passes.json` (mean, max, per zone); `zone_passes` is in
    `live_status.json`.
  - The best run is the one with the most passes per episode, chosen per environment (PyBullet
    and onboard; `best_run_ids`).
  - The chart has a passes-per-episode line; the HUD shows passes this episode, the next pass's
    streak multiplier and the start zone.
- **Streak bonus:** pass k pays ×min(1 + 0.5·(k−1), 3), in both sims. In PyBullet the extra stays
  out of the budget tally, and the pass count is in the observation (83 → 84).
- **Same-side laps** (`lap_dir="same"`, both sims): back round the outside to the entry side
  before the next pass counts. Back through the opening is wrong-way (PyBullet −0.5). The shaping
  leads round via a point 0.55 m beside the ring.
- **Onboard heading memory** (34 → 37 numbers): `pass_heading` and `streak`, from a telemetry-only
  pass detector (`obs.PassDetector`). A ring looks the same from both sides, so without it the
  camera drone can't tell front from back.
  - Old 33/34-wide demos are upgraded on load and old policies widened (or still flyable).
  - The expert orbits round the ring to the entry side, searches with a full turn at ring height,
    and moves on if the ring isn't in view.
  - New demos `onboard_sim_laps_same.npz`: 184/360 episodes kept, 0.77 passes per episode, max 4.
    Every zone has passes; under the hoop is hardest (8/40).
- **Start zones** (`start_mode="zones"`, both sims): 9 zones round-robin (8 outer cells of a 3×3
  grid plus under the hoop). Evaluation is 18 episodes, each zone twice.
- **Dashboard:** "Same side every time", "Start from every area" and "Streak bonus" in the
  Training panel. The backend was restarted to serve them (logs in `webapp/dashboard.*.log`).
- **Tests:** 456 pass (67 `scripts/tests`, 330 `hardware`, 59 `webapp/backend/tests`).
- **Runs:**
  - v4 stopped: `ppo_20261011-221214` cleanly at 39.1M steps (best 0.7 passes per episode, max 1).
    `onboard_v4_20261011-221406` was force-stopped (terminal-started) at 29.8M steps (best 0.65
    passes per episode, max 3).
  - Live since 22:54: `ppo_20261011-225450` (v5, from `-221214`, dashboard) and
    `onboard_v5_20261011-225528` (from `onboard_v4` best, terminal, BC on the new demos + teacher +
    human ×10). _TODO: passes per episode and per zone._

## Done (2026-10-11, night): find → align → through → repeat

- **Diagnosis** (read-only traces of the best checkpoints):
  - PyBullet `ppo_20261011-212417` (v3, 37M steps): 3 of 10 episodes passed once, none twice
    (`max_passes` 1 in every run, ever), and 7 of 10 parked without touching the rim.
    - The lap "armed" flag (the next pass counts only 0.28 m clear) was not in the observation.
    - Budget mode has no signal for the 4–6 s turn back.
  - Onboard `onboard_laps` (best): 23 of 30 front starts ended in a strike, 10 of 12 traced
    strikes came right after a pass, and 18 of 30 hard starts ended in the net.
    - Out-of-cage (−10) was cheaper than searching (−12 to −18).
    - Strike (−20) equalled a pass (+20), so the policy only attempted when more than 50% sure:
      it lined up and hovered.
    - The 25–45° pendulum swing against 8 cm of clearance made passes luck.
    - "Just before the plane" and "just through it" looked identical with the ring out of view.
- **PyBullet (MDP v4):**
  - The `armed_obs` flag (82 → 83). Older policies are widened with zero weights
    (`vec_util.widen_policy_input`).
  - `budget_shape` potential shaping on top of the budget, kept out of the budget tally.
  - `start_z` hover starts (0.8 ± 0.1 m), like the Tello after its automatic takeoff.
  - The `wash_deg` prop-wash ring push.
  - Swing 5° (was 20°).
- **Onboard:**
  - Failures drain the rest of the episode's time cost (strike −8, out of cage −10, miss −5,
    plus the drain).
  - A facing ("find it") potential.
  - Passes count only camera-first.
  - New observation `pushed` (33 → 34): forward distance since the ring was last seen. Old
    demos are upgraded on load and old policies widened or still flyable.
  - The prop-wash ring with a tilting face.
  - The scripted expert flies laps; new demos `hardware/flights/demos/onboard_sim_laps.npz`
    (190/300 kept, 1.59 passes per episode, max 5).
- **Dashboard:** "Ring push from prop wash" and "Start hovering" in the Training panel. The swing
  defaults to 5°. Budget runs default to `budget_shape` 2. 83-input runs are comparable for
  "best run". Robust demo generation now makes lap demos.
- **Why each change, in detail:** README "Why each v4 change was made", and
  `hardware/onboard/README.md` "Why the policy lined up but didn't go through".
- **Tests:** 441 pass (60 `scripts/tests` incl. 14 new, 324 `hardware` incl. 12 new or updated,
  57 `webapp/backend/tests`).
- **Live runs** (started 2026-10-11 22:12 / 22:14, until stopped; `ppo_20261011-212417` and
  `onboard_20261011-212643` were stopped cleanly):
  - `ppo_20261011-221214`: v4, continued from `-212417` at 38.0M steps, 6 envs, budget 10,
    +5 per pass, shaping 2, hover start, wash 15°, swing 5°. _TODO: passes per episode._
  - `onboard_v4_20261011-221406`: from `onboard_laps/best_model.zip` (widened), 8 envs, laps,
    dr 1, wash 15°, swing 5°, 50% hard starts, BC on lap-expert + teacher + human demos
    (human ×10), critic warm-up 30k. _TODO: passes per episode._
- **GitHub:** model checkpoints and run summaries under `results/hoop/runs/` are now committed
  (see `.gitignore`), so teammates get the latest training files.

## Done (2026-10-11, later)

- **Dashboard fixes.**
  - The Best replay now plays the **open run's own best policy**, not the all-time best run, and
    every replay takes a fresh seed (it used to reuse seed 1000, so every replay had the same ring,
    start and mild wind). History → "Open this run" now selects that run.
  - Locked controls look greyed out (a run loading, or no free cores) and say why; unchanged
    "Start from" lists are no longer rebuilt every 2.5 s, which could close an open dropdown.
  - **The gentle swing at 45°:** the slider is the *largest* angle. The swing is wind ramp (0 → 1 over
    `--wind-steps`, 1.5M by default) × a per-episode gust 0–1 × the slider. A new "Swing ramps up
    over" field and a HUD line (`ring swing this episode …`) make that visible.
- **Reward (MDP) editor** in the Training panel and a read-only card per run:
  `HoopNetAviary(reward_weights=...)`, `train_hoop_multi.py --reward-weights`, saved in `meta.json`,
  reused by replays and showcases.
- **Run names, stars and the best run.** Rename (✎) and star (★), kept in `label.json`. Date and
  time always show. The best run is picked by eval success rate among runs on today's env
  (obs size 82), not reward, which is not comparable across reward modes.
- **Onboard runs look dead in the dashboard: fixed.** `train_onboard.py` used to register only the
  eval and stop callbacks, so no `telemetry.jsonl` or camera frames were written. It now has the
  telemetry, camera and real-time callbacks (`OnboardHoopEnv.telemetry()` / `.onboard_frame()`;
  watching never touches the sim's random stream), and a Reward card with the imitation term.
- **Onboard robustness** (`hardware/onboard/sim.py`, `gym_env.py`, `train_onboard.py`):
  - domain randomisation of the drone and its speed calibration, ring placement, a **swinging ring**
    (like the PyBullet env's wind), detector noise and lag;
  - **hard starts**: behind, beside, below, or anywhere, often facing away, with an open ring plane
    (flying under or around is no outcome) and a one-time search bonus;
  - a ramp-up curriculum counted from each run's own start, evaluation always at full strength;
  - demo augmentation (mirror left/right, small noise) so the few human demos go further;
  - robust scripted-expert demos: 228 of 400 passed, 39,839 samples. The expert passes 92% from the
    front under full randomisation and 0–33% from hard starts.
  - Details: the "Robust training" section of `hardware/onboard/README.md`.
- **Tests:** 400 pass (`hardware/onboard/tests` incl. 13 new robustness tests, `scripts/tests`,
  `webapp/backend/tests`).
- **Live runs** (started 2026-10-11 03:26, until stopped):
  - `onboard_20261011-032612`: 8 envs, continued from `onboard_20261011-025518`, six human demos ×10 +
    teacher + robust expert demos, dr 1, swing 15°, 50% hard starts. First 11 evals: success 20–30%
    at full difficulty (down from 50–90% on the easy task, which is expected).
  - `ppo_20261011-032617`: 6 envs, continued from `ppo_20261011-025335`, budget 10, +5 per pass,
    sway 20°. Success 10–20% at ~18.5M steps. _TODO: update when it moves._

## Done (2026-10-11)

- **MDP v3: a pass budget instead of shaping** (`HoopNetAviary(reward_mode="budget")`,
  `train_hoop_multi.py --reward budget --budget 3`).
  - **Reward:** −0.005/step, +1 per pass, −0.05 per rim clip, no shaping.
  - **Termination:** passes never end the episode (endless back-and-forth). It ends when the
    running total reaches −3, at the net (which drains the budget), on a tip-over, or at the
    120 s safety cap.
  - **Why:** v2 run `ppo_20261010-182515` reached 14.5M steps with 0% passes. Telemetry
    showed it parking 18–31 cm in front of the ring, because the shaping paid about +20 for
    getting there and a pass attempt risked −3 at the rim.
  - **Viewers:** `showcase_hoop.py` and `replay_hoop.py` now rebuild the env from the run's
    `meta.json` (`env_kwargs_from_meta`). Before, they always ended at the first pass.
  - **Eval stats:** eval reports passes per episode (`live_status.json`:
    `latest_mean_passes`, `best_mean_passes`, `max_passes`).
  - **Tests:** 41 in `scripts/tests` pass.
  - **New run:** `ppo_20261011-013455` continues from 182515 (sway 35°, until stopped).
  - **Docs:** README has an "MDP changes and why" table (v1 → v2 → v3).

## Done (2026-10-10, evening)

- **Net-env training fixed and restarted.** `ppo_20261010-093427` was stopped at 12.2M steps.
  It peaked at 30% eval success but averaged about 6% over its last 50 evaluations, and had
  been flat since about 1.5M steps. The diagnosis: PPO overwrote the policy every update
  (`approx_kl` ≈ 0.9, `clip_fraction` ≈ 0.8, action std 0.06), the policy couldn't see the
  swinging ring's velocity, and the wind was at full strength from step 0. Changes:
  - `HoopNetAviary` obs is now 82-D (ring velocity and opening radius added before the relative
    vector, so `obs[-3:]` is unchanged).
  - Potential-based shaping, a centring-weighted pass bonus (+20, up to +10 more), rim −3.
  - The net ends the episode with −5 (`net_terminate`, default on).
  - `laps=0` endless back-and-forth with re-arming, and an `episode_sec` option.
  - `train_hoop_multi.py`: a PPO preset for the net env, `WindCurriculumCallback` and
    `--wind-steps`, plus `--laps`, `--episode-sec` and `--[no-]net-terminate`. Everything is
    recorded in `meta.json`.
  - The teacher bridge (`make_teacher_demos.py`) and `ObsBuilder` accept the 82-D layout and
    still accept 78-D.
  - Tests: 36 in `scripts/tests` and 285 in `hardware/` pass.
  - New run `ppo_20261010-182515` started (sway 35°, until stopped).
  - README: new net-env section with training-footage placeholders.

## Done (2026-10-10, later)

- **Up to 3 training runs side by side, each with truly parallel envs.** Env copies now run in
  their own processes (`--vec auto` → `SubprocVecEnv`, shared code in `scripts/vec_util.py`):
  measured with `scripts/bench_vec.py`, the PyBullet net env does 1702 env-steps/s with 4 copies
  vs 806 before (8 copies: 2258), and 3 such runs at once still manage ~944 each. Learning is
  unchanged (same data per PPO update), only wall-clock time. The dashboard allows 3 live runs
  (counting ones started from a terminal) and refuses a run that would need more cores than are
  free (16-core PC: budget 14). Stop still saves cleanly: env workers ignore the Ctrl+Break and
  are closed by the trainer.
- **Continue training from an earlier policy.** `--init <run dir or zip>` on `train_hoop_multi.py`
  (and "Start from" in the Training panel): the step count, curriculum and eval curve carry on;
  refuses a different algo or env.
- **Human flights + PyBullet runs → one onboard policy.** `hardware/onboard/make_teacher_demos.py`
  (PyBullet policy flies the onboard sim as a privileged teacher, in a frame aligned to the ring
  pose it trained on: 0/20 → 13/20 passes for `ppo_20261010-011316`), `bc_train.py --source-weight`
  (human ×3 by default), and `hardware/onboard/train_onboard.py` (PPO from a clone with a fading
  BC term and a frozen-actor critic warm-up). Clones on the sim, 20 episodes: human-only 0/20,
  teacher-only 6/20, mixed 2/20 (18 near-miss strikes); guided RL from the mixed clone 75% after
  150k steps (2 min). Details: `hardware/onboard/README.md` Step 8.
- Note: `ppo_20261010-023538` (the overnight run) times out on every onboard-sim episode as a
  teacher; its latest checkpoint scored 0% success in its own last evaluation too.

## Done (2026-10-10)

- **The ring now swings about the rope knot at its top, and the swing is adjustable.** The noodle hangs from a
  rope tied to its top, so `HoopNetAviary._update_ring` no longer slides the ring rigidly: the knot
  (`ring_knot`) only drifts a little downwind (`sway_top_amp`, 2 cm at 20°) and the ring rotates about it,
  so the bottom swings furthest (at the 20° default: centre ±8 cm, bottom ±14 cm). The swing is set with
  `sway_deg` (constructor arg, `set_sway_deg()`, `--sway-deg` on `train_hoop_multi.py` / `showcase_hoop.py` /
  `replay_hoop.py` / `fly_hoop_demo.py`) and recorded in each run's `meta.json`. On the dashboard it is
  a "Max swing" slider on the Environment tab (live preview) and in the Training panel (sent as `sway_deg`).
  The two sliders stay in sync. `HoopSpec.sway_amp` has been replaced by `sway_tilt_deg` (20) and
  `sway_top_amp`.
- **Sim ring now matches the scanned noodle, sways in the wind, and is blue.** `scripts/measure_scan_hoop.py`
  fits the blue texels of the phone scan (placed exactly as the dashboard overlays it): centre (+0.04, +0.07,
  1.29) m, face turned −29° and leaning 18° down, tube-centre radius 0.180 m (the real 0.18 m, so the overlay
  scale is right). `HoopSpec` uses that pose (`Z_RANGE` follows it). `HoopNetAviary` gained live wind
  (`wind=1.0`, `set_wind()`): per-episode wind direction/gust/period, the ring swings ≤6 cm and leans ≤5°, and a
  rim hit shoves it (damped spring back, ≤15 cm). The observation carries the live pose. All sim rings
  (PyBullet, dashboard, scan viewer) are blue; the Environment tab has a "Wind sway" preview.
  Note: the tape-measured height in `hardware/onboard/README.md` is 1.20 m; the scan says 1.29 m.

## Done (2026-10-06)

- **Tello hardware bridge (`hardware/`)** so the sim-trained policy, keyboard flying and imitation learning can run on
  a real DJI Tello. Nothing has flown yet; everything is tested offline against a simulated Tello (`FakeTello`) and
  the real simulator. See `hardware/README.md` (safety section first).
  - The policy sees the same 75-number observation as `HoopFindAviary`; `sim_check/obs_parity.py` asserts it
    against the live environment (max difference 0.0).
  - Position: overhead ArUco camera fused with Tello dead-reckoning in a Kalman filter (blackout-tolerant,
    latency-aware, timestamp-corrected, yaw health-checked). Tuning guide in `state_estimation/TUNING.md`.
  - Safety: SPACE hold, L safe land, X / BACKSPACE motor cut, plus global Ctrl+Shift+X/L/H hotkeys; geofence on
    the action, battery/tilt/telemetry/estimator/time-cap triggers, height-aware landing escalation.
  - Imitation: demos recorded in the sim's formats, behaviour cloning into a PPO policy the existing tools can load.
    Sim-only check: a fly-straight expert reaches the hoop 60/60, its clone 30/30 (the 500k PPO: 9/20).
  - 99 offline tests (8 need opencv + pygame). Real-drone steps (axis + speed calibration, first flights) are still to do.

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

0. **Loops on the real drone and net-hit landing.** Lap training is done in both sims
   (2026-10-11 night). Still open for real flights:
   - A net-contact land trigger in `fly_hoop.py` (IMU spike, or commanded vs measured
     velocity mismatch).
   - Relax `fly_hoop.py`'s forward-travel budget and "record at the first lock" for laps and
     for starts away from the front runway.
   - Measure the real prop-wash tilt from a video of an approach and set `--wash-deg` to match
     (15° is an estimate).
   - Verify the `vgx` unit (the `pushed` odometer and the `vel` field both assume cm/s).
1. **Stabilise training** — drop PPO learning rate to `1e-4`, add a fixed seed sweep,
   confirm the 40k-step reward dip disappears. Log to TensorBoard
   (`pip install tensorboard`, uncomment the `tensorboard_log=` line in
   `quickstart_train.py`).
2. **Harder action space** — retrain hover with `act="rpm"` (4 independent motors)
   and `act="pid"` (position setpoints), compare sample efficiency.
3. **New task — trajectory following** — switch env to `FlyThruGateAviary` or
   write a custom circular-waypoint reward on top of `BaseRLAviary`.
4. **Robustness / domain randomisation** — done for the onboard-camera task (see above). Still open
   for hover / the PyBullet envs: randomise mass, motor gain and control latency each reset and
   measure how much hover reward drops.
5. **Compare algorithms** — SAC vs PPO on the same task (SB3 makes this a one-line
   change).
6. **Report + short video** — record a GUI rollout (`play.py --record_video true`),
   write up method + results for the assignment.

## Open questions

- Before a real flight from behind or beside the ring: `fly_hoop.py`'s forward-travel budget and
  "record at the first lock" assume a front start and need relaxing. Monday's new human demos
  should include other starting positions.
- How good is good enough on the robust onboard task before trying the real Tello (success rate at
  full randomisation)? The old 90%-success run `ppo_20261010-011316` is on an older env version
  (no swing setting, other observation layout), so it is not comparable and is never picked as best.

- ~~Which physical drone flies the 2x2x2 m cage demo?~~ Answered: a DJI Tello, driven with velocity
  commands, so `HoopFindAviary`'s `vel` policy can run on it (see `hardware/`). Still open: how well it transfers
  (takeoff height mismatch, response lag, see `hardware/README.md` section 5).
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
