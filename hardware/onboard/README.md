# onboard/: flying the Tello through a hoop with its own camera

Three parts, like `hardware/README.md`:
[what this is](#part-1-what-this-is) · [how to use it](#part-2-how-to-use-it) · [how it works](#part-3-how-it-works).

---

# Part 1. What this is

Fly a real DJI Tello **through** a blue ring, using **only the drone's own front camera**.
Record every flight in a form that behaviour cloning and reinforcement learning can both train on.

**Why this exists separately from `hardware/`.** The rest of `hardware/` implements the *simulator's*
task, and two facts make it the wrong tool for this job:

1. **The simulator's task is not flying through a hoop.** `scripts/hoop_env.py` rewards `-distance` to a
   *point* and **terminates the moment the drone is within 0.15 m of it**. Yaw is held by a PD loop and is
   never commanded by the policy, so the policy cannot orient to a ring plane; and on hardware, success
   calls `request_land("reached_hoop")`. A policy trained there is rewarded for *stopping at the centre*.
   It has no concept of a ring and could not learn one. Flying through needs a different task.
2. **That path needs an overhead camera you do not have.** A real flight there builds `OverheadCamera`
   unconditionally, and without it `est.ready` is never true and `pos_health` is permanently `LOST`, so
   takeoff is refused for ever. It also needs four printed ArUco markers on the cage floor, one on the
   drone, a webcam about 3.5 m up, and two calibration flights.

This folder needs none of that: no overhead camera, no markers, no calibration flight. It needs a blue
ring and a charged battery.

**Status, stated plainly.** Everything here is built and tested **without a drone** (159 tests), and
calibrated to the **measured** cage rather than an assumed one:

| | measured |
|---|---|
| ring opening (the gap the drone flies through) | **30 cm** |
| noodle thickness | 6 cm |
| outer diameter (what the camera sees) | **42 cm** |
| ring centre height | **120 cm** |
| approach runway | **110 cm** one side, 90 cm the other |
| ring offset from the cage centreline | 40 cm |
| aircraft width (no prop guards) | **14 cm** |
| **radial clearance, drone centre to noodle** | **8 cm** |

Eight centimetres is the entire error budget, and it is what every gate in `expert.py` is set
against. The simulator scores a pass by asking whether the drone's *edge* clears the noodle, not
whether its centre is inside the opening — see `sim.py::_check_crossing`.

**The scripted expert flies it. A behaviour clone does not.**

| | result at the measured 30 cm ring |
|---|---|
| expert, ideal profile | 94 % (47/50) — the 3 failures are safe aborts, 0 strikes |
| expert, typical profile | **100 % (50/50)** |
| expert, harsh profile | 86 % (43/50), 2 strikes |
| expert, real OpenCV detector in the loop | **60/60** |
| expert, through the real flight loop (`--dry-run`) | 10/10 typical, 8/10 harsh |
| **behaviour clone** (`onboard_clone`, analytic-model demos), 12 episodes | **11/12 against the analytic model, 6/12 against the real detector** |
| behaviour clone trained on `--render` demos (`onboard_clone_render`), 12 episodes | **0/12 against both — 12 strikes** |

The clone is the honest headline. It copies the expert closely on held-out data, yet it is far
below the expert's 100 % / 60 of 60: it falls from 11/12 to **6/12 once the real detector's noise
is in the loop**, and the `--render`-trained clone strikes every time. (12 episodes, seed 100, so
one or two episodes either way is noise.) Why the `--render`-trained clone strikes every time has
not been investigated. Cloning an expert whose behaviour hinges on an explicit threshold (the
commit gate) smooths across that threshold, which is the likely reason a clone has less slack than
the expert at an 8 cm clearance, but that has not been confirmed. Closing the gap is open work, not
a solved problem.

**Nothing has flown on a real Tello yet**, and the detector has never seen a real camera
frame — step 1 below is what fixes that, and it involves no flying.

---

# Part 2. How to use it

Set up a shortcut once, from the `CMS` folder:

```powershell
$PY = "C:/Users/miche/miniconda3/envs/drones/python.exe"
& $PY -c "import djitellopy, cv2, numpy, yaml; print('ok')"
```

## Stage A — no drone

```powershell
& $PY -m pytest hardware/onboard/tests -q      # 159 tests, no hardware
& $PY hardware/onboard/fly_hoop.py --dry-run --verbose
& $PY hardware/onboard/fly_hoop.py --dry-run --attempts 10 --max-travel 1.8
```

The dry run flies `sim_drone.SimDrone` through **the real flight loop** — the same detector, observation
builder, safety checks and landing path that fly for real. You should see it reach `commit`, then
`sim: PASSED radial 0.04 m`.

Train a policy with no drone at all. (`--render` is optional: it helped at the old 50 cm ring but
not at the measured 30 cm ring, see below.)

```powershell
& $PY hardware/onboard/make_demos.py --episodes 50 `
      --out hardware/flights/demos/onboard_real.npz
& $PY hardware/imitation/bc_train.py --demos hardware/flights/demos/onboard_real.npz `
      --out hardware/flights/bc/onboard_clone --eval-episodes 20
& $PY hardware/onboard/fly_hoop.py --dry-run --mode policy `
      --model hardware/flights/bc/onboard_clone/bc_model.zip
```

`--render` puts the real OpenCV detector in the loop while the demonstrations are generated, instead
of the fast analytic model of it. It is several times slower and it matters a lot. Measured on the
same clone:

At the **50 cm** ring this folder was originally built against, the choice decided whether a clone
flew at all:

| Demonstrations generated with | Scored against the analytic model | Scored against the real detector |
|---|---|---|
| the analytic model (fast) | 12/12 | **5/12** |
| `--render` (real detector) | — | **10/12** |

A clone trained on the analytic model learns that model's exact numbers and falls apart on the real
detector's noise. The lesson still holds: **whatever a learned policy will be asked to fly against,
generate its demonstrations against that.** On real hardware that means real flights
(`fly_hoop.py --record`).

At the **measured 30 cm** ring this reverses. Re-scored on 12 episodes (seed 100):

| Demonstrations generated with | Against the analytic model | Against the real detector |
|---|---|---|
| the analytic model (default) | 11/12 | 6/12 |
| `--render` (real detector) | 0/12 | 0/12 |

So at 30 cm the one `--render`-trained clone we have is worse than the default clone. That is one
clone each, not a controlled comparison. The scripted expert passes 60/60 against the real detector, because its decisions are
explicit gates rather than a fitted function. The likely difficulty for a clone is that the commit
decision is a discontinuity and regression averages across both sides of it, but that is a
hypothesis, not a measurement.

## Stage B — measure your ring, with the drone on the floor

**Measure three numbers first:** the ring's **opening diameter**, the **noodle thickness**, and the ring's
**centre height**. The first two set the range estimate; getting them wrong scales every distance.
The values below are the ones measured for this cage, and `spec.py` already holds them — pass your
own only if your ring differs.

Power the drone on, join its `TELLO-xxxxxx` Wi-Fi, then:

```powershell
& $PY hardware/onboard/tune_detector.py --ring-opening 0.30 --ring-noodle 0.06
```

A window opens with the camera view beside its blue mask, and sliders for every threshold. **Nothing
flies.** Point the drone at the ring and check three things:

* the ring is outlined and nothing else is;
* the outline **survives being moved to the edge of the frame** — that is the `clipped` path, which the
  whole approach depends on;
* the printed range roughly matches a tape measure at two distances.

Press **S** to save `detector.json`. If the range is off by a constant factor the error is in your
measured diameter or `--hfov`, not the thresholds.

## Stage C — first flights, one step at a time

> Cage closed, nobody inside. Battery above 30 %. Phone Tello app closed.
> **A hand on Ctrl+C the whole time** — it requests a landing.

Each step must work before the next. They are flags on the same script.
**Every step below is also a button in the dashboard's Drone tab** (`webapp/README.md`),
including keyboard flight and saving the run, so you can do all of Stage C without a terminal.
`$PY` must be set in your terminal first (see the top of Part 2).

**Step 1 — hover test.** Takes off, sweeps yaw left/right while climbing to find the ring,
holds position once it does, then lands. Never commands forward motion.

```powershell
& $PY hardware/onboard/fly_hoop.py --mode hover --max-airborne 20
```

**Step 2 — alignment only.** Yaw and height tracking work, and in the right direction.
No forward motion. Replaces `calibrate_axes.py`.

```powershell
& $PY hardware/onboard/fly_hoop.py --no-forward
```

**Step 3 — bounded approach.** A short forward run, capped at 0.4 m of commanded travel,
no blind push. After this run, **measure how far it actually travelled** against `travel_m`
in the log and set `--commit-speed` to the real metres per second — the blind push duration
is computed from it.

```powershell
& $PY hardware/onboard/fly_hoop.py --no-commit --max-travel 0.4
```

**Step 4 — full flight.** The complete run: align, approach, and blind push through the ring.

```powershell
& $PY hardware/onboard/fly_hoop.py --max-travel 1.8
```

**Step 5 — fly it yourself and record it.** (New to imitation learning? The full walkthrough,
with what each number means, is in [`../imitation/README.md`](../imitation/README.md).) You fly from the keyboard; the same camera
observations and the same safety layers apply, and the run is recorded in exactly the form
`bc_train.py` reads. Install pygame once (it reads held keys), and try the keys with nothing
spinning first:

```powershell
& $PY -m pip install pygame
& $PY hardware/onboard/fly_hoop.py --dry-run --mode human --record --countdown 0
```

A small window opens: **click it so it has focus.** Keys:

| Key | Does |
|---|---|
| **W / S** | forward / back |
| **A / D** | **strafe** left / right: slide sideways, the nose keeps pointing ahead |
| **Up / Down arrow** | up / down |
| **Left / Right arrow** | **rotate** left / right: turn in place |
| **L** or **Esc** | land |
| **X** or **Backspace** | emergency motor cut |

Strafe and rotate are two different channels of the policy's action (`lr` and `yaw`), and the
expert uses both: it rotates to point the nose at the ring and strafes to get onto the ring's
axis. Rotating alone cannot fix a sideways offset, which matters with only 8 cm of clearance,
so use both in your demonstrations.

Ctrl+C in the terminal still lands. A held key commands 60 % of full deflection
(`--human-gain`, default 0.6) because the expert is continuous, and a full-deflection key
would record actions the policy can never reproduce. Then the real flight:

```powershell
& $PY hardware/onboard/fly_hoop.py --mode human --record --max-travel 1.8 --max-airborne 60
```

During the 5 s countdown click the window. Press **L** when you are done. **Once the drone has
landed** it asks:

```
Save this run (N samples)? [y/n]
Did it go through the ring? [y/n]
```

`n` to the first throws the run away (do this for every `--dry-run` practice run, so
simulated flights never reach your training data). A run that went through is saved as
`demo.npz` and one that missed as `demo_failed.npz` in that flight's folder, so the training
glob in step 6 takes passes only. A flight that ended in an error, Ctrl+C or a motor cut is
never offered for saving.

Recording starts at the first time the ring is seen and **includes the blind push** after it
leaves the frame (the push is the part a clone most needs to see). This also applies to
`fly_hoop.py --record` with the scripted expert.

**Step 6 — train a clone on your flights.** `--demos` takes several files and globs:

```powershell
& $PY hardware/imitation/bc_train.py --demos hardware/flights/tello_*/demo.npz `
      --out hardware/flights/bc/human_clone --eval-episodes 20
```

**Be realistic about the result.** Even a perfect scripted expert's clone scored only 6/12
against the real detector at the 30 cm ring (see the table in Part 1). Human keyboard demonstrations are noisier and far
fewer: expect dozens of flights, a Tello battery gives only a handful each, and the 8 cm
clearance has to be judged by eye. Treat the clone as a baseline and the recorded flights as
a real-hardware dataset, not as a likely first pass.

**Step 7 — fly the cloned policy.**

```powershell
& $PY hardware/onboard/fly_hoop.py --mode policy `
      --model hardware/flights/bc/human_clone/bc_model.zip --max-travel 1.8
```

`--max-travel` is **the integral of the forward command over the whole flight** — the approach plus
the entire blind push — not the displacement from takeoff. Measured through this loop it reaches
1.24–1.37 m on the typical profile and up to 1.63 m on harsh, so **use 1.8**. A smaller budget
truncates the push, and it does so *quietly*: the clamp fires near the end, the drone coasts through
anyway, and the log looks clean right up until a slower speed calibration leaves it stopped in the
ring plane.

Useful flags: `--min-height` / `--max-height` (default 25–160 cm; the ceiling must clear your ring's
centre height with room for overshoot — at a 120 cm centre the ring's top is 141 cm and the expert
peaks at 120–129 cm), `--max-airborne` (default 40 s), `--ring-opening`, `--ring-noodle`,
`--drone-width` (default 14 cm, no prop guards; use 0.18 with them, which cuts the clearance from
8 cm to 6 cm), `--hfov`, `--commit-range`.

**Step 8 (optional) — mix your flights with a PyBullet run, then improve the mix with RL.**
Your flights and the PyBullet runs (`results/hoop/runs/`) are different tasks: the PyBullet policy
reads 82 numbers of true state (78 for runs before 2026-10-10 evening; both work as teachers, and
so do `--reward budget` runs, which use the same 82), this task 33 camera numbers, and a flight
records no true state.
So the bridge goes the other way: `make_teacher_demos.py` lets a PyBullet policy fly the onboard
simulator (which does know the true state) and records what the camera would have seen. The
scene is first rotated so the ring sits where the PyBullet ring always hangs (`--frame canonical`,
the default): measured with `ppo_20261010-011316`, that took the teacher from 0/20 passes (aiming
~15 cm low and to one side) to 13/20. Only passed episodes are kept.

```powershell
# 1. teacher demos from your best PyBullet run (prints its pass rate)
& $PY hardware/onboard/make_teacher_demos.py --run results/hoop/runs/ppo_20261010-011316 --episodes 150
# 2. one clone on both; your flights count x3 per sample (--source-weight human=3, the default)
& $PY hardware/imitation/bc_train.py --demos hardware/flights/tello_*/demo.npz `
      hardware/flights/demos/teacher_*.npz --out hardware/flights/bc/mixed --eval-episodes 20
# 3. RL from that clone, the same demos still pulling it towards them (fading 1.0 -> 0.1)
& $PY hardware/onboard/train_onboard.py --init hardware/flights/bc/mixed `
      --demos hardware/flights/tello_*/demo.npz hardware/flights/demos/teacher_*.npz `
      --timesteps 500000 --run-id onboard_mixed --out results/hoop/runs/onboard_mixed
# continue it later: --init results/hoop/runs/onboard_mixed (steps carry on)
```

`train_onboard.py` freezes the actor for the first 20k steps while the critic learns
(`--critic-warmup-steps`), then after every PPO update takes as many steps on the demo loss
(`--bc-coef`, `--bc-coef-min`, `--bc-decay-steps`). Measured on the typical profile, 20 fixed
evaluation layouts, 150k steps each: the guided run went 15% → 55% → 75% steadily; PPO from
scratch reached 90% at 60k but then collapsed to 0% for 40k steps before recovering to 70%.
The simulator cannot tell you whether your style transfers better to the real Tello; `--bc-coef`
is how much you bet on it. The run's `best_model.zip` flies with `fly_hoop.py --mode policy`
like a clone, and the dashboard lists it (Training panel: Environment "Onboard camera").

**Looping and landing (planned, not built yet).** PyBullet runs can now learn endless
back-and-forth passes (`train_hoop_multi.py --laps 0`, see the main README). The onboard side
still needs the same `laps=0` option in `gym_env.py`/`sim.py`: a pass flips the entry side, and
the next pass counts once the drone is 0.35 m clear. The teacher's yaw-to-ring controller
already turns the nose back toward the ring after a pass, so the student can learn the
turn-round from teacher demos. Landing stays scripted, never learned:
- force-stop already exists (Ctrl+C, `land.flag`/`kill.flag`, dashboard Stop);
- landing on a net hit is planned as a `FlightRunner._safety_overrides` check that requests a
  landing when the IMU sees an impact spike, or when commanded and measured velocity disagree
  for more than 0.3 s.

## Robust training

The simulator used to be easy in ways the real cage is not: one drone, a ring that barely moved, a
perfect detector, always the same front runway. `OnboardSim` can now randomise all of that, off by
default (`dr=0` reproduces the old behaviour exactly, so older tests and demos are unchanged).
The dashboard's Training panel (Environment "Onboard camera") exposes it and turns it on.

```bash
python hardware/onboard/make_demos.py --episodes 300 --dr 1 --sway-deg 5 --wash-deg 15 \
    --hard-start-p 0.3 --laps 0 --seed 1000 --out hardware/flights/demos/onboard_sim_laps.npz
python hardware/onboard/train_onboard.py --until-stopped --n-envs 8 \
    --init results/hoop/runs/onboard_laps/best_model.zip \
    --demos hardware/flights/demos/onboard_sim_laps.npz hardware/flights/demos/teacher_*.npz \
            hardware/flights/tello_*/demo.npz \
    --source-weight human=10 --critic-warmup-steps 30000 \
    --dr 1 --sway-deg 5 --wash-deg 15 --hard-start-p 0.5 --laps 0 --run-id onboard_v4 --out ...
```

| Flag | What it does |
|---|---|
| `--dr` (0..1) | domain randomisation strength (below) |
| `--sway-deg` | largest residual swing of the ring on its rope (default 5°, was 25°) |
| `--wash-deg` | largest tilt of the ring when the drone's prop wash pushes it away (default 15°, 0 = off) |
| `--laps` | 0 = endless back-and-forth (through, clear the ring, turn round, find it, again); 1 = one pass per episode |
| `--hard-start-p` | share of episodes starting away from the front runway |
| `--dr-steps` | `dr` and the hard starts ramp from 0 to full over this many steps, counted from the run's own start (0 = full at once); evaluation is always at full strength |
| `--[no-]bc-augment` | mirror half of each demo batch left/right and add small observation noise (default on) |

**What `dr` randomises** (each episode, from the episode's seed):
- The drone: every `Profile` number is drawn between "ideal" and "harsh", `dr` of the way out from
  "typical"; speed scale ±25% and yaw-rate scale ±20% (the rc→speed calibration is unverified, which is
  probably the biggest sim-to-real gap).
- The ring's placement (height, sideways, yaw) widens. A small residual swing (a pendulum about
  the knot at its top, 1.5–3 s period, with a small twist) stays. The main movement is now the
  prop-wash push (see below). The camera, the pass/strike scoring and the dashboard all use the
  ring's live pose.
- The detector: noise on cx, cy, size and axis ratio (range follows the size, as in reality), and a
  one-tick camera lag in 15% of episodes.

**Hard starts** (`start_type` in the step `info`): `front` (the measured runway), `behind`
(0.45–0.75 m back on the far side), `beside` (in the ring's plane, 0.35–0.7 m to a side), `below`
(under it, 0.7–0.9 m up) and `anywhere`. Yaw is random, so the ring is often out of view and the
policy must turn and look (`found`, `age` and `last_centre` in the observation exist for this).
- With hard starts the ring plane is **open**: crossing it clear of the noodle (under or around the
  ring) is no outcome and the episode goes on. Passing from either side counts. Only a pass or a
  strike ends it; there is no "missed".
- A one-time bonus `R_FOUND` (+0.5) the first time the ring comes into view when it started out of
  view, and a 40 s episode limit (30 s for front starts). All reward weights are saved in
  `meta.json` (`reward`, and `robust` for the settings).

**Why the sim starts in the air.** Your human demos begin when the drone already hovers because
the Tello takes off by itself and the recorder starts at the first ring lock
(`fly_hoop.py`); the simulated episode starts at that same point. Robust starts add the cases the
demos don't cover.

**Expert check** (scripted expert, `make_demos.py`, full randomisation): 92% pass from the front
runway, 24% from "anywhere", 21% beside, 33% behind and 0% below, which is what RL has to learn.
`make_demos.py --dr 1 --hard-start-p 0.5` keeps only the passed episodes (228 of 400 in the first batch).

**Not done for real flights.** `fly_hoop.py`'s forward-travel budget and "record at the first
lock" assume a front start and a single pass. Before flying the real Tello from behind or beside
the ring, or for laps, those limits need relaxing. A lap flight covers ~1 m forward per pass, and
`--max-travel` would stop it after the first. A net-contact land trigger is also still to do.

**Live view.** Onboard runs now write the same files as PyBullet runs (`telemetry.jsonl`,
`camera_latest.jpg`, the real-time flag), so the dashboard shows the drone and ring, the drone's
camera and the Reward card while they train.

### Why the policy lined up but didn't go through

Watching the onboard policy, it could line itself up with the ring but then would not commit to
entering it. From hard starts it often flew into the net. I measured the best lap-mode policy
(`onboard_laps/best_model.zip`) on 30 front starts and 30 hard starts (full randomisation, 25°
swing). Four things were wrong, two in the reward, one in the simulator and one in the
observation:

| Measured | Cause |
|---|---|
| Front starts: 23 of 30 ended in a strike, 7 out of the cage | 10 of 12 traced strikes came **right after a pass**: the drone stopped 4–10 cm past the plane, blind with the ring behind it, and the swinging ring hit it |
| Hard starts: 18 of 30 ended **in the net** | Out-of-cage cost −10. Searching to the time limit without a pass cost −12 to −18 in time penalty, so the net was the cheaper way out |
| It lines up and hovers instead of entering | A strike (−20) cost exactly what a pass earned (+20), so an attempt only paid if the policy was more than 50% sure. Below that, hovering in front was the best it could do |
| Passes were mostly luck under the swing | The ring swung up to 25° (45° in `onboard_20261011-212643`): its centre moved ±9–15 cm, against **8 cm** of clearance and about 1 s of blind flight at the end |

**The fixes** (`gym_env.py`, `sim.py`, `obs.py`, `expert.py`; tests in
`tests/test_v4_onboard.py`):

1. **A crash is never cheaper than carrying on.** Every failure (strike, out of the cage, miss)
   now also costs the time penalty of every tick left in the episode (`drain`, the same idea as
   the PyBullet budget mode's "the net drains the budget"), on top of a fixed penalty: strike
   −8 (was −20), out of cage −10, miss −5. Ending an episode early can no longer beat flying
   on. Compared with hovering, an attempt now pays off at about 29% confidence (8 / (20 + 8))
   instead of 50%.
2. **A "find it" term in the shaping.** The potential gains `W_FACE × cos(angle from the camera
   to the point 0.3 m behind the ring along the pass direction)`. It pays for turning towards a
   ring that is out of view. After a pass the pass direction flips, so the same term pays for
   **turning back round** to find it again. Being potential-based, it can't be farmed: a full
   spin on the spot or hovering earns nothing (a test checks this). This gives the order you
   asked for, as one smooth signal: find (facing), align (the axis pull), go through (the
   approach-then-exit path and the +20).
3. **Passes count only camera-first.** A pass in the right direction counts only if the camera
   faces the way the drone is going (`forward · pass direction ≥ 0.5`). Reversing or sliding
   sideways blind through the opening scores nothing (phase `blind_pass`). Backing straight back
   through the wrong way still costs `R_WRONG_WAY`.
4. **A new observation number, `pushed` (33 → 34).** It is the forward distance flown since the
   ring was last seen fresh, integrated from the Tello's own forward speed `vgx`.
   - **Why:** with the ring out of view, "just before the plane, keep pushing" and "just through
     it, clear the ring and turn round" looked identical. `found`, `age` and `last_range` are the
     same in both. Now `pushed < last_range` means the ring is still ahead, and
     `pushed > last_range` means the drone is through.
   - **On the real drone:** the same `ObsBuilder` computes it from the same telemetry, so it
     flies as is. It inherits the existing unverified `vgx` unit (cm/s assumed, like the `vel`
     field), so apply `calibrate_speed.py`'s result to both together.
   - **Old data still works:** 33-wide demos are upgraded on load by replaying the odometer from
     their stored `vel`/`age` columns (`obs.upgrade_obs`). 33-input policies are widened when
     continued (zero weights on the new input, so they start out flying exactly as before), and
     `policy.py` still flies an old one by dropping the column.
5. **The ring moves like the real one.** The real noodle doesn't swing back and forth by
   itself. The drone's prop wash pushes it away as the drone comes close: it tilts up about the
   rope at its top, the bottom moving away from the drone, and settles back once the drone has
   gone.
   - **Model (`--wash-deg`, 15° default):** the push fades over about 0.45 m from the ring
     centre, each episode draws 50–150% of it, it follows through a damped spring, and its
     direction fades smoothly through the ring's plane (no snap-back into a drone that is
     halfway through). The ring's face can now tilt (`Ring.tilt_deg`), so the camera's ellipse,
     the pass/strike test and contact all see the tilt.
   - **Residual swing:** 5° (`--sway-deg`, was 25°).
   - **15° is an estimate:** film an approach and measure the real tilt.
6. **The scripted expert flies laps** (`ExpertConfig.laps`, `make_demos.py --laps 0`). After its
   blind push it carries on 0.15 m to clear the ring, then turns one way at constant height
   until the ring is back in view, and goes again. The demos used to stop at the first pass, so
   a clone had never seen what to do after one. With full randomisation, the prop wash and 30%
   hard starts it averages 1.6 passes per episode (up to 5). Failed tails (a strike after the
   last pass) are cut from the demos.

**Start from hover, not the floor.** The real Tello cannot be flown off the floor with stick
commands: its motors only start with the firmware `takeoff` command, which climbs to about
0.8 m by itself, and `fly_hoop.py` hands control to the policy only after that (takeoff → about
2 s settle → policy). So the simulator starts there too, at 0.8 ± 0.1 m (`TAKEOFF_Z`). Learning a
takeoff would train something the real drone never does. Robustness comes from varying where
control is handed over: height, position, facing and the hard starts.

## If something goes wrong

| What you see | What it means |
|---|---|
| `No camera frame within 6 s` | It refuses to take off blind. Check the Wi-Fi and that nothing else holds the video stream |
| `Airborne height N cm is outside ...` | Raise `--max-height`, or the drone is drifting vertically |
| `requires at least N%` | Battery below the threshold for that phase |
| `Mainboard reports N C` | Let it cool. The Tello runs hot on the ground with the stream on |
| `commit push clamped ... It will stop short` | `--commit-speed` is too low for `--commit-range`; measure it at step 3 |
| It searches and never finds the ring | Redo stage B in the lighting you are actually flying in |
| It stalls about half a metre out | The commit gate is not being met. The log's `axis_ratio` and `range_m` say which condition failed |

---

# Part 3. How it works

```
Tello front camera (RGB) ──> ring_detect ──> obs ──> controller ──> spec.action_to_rc ──> send_rc_control
  30 Hz, 960x720            blue mask,      33 floats   expert |      clamp to RC_MAX      15 Hz
                            ellipse fit               policy | hover
                                  │                        │
  Tello telemetry (10 Hz) ────────┴────────────────────────┘
      battery, height, attitude, velocity
                                  │
            every tick: safety (battery, temperature, height, travel, time, flags) ──> land
            every tick: ticks.jsonl + status.json, read by the dashboard
```

| File | What it is |
|---|---|
| `spec.py` | **The task definition**: the 33-number observation layout, the 4-number action, the normalisation, and the rc clamp. Everything else imports it, so the sim, the real flight, the recorder and the trainers cannot disagree about what a number means |
| `ring_detect.py` | Blue ring detection and tracking. Adapted from the kit's `perception.py` (credited in its docstring) with every threshold parameterised |
| `obs.py` | Builds the observation from a detection plus one telemetry packet. `ring_from_obs` is the inverse |
| `expert.py` | The scripted visual-servo controller, and `ExpertPolicy`, which gives it a learned policy's interface |
| `human.py` | `--mode human`: the keyboard as a controller with the expert's call signature, so a human flight is recorded in the same form |
| `sim.py` | A lightweight simulator: ring, drone dynamics, pinhole camera. Can produce a detection analytically (fast, for RL) or render a frame for the real detector |
| `sim_drone.py` | A pretend Tello with djitellopy's API, so `--dry-run` exercises the real flight loop |
| `gym_env.py` | A Gymnasium environment, so the task can be trained by RL |
| `fly_hoop.py` | The flight script and all the safety layers |
| `make_demos.py` | Demonstrations from the expert in simulation |
| `policy.py` | Loads a trained policy and refuses one belonging to another task |
| `tune_detector.py` | Stage B: ground-only threshold tuning |
| `kit.py` | The only file that imports Tello-Friend-Kit |

## The three decisions that shaped it

**1. Yaw points the camera; strafing moves the drone.** One camera cannot distinguish "my nose is off to
the side of the ring" from "I am level with the ring but standing off to one side of its axis". Both show
up as the same horizontal offset, and they need opposite corrections. The resolution uses a second,
independent measurement: a circle seen off its own axis projects to an **ellipse**, so `axis_ratio` says
how far off the ring's axis the drone is while the horizontal offset says where the nose points. So yaw
tracks the offset, and lateral position is corrected by strafing until the ring looks circular. The
ellipse gives no sign, only magnitude, so the strafe direction is found by trying one and reversing if the
ring did not get rounder.

**2. The last half metre is blind, and nothing can change that.** Once the ring fills the frame its
outline leaves the field of view, so too little arc remains to fit and detection stops. Everything before
`COMMIT` exists to make that blind push start from a good enough position. Its duration is computed from
the last measured range and an assumed forward speed, not hardcoded, so recalibrating the speed corrects
the push automatically.

**3. Thresholds are in metres, not pixels.** An image offset means nothing on its own: 0.08 of a half-frame
is 2.2 cm of error at 0.42 m but 9 cm at 1.6 m. Since the question is always "will the drone fit through a
30 cm hole", the gates are metres and `spec.offset_metres` converts. This was not cosmetic — with the
gates in pixel units the expert passed 19/30 on the harsh profile; in metres, 30/30. It is also what let
the whole task be recalibrated from a 50 cm ring to the measured 30 cm one by changing constants.

## Safety

Every layer reuses the Tello-Friend-Kit primitives that have actually flown this drone.

| When | Check |
|---|---|
| before takeoff | cage height; battery ≥ `minimum_battery_percent`; mainboard under 80 °C; on the ground; **a camera frame received** |
| every tick | battery against the lower in-air threshold; under 85 °C; height inside `--min-height`/`--max-height`; a 3 s frame watchdog; the airborne budget; the **forward-travel budget** |
| any time | `land.flag` or `kill.flag` appearing in the flight directory |
| always | the `finally` block zeroes the rc and lands; `takeoff_attempted` is set **before** `drone.takeoff()`, so a lost acknowledgement still lands |

Two of these are worth explaining.

**The forward-travel budget** substitutes for position feedback. There is no overhead camera, so the only
way to bound where the drone goes is to bound how far it has been *told* to go: commanded forward speed is
integrated and forward is cut at the limit. It over-counts while accelerating and under-counts while
coasting, which is why it is a safety bound and not a measurement.

**Flag files, not a socket**, for the dashboard's commands. If the backend dies no flag appears and the
flight's own safety still lands the drone; if the flight dies a flag is simply never read. Both flags only
ever make the drone safer.

`survey.health` from the kit is reused for battery and temperature but **not** for height: its 30–120 cm
window was chosen for a camera survey that hovers over its takeoff spot, and this task has to fly at the
*ring's* height. A ring at 1.15 m plus normal overshoot exceeds 120 cm, which aborted every flight until
the window was separated out and made configurable.

## Training by any method

`spec.py` is what makes the controller slot interchangeable.

* **Scripted** — `expert.py`, the baseline that generates demonstrations.
* **Behaviour cloning** — `make_demos.py` then `hardware/imitation/bc_train.py`, which now reads the task
  from the demo file's metadata and picks the right observation shape and loss. The simulator's
  cosine-direction loss is correct for its direction-plus-speed action but wrong for four independent rc
  channels, which want plain MSE; existing simulator demos are unaffected.
* **RL** — `gym_env.py`. Its reward is deliberately not the simulator's sphere-reach: the geometry is split
  into distance *along the ring's axis* and distance *from* it, and progress on both is rewarded, so the
  approach **direction** matters. Crossing the plane ends the episode with a large bonus through the
  opening, a penalty into the noodle, and a smaller one for missing entirely.

Two findings from making cloning work are worth keeping in mind if you change the expert:

* **A discontinuous expert is hard to clone.** A one-tick jump from 0.28 to 1.0 forward throttle appears
  once per episode (about 40 samples in 6000) while its continuation appears 45 times, so a clone learned
  to *continue* a push but never to *start* one, and stalled every time. The approach now ramps smoothly
  into the commit throttle, which puts the same information in every tick.
* **The observation has to carry what the policy needs when it cannot see.** During the blind push,
  `found = 0` and the range reads "far" — which is exactly what searching at the start of a flight looks
  like, and those demand opposite actions. The observation therefore carries where the ring was **last
  seen**. Without those fields a cloned policy flew to within 17 cm of the ring and then hovered until the
  episode timed out, 0/12. With them, it goes through.

## What is not verified

* **No real flight.** Everything about how your drone responds is open.
* **The detector has never seen a real camera frame.** Stage B is what fixes that.
* **`send_rc_control` has no precedent in the kit**, whose motion is all blocking `move_*`, so the rc sign
  conventions are assumptions until step 2.
* **The rc-to-speed scale** (`RC_PER_MPS = 125`) is inherited from a tutorial measurement, not measured on
  this drone. Step 3 measures it.
* **The simulator is a model, not evidence.** It is my model of a Tello, like the project's other fake
  drone. It proves the software logic; it says nothing about your aircraft.
* A ring hung near the **centre** of a 2 m cage leaves roughly 1 m of approach. Moving it toward one wall
  buys runway and makes step 4 considerably easier.
