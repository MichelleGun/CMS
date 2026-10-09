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

**Status, stated plainly.** Everything here is built and tested **without a drone** (131 tests), and
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
| **behaviour clone, trained on 60 expert episodes** | **0/12 — 12 strikes** |

The clone is the honest headline. It reaches a validation action-MSE of 0.0002 and a mean cosine of
0.999 against the expert, and still strikes every time: its radial error at the crossing is
**~16 cm median** against an 8 cm budget. The same clone would pass a 50 cm ring 10–12 times out of
12, which is what this folder reported before the ring was measured. Cloning an expert whose
behaviour hinges on an explicit threshold (the commit gate) smooths across that threshold, and at
50 cm there was enough slack to absorb it. At 30 cm there is not. Closing that gap is open work,
not a solved problem.

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
& $PY -m pytest hardware/onboard/tests -q      # 131 tests, no hardware
& $PY hardware/onboard/fly_hoop.py --dry-run --verbose
& $PY hardware/onboard/fly_hoop.py --dry-run --attempts 10 --max-travel 1.8
```

The dry run flies `sim_drone.SimDrone` through **the real flight loop** — the same detector, observation
builder, safety checks and landing path that fly for real. You should see it reach `commit`, then
`sim: PASSED radial 0.04 m`.

Train a policy with no drone at all. **Use `--render`**, for the reason below:

```powershell
& $PY hardware/onboard/make_demos.py --episodes 50 --render `
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

At the **measured 30 cm** ring the choice no longer rescues it — a `--render`-trained clone strikes
0/12 against both the analytic model and the real detector, because the 8 cm clearance is tighter
than the clone's ~16 cm radial precision either way. The scripted expert passes 60/60 against the
real detector, because its decisions are explicit gates rather than a fitted function. Making a
clone fly *this* ring needs more than better demonstrations: the commit decision is a
discontinuity, and regression on it averages across both sides of it.

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

**Step 5 — record demonstrations, then train a clone.** Fly several times with `--record`;
each flight saves a `demo.npz` inside its timestamped folder under `hardware/flights/`.
The path is printed at the end of each flight — pass it to `bc_train.py`.

```powershell
& $PY hardware/onboard/fly_hoop.py --record --max-travel 1.8 --attempts 5
# The flight prints: [demo] N samples -> hardware/flights/tello_YYYYMMDD-HHMMSS/demo.npz
# Collect a few flights, then train (replace the path with your actual file):
& $PY hardware/imitation/bc_train.py `
      --demos hardware/flights/tello_YYYYMMDD-HHMMSS/demo.npz `
      --out hardware/flights/bc/onboard_clone --eval-episodes 20
```

**Step 6 — fly the cloned policy.**

```powershell
& $PY hardware/onboard/fly_hoop.py --mode policy `
      --model hardware/flights/bc/onboard_clone/bc_model.zip
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
  30 Hz, 960x720            blue mask,      30 floats   expert |      clamp to RC_MAX      15 Hz
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
| `spec.py` | **The task definition**: the 30-number observation layout, the 4-number action, the normalisation, and the rc clamp. Everything else imports it, so the sim, the real flight, the recorder and the trainers cannot disagree about what a number means |
| `ring_detect.py` | Blue ring detection and tracking. Adapted from the kit's `perception.py` (credited in its docstring) with every threshold parameterised |
| `obs.py` | Builds the observation from a detection plus one telemetry packet. `ring_from_obs` is the inverse |
| `expert.py` | The scripted visual-servo controller, and `ExpertPolicy`, which gives it a learned policy's interface |
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
