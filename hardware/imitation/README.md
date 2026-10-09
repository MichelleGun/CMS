# imitation/: teach the drone policy by showing it how to fly

This folder turns **recordings of a good pilot** into a **policy** (a small neural network) that
flies the Tello through the vertical ring on its own. The pilot can be you at the keyboard or the
scripted expert in the simulator. The method is **behaviour cloning**: ordinary supervised
learning, with no reward and no trial and error in the air.

```
 pilot flies  ->  recorder saves (what the drone saw, what the pilot did)  ->  bc_train.py
      ^                                                                           |
      |                                                                           v
  you / expert                                              bc_model.zip  ->  fly_hoop.py --mode policy
```

Every command below is copy-paste. Run them from the `CMS` folder in PowerShell.
**Or skip the terminal:** the dashboard (`webapp/README.md`) does all of this. Fly and save
demonstrations in its Drone tab, and generate demos, train, evaluate and fly clones in its
Imitation tab.

## 0. Contents

| File | What it does |
|---|---|
| `record_demo.py` | The demonstration file format (`DemoRecorder`, `load_demo`). Not run directly. |
| `bc_train.py` | Trains a policy from demonstration files and (optionally) tests it in the simulator. |
| `make_sim_demos.py` | Older, for the simulator's point-reaching task. See the appendix. |
| `../onboard/make_demos.py` | Makes demonstrations for the **drone task** with the scripted expert, no hardware. |
| `../onboard/fly_hoop.py --mode human --record` | Records **you** flying the real drone. |

## 1. The idea, in drone terms

A policy is a function: **observation in, action out**, 15 times a second.

* **Observation** (33 numbers, `onboard/spec.py`). The drone does not see the image. It sees what
  the ring detector measured, plus its own telemetry:

  | Group | Numbers | Meaning |
  |---|---|---|
  | `found`, `age` | 2 | Is the ring locked right now, and how many ticks since it last was |
  | `centre` | 2 | Ring centre offset from the middle of the image (left/right, up/down) |
  | `scale`, `axis_ratio` | 2 | How big the ring looks, and how square-on it is |
  | `range` | 1 | Estimated metres to the ring |
  | `last_range`, `last_centre` | 3 | **Where the ring was when last seen.** This is the policy's memory |
  | `height`, `vel`, `att` | 7 | Tello height, velocity, pitch/roll/yaw rate |
  | `actions` | 16 | The last 4 commands that were actually sent |

* **Action** (4 numbers, each in -1..1), the same four channels `send_rc_control` takes:

  | Channel | Meaning |
  |---|---|
  | `lr` | **strafe** left/right (slide sideways, nose keeps pointing ahead) |
  | `fb` | forward/back |
  | `ud` | up/down |
  | `yaw` | **rotate** left/right (turn in place) |

Behaviour cloning collects many pairs *(observation, what the pilot did)* and trains the network
to output the pilot's action when it is shown that observation. Nothing more. It is why a
demonstration is cheap: a Tello battery gives a handful of flights, and every 1/15 s tick of every
flight is a training example.

**Why the "last seen" numbers matter.** The ring leaves the camera frame on the final push, so
`found` drops to 0 and `range` reads "far". That looks exactly like searching at the start of a
flight, which needs the opposite action (turn and look rather than push on). Without the
last-seen fields a clone flew to 17 cm of the ring and hovered there until timeout. With them it
can tell "I am mid-push" from "I have not found it yet".

## 2. The data file

One flight is one **episode**. It is saved as a compressed `.npz`:

| Array | Shape | Meaning |
|---|---|---|
| `obs` | (N, 33) | The observation at each tick |
| `intent` | (N, 4) | What the pilot asked for. **This is the training target** |
| `executed` | (N, 4) | What was actually sent after the safety layers trimmed it |
| `t`, `episode` | (N,) | Seconds since the start, and which flight |
| `meta` | JSON | Task name, observation shape, source, whether it passed |

Why `intent` and not `executed`: the pilot wanted that motion even if the geofence cut it. Use
`--use executed` if you would rather the clone learn what the drone really did.

Real flights land in `hardware/flights/tello_<timestamp>/`:
**`demo.npz`** if the run went through the ring, **`demo_failed.npz`** if it missed. The training
glob below picks up `demo.npz` only, so misses are kept but never trained on. Nothing is ever
overwritten: each flight has its own timestamped folder.

## 3. Tutorial

### Step 0. Set up

```powershell
$PY = "C:/Users/miche/miniconda3/envs/drones/python.exe"
& $PY -m pip install pygame
& $PY -m pytest hardware/onboard/tests -q
```

### Step 1. Run the whole loop with no drone

The scripted expert flies the simulator, the recorder saves it, and a clone is trained and tested.
This takes a few minutes and proves the pipeline works before anything has propellers.

```powershell
# 1a. the expert flies 60 episodes; only the ones that pass through are kept
& $PY hardware/onboard/make_demos.py --episodes 60 --out hardware/flights/demos/onboard_sim.npz

# 1b. clone it, then test the clone in the simulator on 20 fresh episodes
& $PY hardware/imitation/bc_train.py --demos hardware/flights/demos/onboard_sim.npz --out hardware/flights/bc/onboard_clone --eval-episodes 20
```

Optional `--render` puts the real OpenCV detector in the loop instead of a fast stand-in. It is
slower, and at the measured 30 cm ring it has not helped: the one clone trained that way scored 0/12.

What the output means (a 20-episode run, so the numbers are small):

```
[make_demos] 3035 samples from 20 episodes, 20/20 passed -> ...
[bc] task 'onboard_hoop', obs (33,), action (4,), loss 'mse'
[bc] epoch  199  val action-mse 0.0003  speed-mse 0.0000  mean cos 0.965
[bc] onboard sim rollout: passed 100% (10/10), struck 0, missed 0, mean reward 21.6
```

* **val action-mse**: squared error between the clone's action and the pilot's on 15 % of the data
  it never trained on. Lower is better. It only says "copies the pilot", not "flies well".
* **mean cos**: how well the clone's action points the same way as the pilot's (1.0 is perfect).
* **sim rollout**: the number that counts. The clone flies fresh episodes alone. `struck` means it
  hit the noodle, `missed` means it went past the ring.

### Step 2. Practise the keys with nothing flying

```powershell
& $PY hardware/onboard/fly_hoop.py --dry-run --mode human --record --countdown 0
```

A small window opens. **Click it so it has focus.** Fly the simulated drone, press **L** to land,
and answer **`n`** to "Save this run?" so practice never reaches your training data.

| Key | Does |
|---|---|
| **W / S** | forward / back |
| **A / D** | **strafe** left / right |
| **Up / Down arrow** | up / down |
| **Left / Right arrow** | **rotate** left / right |
| **L** or **Esc** | land |
| **X** or **Backspace** | emergency motor cut |

Ctrl+C in the terminal also lands. A held key sends 60 % of full stick (`--human-gain`, default
0.6): the policy's output is smooth, and a clone cannot reproduce a slam to full throttle.

### Step 3. Record real demonstrations

Cage closed, nobody inside, battery above 30 %, phone Tello app closed. Then:

```powershell
& $PY hardware/onboard/fly_hoop.py --mode human --record --max-travel 1.8 --max-airborne 60
```

Click the window during the 5 s countdown. Fly to the ring, through it, and press **L**. After the
drone has **landed** you are asked two questions:

```
Save this run (N samples)? [y/n]
Did it go through the ring? [y/n]
```

* First `n`: throw the run away. Second `y`/`n`: saved as `demo.npz` / `demo_failed.npz`.
* A flight that ended in an error, Ctrl+C or a motor cut is never offered for saving.
* Recording starts the first time the ring is seen and **continues through the blind push**. That
  push is the part the clone most needs to see.

Use **both** strafe and rotate: rotate to point the nose at the ring, strafe to slide onto its axis.
Rotating alone cannot fix a sideways offset, and there is only about 8 cm of clearance.

### Step 4. How many runs, and from where

* Aim for **30 to 50 passed flights**. A few dozen is enough for a first clone; more varied is
  better than more of the same.
* **Vary the start between runs**, not during one: distance from the ring, sideways offset, height,
  and which way the nose points (include some that start facing away). The clone only knows the
  situations it was shown, so a spread of starts is what lets it work from more than one spot.
* Keep each run clean. Clean, decisive passes teach better than ones where you hesitated. Answer
  `n` to a messy run rather than saving it.
* **Camera:** the drone's own camera is already the policy input (the ring detector reads it and
  its output is in `obs`). You do not need to record video for training. A phone filming from
  outside is useful only for reviewing your own flying.

### Step 5. Train the clone

```powershell
& $PY hardware/imitation/bc_train.py --demos hardware/flights/tello_*/demo.npz --out hardware/flights/bc/human_clone --eval-episodes 20
```

Mix in simulated demonstrations by listing more files:

```powershell
& $PY hardware/imitation/bc_train.py --demos hardware/flights/tello_*/demo.npz hardware/flights/demos/onboard_sim.npz --out hardware/flights/bc/mixed_clone --eval-episodes 20
```

| Flag | Default | Meaning |
|---|---|---|
| `--demos` | required | One or more `.npz` files or globs. All must be the same task |
| `--out` | required | Folder that gets `bc_model.zip` and `meta.json` |
| `--epochs` | 200 | Passes over the data |
| `--lr` | 0.003 | Learning rate |
| `--use` | `intent` | Train on `intent` or `executed` |
| `--eval-episodes` | 0 | Fly this many simulator episodes after training |
| `--seed` | 0 | Reproducibility |

The loss is plain mean-squared error on the four channels. `--warm-critic` and `--as-run` only
apply to the simulator's older task; for the drone task `--warm-critic` stops with an error and
`--as-run` is skipped.

### Step 6. Fly the clone

```powershell
& $PY hardware/onboard/fly_hoop.py --mode policy --model hardware/flights/bc/human_clone/bc_model.zip --max-travel 1.8
```

The clone goes through the same safety layers as everything else: geofence, height limits,
`--max-travel`, `--max-airborne`, Ctrl+C. Keep `--max-travel 1.8`: it is the total forward command
over the whole flight including the blind push, and a smaller value quietly cuts the push short.
The loader refuses a model with the wrong observation or action shape.

### Step 7. Improve it

1. Look at the flights where it failed and record more demonstrations that start from those
   situations. Failures are usually a start the data never covered.
2. Keep the passes and leave `demo_failed.npz` out, so the clone copies only successful flying.
3. Mix simulated and real demonstrations (Step 5) if real data is scarce.
4. If a clone passes in the simulator but not on the drone, trust the drone: re-run the
   speed/axis calibrations in `onboard/README.md` and record more real flights.
5. Reinforcement learning from the clone's weights is the next step: `onboard/gym_env.py`.

## 4. How it applies to the drone policy

Take one tick of a recorded flight. The ring is 0.8 m ahead, 0.1 of a half-frame to the left, and
the pilot is holding W and tapping Left:

```
obs    = [found=1, centre=(-0.1, 0.0), scale, axis_ratio, range=0.8/2, ... last 4 actions]
intent = [lr=0, fb=+0.6, ud=0, yaw=-0.6]
```

`bc_train.py` shows the network `obs` and nudges its weights until it outputs `intent`. Do that for
thousands of ticks and the network learns the rule "ring left of centre and far, so move forward
and rotate left" without anyone having written it down.

Three consequences:

* **The clone copies style, mistakes included.** If your demonstrations hesitate, so will it.
* **The clone has never seen states you did not visit.** Small errors push it into situations the
  pilot avoided, where it has no example to follow. This is the main reason imitation learning
  needs varied starts.
* **A sharp jump in the expert's behaviour is hard to copy.** A single-tick jump from 0.28 to 1.0
  forward throttle appeared about 40 times in 6000 samples while its continuation appeared 45
  times, so the clone learned to continue a push but never to start one. The scripted expert now
  ramps smoothly into the push, and a human flying smoothly helps for the same reason.

## 5. Honest limits

* **Nothing here has been verified on a real flight yet.** Simulator passes use a stand-in for the
  camera and a model of the drone.
* Re-scored on 12 simulator episodes at the measured 30 cm ring (so one or two episodes is noise):

  | Clone | Analytic sim | Real OpenCV detector |
  |---|---|---|
  | trained on analytic-model expert demos | 11/12 | 6/12 |
  | trained on `--render` expert demos | 0/12 | 0/12 |

  The scripted expert itself passes 60/60 against the real detector, so the clone is clearly
  weaker than the expert once detector noise is in the loop. Why the `--render` clone fails has
  not been investigated.
* Human keyboard demonstrations are noisier and far fewer than the scripted expert's, so expect a
  first clone to be a baseline and the recordings to be the real prize: a dataset from the real
  drone that later methods can reuse.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no demo files match` | The glob found nothing. Check `hardware/flights/tello_*/demo.npz` exists, and that you answered `y` to both prompts |
| `demo files are from different tasks` | You mixed simulator-task and drone-task files. Train them separately |
| `--mode human needs pygame` | `& $PY -m pip install pygame` |
| Keys do nothing | The small window lost focus. Click it |
| Model refuses to load | It was trained for a different task (wrong observation/action shape). Retrain on drone-task demos |
| Clone hovers short of the ring | It cannot tell "mid-push" from "searching". Record demonstrations that include the full push |
| Clone passes in the simulator but misses on the drone | Normal. Record more real demonstrations from varied starts |

## Appendix: the simulator's point-reaching task

`make_sim_demos.py` and the `sim_vel` loss belong to the older `HoopFindAviary` task, where the
observation is 75 numbers and the action is *a direction plus a speed fraction*. The loss there is
a cosine loss on direction plus a speed error, because the simulator's controller ignores the
direction's length. `bc_train.py` picks the right loss from the demo file's `meta["task"]`, so the
same command trains either kind. `--warm-critic` and `--as-run` are for that task only.

```powershell
& $PY hardware/imitation/make_sim_demos.py --expert scripted --episodes 60 --out hardware/flights/demos/sim_demo.npz
& $PY hardware/imitation/bc_train.py --demos hardware/flights/demos/sim_demo.npz --out hardware/flights/bc/first_clone --warm-critic --eval-episodes 20
```
