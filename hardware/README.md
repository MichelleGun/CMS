# hardware/: flying the drone policies on a real DJI Tello

This README has three parts.

1. [Purpose of the folder](#part-1-purpose-of-the-folder): what this is for, in one screen.
2. [How to use it](#part-2-how-to-use-it): a step by step walkthrough that assumes you have never done this before.
3. [Full explanation](#part-3-full-explanation): what is inside, why it is built this way, and how it was made.

---

# Part 1. Purpose of the folder

The rest of this project trains drone policies in a simulator (`scripts/`, the `HoopFindAviary` task: fly a drone
to a hoop placed anywhere in a 2x2x2 m cage). A policy that is good in a simulator is only interesting if it also
works on a real drone. This folder is the bridge to a real **DJI Tello** in a real 2x2x2 m cage, so that the question
"does what we trained transfer?" can be answered with an experiment instead of a guess.

With it you can:

* **Fly the Tello with the keyboard**, with an emergency stop and safe landing always available.
* **Run a trained policy on the Tello**: a policy trained in the sim (any run from the dashboard or `results/hoop/`)
  or one you cloned from your own flying.
* **Record your own flying** (what the drone sees and what you did) and **clone it** into a policy by imitation
  learning ("behaviour cloning"), in the same file format the simulator tools already load.
* **Measure the gap** between the real flight and the simulator, so you know what to fix.
* **Rehearse everything with no drone at all**, using a simulated Tello (`--dry-run`), so you can learn the tools,
  and catch software mistakes, before anything with spinning propellers is involved.

It also contains the machinery that makes this possible: an overhead-camera position tracker (the Tello cannot
tell you where it is in the room), a filter that blends that camera with the Tello's own sensors, the safety layer,
the calibration scripts, and an offline test suite.

**Emergency keys (they always work, in every script):**

| key | what it does |
|---|---|
| **X** or **BACKSPACE** | **EMERGENCY: cut the motors.** The drone drops. Use it only if landing is not an option |
| **L** | safe land |
| **SPACE** | hold: hover in place (press again to release) |
| ESC or Ctrl+C | safe land, then quit |
| Ctrl+Shift+X / L / H | the same three, **from any window** (needs the optional `pynput` package) |

**Status, stated plainly.** Everything here is built and tested **without a drone** (99 tests, against a simulated
Tello and the real simulator). **Nothing has flown on a real Tello yet.** A few Tello facts (which direction each
control moves, the unit of its speed readings) are assumptions until you run the calibration scripts on your own
drone, and real takeoff is refused until you have. Part 2 walks you through that safely.

**What this folder is not:** it does not train new policies (that is still `scripts/` and the dashboard), and it does
not change the simulator.

---

# Part 2. How to use it

Read this part top to bottom the first time. It has three stages:

* **Stage A (steps 1 to 6):** no drone, no camera needed. Safe. Do this first.
* **Stage B (steps 7 to 11):** set up the real room: markers, camera, Wi-Fi.
* **Stage C (steps 12 to 20):** first real flights, one careful step at a time.

Every step says **what it does and why**, gives **the command**, tells you **what you should see**, and tells
you **what to do if you do not**.

## Words you will meet

| word | meaning |
|---|---|
| **cage** | the 2x2x2 m netted space the drone flies in |
| **cage frame** | the coordinate system the code uses: origin at the cage floor centre, x and y in metres across the floor, z up. x runs between two of the cage walls, y between the other two (you define which by where you put the corner markers in step 7) |
| **policy** | the trained neural network that decides how the drone should move |
| **hoop** | the target the drone has to fly to (only a point in space for the policy: "get within 15 cm of it") |
| **dry run** (`--dry-run`) | using a *simulated* Tello instead of a real one. Nothing spins. Nothing is connected |
| **ArUco marker** | a printed black and white square that a camera can find and measure. We use them to find the drone and the floor corners |
| **calibration** | measuring a fact about your particular drone/camera instead of assuming it |
| **demonstration** | a recording of you flying, used to teach a policy by imitation |
| **rc** | the Tello's remote-control command: four numbers (left/right, forward/back, up/down, turn) from -100 to 100 |
| **estimator** | the code that works out where the drone is by combining the camera and the Tello's sensors |

## Step 0. What you need

**Hardware (for Stages B and C):**

* a DJI **Tello** (regular or EDU both work), **at least two charged batteries**, and **propeller guards**
* a **2x2x2 m cage** with nets on all sides and the top, with **nobody inside while it flies**
* a **USB webcam** mounted **looking straight down** over the cage centre, about **3.5 m up** (see step 8)
* a **printer** and paper (matt, not glossy), tape, scissors
* the PC must be able to join the Tello's own Wi-Fi network (so it will lose internet while flying)

**Software:** Windows, and the project's `drones` conda environment (Python 3.12).

**Set up a shortcut for Python once.** On this machine `conda activate drones` does not work (the environment lives
under `miniconda3`, and the shell's `conda` is a different install), so every command below calls that Python
directly. In PowerShell, from the project folder (`CMS`):

```powershell
cd <path to>\CMS
$PY = "C:/Users/miche/miniconda3/envs/drones/python.exe"     # use YOUR path to the drones environment's python.exe
```

Everywhere below, `& $PY ...` means "run this with the `drones` Python". Always run commands **from the `CMS`
folder**.

---

## Stage A: no drone needed

### Step 1. Install the extra packages

*What and why:* the simulator side is already installed. Talking to the real Tello, the camera and the keyboard
window needs four more packages: `djitellopy` (talks to the Tello), `opencv-contrib-python` (finds the markers),
`pygame` (the keyboard window), `pynput` (emergency keys that work from any window).

```powershell
& $PY -m pip install -r hardware/requirements-hardware.txt
```

*You should see:* pip downloading and ending with `Successfully installed ...`.
*If it does not:* if pip complains about two OpenCV packages, run `& $PY -m pip uninstall -y opencv-python` and
then repeat; keep `opencv-contrib-python`. If `pynput` fails to install you can skip it: everything still works, but
the emergency keys then only work while the keyboard window has focus.

### Step 2. Run the self-tests

*What and why:* the folder comes with 99 automated checks. They need no drone. Running them proves your install works
and that nothing was broken by a later change. Do this whenever you edit anything.

```powershell
& $PY -m pytest hardware/tests -q
```

*You should see:* a row of dots and `99 passed` after about a minute. (If you skipped step 1, you will see `91 passed,
1 skipped`: the skipped file is the one that needs OpenCV and pygame. That is fine.)
*If it does not:* a failing test names the file; read the message. The most common cause is running from the wrong
folder: `cd` to `CMS` first.

### Step 3. Check that the policy's inputs are identical to the simulator's

*What and why:* the policy was trained on a particular list of 75 numbers. If the hardware code built that list
even slightly differently, the policy would behave strangely and it would be very hard to see why. This check runs
the **real simulator** and the hardware code on the same state and compares the lists.

```powershell
& $PY hardware/sim_check/obs_parity.py
```

*You should see* (the last lines):

```
[obs_parity] 60 steps, obs shape (1, 75), max |sim - ours| = 0.000e+00
[obs_parity] OK
```

*If it does not:* `max |sim - ours|` above `1e-05` means the two have drifted apart (someone changed
`scripts/hoop_env.py` or `hardware/common/constants.py`). Fix before flying anything.

### Step 4. Fly a pretend Tello with your keyboard

*What and why:* the best way to learn the controls and the safety keys is on a drone that cannot hurt anything.
`--dry-run` replaces the Tello and the camera with simulated ones.

```powershell
& $PY hardware/control/keyboard_teleop.py --dry-run
```

A small window opens. **Click on it** so it has keyboard focus. Then:

1. Wait two seconds, press **T** (takeoff). If the window prints `[takeoff refused] ... not ready`, wait a second
   and press **T** again: the position estimate needs a moment of data first.
2. The drone rises. After about 3 seconds the title line says `FLYING`.
3. Fly with the **arrow keys** (left/right = x, up/down = y) and **W / S** (up / down). Keys **1 / 2 / 3** set the
   speed to 40 / 70 / 100 %. Yaw (turning) is held automatically, so there are no turn keys.
4. Practise every safety key: **SPACE** (hover, press again to release), **L** (lands by itself), and, in a new
   run, **X** (motors cut; the status turns red and says `KILLED`).

*You should see* a status panel like this (real output):

```
FLYING  MANUAL
pos  x+0.16 y+0.03 z+0.95 m   +-0.01   yaw +0 deg
vel  +0.19 +0.01 -0.00 m/s    rc (0, 22, 0, 0)
health pos=OK yaw=OK  cam_age=0.00s tele_age=0.10s
battery 84.0 %   temp 55.0 C   speed 70 %
recording: ON  samples 120
```

(This sample was taken with recording on; without `--record` the last line says `recording: off  samples 0`.)

How to read it: `pos` is where the estimator thinks the drone is (metres in the cage frame) and `+-0.01` how sure it
is; `rc` is what is being sent to the Tello; `health pos=OK` means the camera and sensors agree (`DEGRADED` = the
camera has been missing for half a second and the drone is running on its own sensors; `LOST` = the position
cannot be trusted and the system will hold, then land); `cam_age`/`tele_age` are seconds since the last camera
frame / Tello packet.

Try to push the drone into a wall: it slows down and stops short of it. That is the **geofence** doing its job.

### Step 5. Run a trained policy on the pretend Tello

*What and why:* this runs the actual sim-trained policy through the entire hardware path (estimator, observation
builder, safety, action mapping). You give it a hoop position; it flies there and lands by itself.

```powershell
& $PY hardware/control/run_policy.py --model results/hoop/best_model.zip --dry-run --headless --hoop 0.4 -0.3 1.0 --speed-scale 1.0
```

`--headless` means "no window, take off by yourself". The three numbers are the hoop's x, y, z in metres (x and y
must be within +-0.75, z between 0.3 and 1.6, the range the policy trained on). `--profile ideal|typical|harsh`
changes how rough the pretend drone is.

*You should see* (real output, default `typical` profile):

```
[policy] PPO results\hoop\best_model.zip
[dry-run] FakeTello profile 'typical', no hardware is touched
[hoop] target: x=+0.40 y=-0.30 z=+1.00 m
[policy] finished: phase=landed reached_hoop=True ticks=417
```

`reached_hoop=True` and `phase=landed` is success. (`ticks` is how many 1/30-second steps it took: about 14 s. Try `--profile ideal` and `--profile harsh` as well; the policy also reaches the hoop and lands on both.)
*If it does not:* `--hoop ... is outside` means the hoop is somewhere the policy never saw (use `--force` only
if you understand why); `expects observations of shape ...` means the model was not trained on `HoopFindAviary`.

### Step 6. Try the whole imitation-learning loop with no hardware

*What and why:* before you record yourself, prove the pipeline works by letting the **simulator** play the expert.
This also produces a useful starting policy.

```powershell
# 6a. generate 60 demonstration flights in the simulator (about a minute)
& $PY hardware/imitation/make_sim_demos.py --expert scripted --episodes 60 --out hardware/flights/demos/sim_demo.npz

# 6b. teach a policy to copy them ("behaviour cloning"), and test it in the simulator
& $PY hardware/imitation/bc_train.py --demos hardware/flights/demos/sim_demo.npz --out hardware/flights/bc/first_clone --warm-critic --eval-episodes 20

# 6c. fly the clone on the pretend Tello
& $PY hardware/control/run_policy.py --model hardware/flights/bc/first_clone/bc_model.zip --dry-run --headless --hoop 0.4 -0.3 1.0 --speed-scale 1.0
```

*You should see* (real output from one run):

```
[make_sim_demos] 7433 samples, 60/60 episodes reached the hoop -> ...sim_demo.npz
[bc] epoch   0  val dir-loss 0.0695  speed-mse 0.0243  mean cos 0.931
[bc] epoch 149  val dir-loss 0.0002  speed-mse 0.0000  mean cos 1.000
[bc] critic warmed up, final loss 0.0010
[bc] sim rollout: success 100%, mean final distance 0.15 m
[policy] finished: phase=landed reached_hoop=True ticks=1477
```

How to read it: `mean cos` is how closely the clone's chosen direction matches the expert's (1.000 = identical);
`sim rollout: success` is the clone actually flying 30 fresh random episodes in the simulator. The clone reached the hoop on
the pretend Tello too, though it took about three times as long as the 500k-step PPO (1477 ticks vs about 420); I did not investigate why.

To record **yourself** instead of the simulator, use the keyboard (you need the window):
`& $PY hardware/control/keyboard_teleop.py --dry-run --record --hoop 0.4 -0.3 1.0`. Fly to the hoop a dozen
times (T to take off, fly, L to land, run again). Demonstrations are saved to `hardware/flights/demos/`. Recording
starts at your first key press after takeoff and stops when you land.

---

## Stage B: set up the real room

### Step 7. Print the markers and place them

*What and why:* the Tello cannot tell you where it is in the room. The overhead camera can, if it can see markers.
Four markers on the floor define the cage coordinates; one on the drone says where the drone is; one under the hoop says where the hoop is.

```powershell
& $PY hardware/calibration/gen_markers.py
```

This writes PNGs to `hardware/calibration/markers/`. Print them at **100 % scale** (no "fit to page"), on matt paper.

| marker id | where it goes | size on paper |
|---|---|---|
| **0, 1, 2, 3** | flat on the floor, centred on the cage corners at the cage-frame points (-1,-1), (+1,-1), (+1,+1), (-1,+1) metres | about 15 to 20 cm |
| **10** | **on top of the Tello**, flat, with the marker's **top edge toward the drone's nose** (the front camera side) | about 7 to 10 cm |
| **20** | flat on the floor **directly under the hoop's centre** | about 15 to 18 cm |

Decide which way is "+x": marker 0 is at (-1,-1) and marker 1 at (+1,-1), so **+x runs from marker 0 toward marker 1**
and **+y from marker 0 toward marker 3**. The keyboard's arrow keys use these directions. **Place the drone on the floor
at takeoff with its nose toward +x** (toward marker 1); if it is not, it turns to face +x by itself right after taking off.

*You should see:* six PNG files named `00_reference_0.png` ... `20_hoop_floor.png`.

### Step 8. Mount and aim the overhead camera

*What and why:* the camera must see all four corner markers **and** the drone anywhere in the cage. A typical webcam
(about 70 degree field of view) needs to be **at least about 3.5 m above the floor** to see a 2x2 m floor with some
margin on its shorter side; at 3 m the corner markers sit on the very edge of the picture.

1. Fix the camera pointing **straight down**, over the cage centre.
2. Measure the height from the camera lens to the cage floor, in metres.
3. Tell the software. Create `hardware/config.local.yaml` (this file is yours; it is not committed to git) with:

```yaml
camera:
  height_m: 3.6        # your measured height
```

*You should see:* nothing yet. You check in step 11. *If the corners are cut off* in step 11: raise the camera, or
move the corner markers inward and edit `aruco.reference` in `hardware/config.yaml` to their new positions.

### Step 9. Calibrate the camera lens (recommended)

*What and why:* webcam lenses bend straight lines near the edges; the corner markers are exactly there. This
measures the bending once so the software can undo it. You need a printed **chessboard** (9x6 inner corners, any
size; 25 mm squares is the default), stuck on something flat.

```powershell
& $PY hardware/calibration/calibrate_camera.py --cols 9 --rows 6 --square-mm 25
```

Hold the board in front of the camera, tilting and moving it so it appears in every part of the picture (especially
the corners). Press **SPACE** each time the board is highlighted (at least 10, aim for 15 or more), then **C** to compute, **Q** to quit.

*You should see:* `saved ...camera_intrinsics.json   RMS reprojection error 0.3 px (below ~0.5 px is good)`.
*If it does not:* "RMS" above about 1 px means too few or too similar pictures; redo with more varied positions.

### Step 10. Power up and connect to the Tello

*What and why:* the Tello makes its own Wi-Fi network. Your PC must join it, so **the PC will have no internet** while
you fly (the webcam is USB, so it is unaffected).

1. Fit the **propeller guards**. Insert a **charged battery**. Place the drone on the cage floor, nose toward +x.
2. Press the power button. Wait until the light stops flashing different colours and blinks a single colour.
3. On the PC, join the Wi-Fi network called `TELLO-xxxxxx`.

*If the PC will not connect:* wait 20 seconds and look again; if the drone's light is red, change the battery.

### Step 11. Look before you fly

*What and why:* before any flight, check that the camera sees what you think it sees.

```powershell
& $PY hardware/calibration/validate_hoop.py
```

*You should see* a live camera window with a **red outline** joining the four floor corners and, when the drone
(with its marker) is in view, a **blue circle** on the marker. If you have already located a hoop (step 16), a
**green circle** shows it. Press **Q** to quit.
*If the red outline is missing or wrong:* a corner marker is not visible or is at the wrong place; fix it before going on.

---

## Stage C: first real flights

> **Safety rules for every flight in this stage:** prop guards on; cage closed; nobody inside; battery above 30 %
> (the software refuses to take off below it); **one person with a hand on the X key** (or Ctrl+Shift+X from any
> window) the whole time. If anything looks wrong, press **L**. If the drone is out of control, press **X**.

### Step 12. Props off, drone in your hand: see what the software thinks

*What and why:* run the real keyboard program with the real drone connected, but **without flying it**. Hold the drone
(propellers will not spin because we will not get past takeoff).

```powershell
& $PY hardware/control/keyboard_teleop.py
```

*You should see* a warning at start-up, then the status panel, then, if you press **T**:

```
[warning] conventions.axes_calibrated is false: takeoff will be refused. Run calibration/calibrate_axes.py first ...
[takeoff refused] axes not calibrated: run calibration/calibrate_axes.py (sets conventions.axes_calibrated)
```

(Do this step **before** step 13. Once the axes are calibrated, **T** really takes off.)
That refusal is **correct and expected**. In the panel, `battery` should show a number and `health pos=OK` once
the drone's marker is visible to the camera. If the panel never shows `cam_age` below one second, the drone marker is
not being seen (check its orientation and that it is in view). Press ESC to quit.

### Step 13. Calibrate the axes (how your Tello's controls really behave)

*What and why:* the code has to translate "move +x" into the Tello's four rc numbers. Which number does what, and
with which sign, is written down in the Tello's documentation in a way that is easy to misread, so this script
**measures it instead**. The drone takes off and makes eight small hops (forward and back, left and right, up and
down, turn both ways) using raw commands; the overhead camera and the Tello's height reading say what each one
really did. This is the only step that flies with an unverified assumption, so the hops are small (20 to 30 cm).

First rehearse (nothing flies, nothing is written): `& $PY hardware/calibration/calibrate_axes.py --dry-run`. Then, with the cage clear and your hand on the X key:

```powershell
& $PY hardware/calibration/calibrate_axes.py
```

Type `fly` when asked. The drone takes off, makes the hops, and lands.

*You should see* a report. If your drone matches the assumptions:

```
[axes] results
  conventions -> {'vel_to_mps': 0.1, 'rc_signs': {'lr': -1, 'fb': 1, 'ud': 1, 'yaw': -1}, 'vel_signs': [1, -1, -1], ...}
Write these to config.local.yaml? [y/N]
```

Answer `y`. If something differed you will instead see lines such as `lr: positive rc moved the drone LEFT -> flipping
rc_signs.lr`; the corrected values are what gets written. Lines starting `! UNRESOLVED:` mean the script could not
establish something (for example the drone barely moved): it then refuses to mark the axes as calibrated, and you
fix the cause and run it again. (With a deliberately mirrored pretend drone, `--dry-run --fake-axis-flip` shows the
mismatch being found.)

### Step 14. Calibrate the speed

*What and why:* the simulator's "full speed" means exactly 0.25 m/s. The Tello's rc number that gives 0.25 m/s is
not known in advance (the tutorial this project started from measured about 12 cm/s for rc 15, quite different from
what the numbers suggest). This flies three short passes forward and back at different rc values, measures the real
speed with the camera, and fits the conversion.

```powershell
& $PY hardware/calibration/calibrate_speed.py
```

*You should see* (real output from the pretend drone, true value 125):

```
[speed] rc  15 -> 0.109 m/s, 63 % in 0.30 s
[speed] rc  25 -> 0.181 m/s, 63 % in 0.33 s
[speed] rc  35 -> 0.310 m/s, 63 % in 0.50 s
[speed] rc_per_mps = 120.8   (config currently 128.0)
```

Answer `y` to write it. Read the line `the sim's full speed needs more rc than rc_max allows`, if it appears: it means
the drone is too slow for the policy's top speed under the current safety clamp. You can raise `control.rc_max`
deliberately, or just keep `--speed-scale` below 1.

### Step 15. First keyboard flight

*What and why:* now that the controls are known, fly by hand, gently, to learn how this drone feels in this cage.

```powershell
& $PY hardware/control/keyboard_teleop.py
```

Press **1** (40 % speed), then **T**. Wait for `FLYING`. Move a little in each direction. Land with **L**. Take off
again to prove a second takeoff works. Practise **SPACE** (hover) and **L** (land) in the air.

*You should see* the drone hover steadily about 0.8 m up after takeoff, move where the arrow keys say, slow down
when it nears a wall, and land softly. *If it moves the wrong way for a key:* land and re-run step 13.

### Step 16. Tell the system where the hoop is

*What and why:* the policy needs to know the hoop's position (it is part of its input). Measuring it with the
camera beats typing coordinates by hand.

```powershell
& $PY hardware/calibration/hoop_locator.py            # the camera finds marker 20 on the floor under the hoop
& $PY hardware/calibration/validate_hoop.py           # check the green circle is on the hoop's floor spot
```

If you set the hoop's height other than 1.0 m, say so: add `--z 1.2` (the overhead camera cannot measure height, so
height always comes from you; or set `hoop: {z: 1.2}` in `hardware/config.local.yaml`). To skip the camera:
`hoop_locator.py --manual 0.4 -0.3 1.0`.

*You should see:* `[hoop] x=+0.400 y=-0.300 z=+1.00 m (std 0.4 cm, 30 frames)` and `[hoop] wrote ...hoop.json`. The reading is
refused if it is noisy, or if the hoop is outside the range the policy trained on. The hoop file expires after 10 minutes
so a stale position cannot be used by accident.

### Step 17. Record a demonstration

*What and why:* this is the imitation-learning data: what the drone saw, and what you asked it to do.

```powershell
& $PY hardware/control/keyboard_teleop.py --record
```

It reads the hoop from step 16 and asks you to confirm. Take off (**T**), fly to the hoop (stop within about 15 cm of it), land
(**L**). Recording starts at your first key press and ends at landing; the file goes to `hardware/flights/demos/`.
Do this 10 to 20 times, with the hoop in different places (repeat step 16 each time you move it). Fly smoothly
and directly at the hoop: the clone copies your style, including your mistakes.

### Step 18. Train a policy from your demonstrations

```powershell
& $PY hardware/imitation/bc_train.py --demos hardware/flights/demos/demo_*.npz --out hardware/flights/bc/my_clone --warm-critic --eval-episodes 20 --as-run
```

*You should see* the same kind of lines as step 6. `--as-run` also saves it as `results/hoop/runs/bc_<time>/`, so the
dashboard's replay tools can open it. If the line `sim rollout: success` is low, you need more or cleaner demonstrations.

### Step 19. Run a policy on the real drone, slowly

*What and why:* the policy is held back at first by `--speed-scale` (a multiplier on its commanded speed). Raise it only after a flight looks good.

```powershell
& $PY hardware/control/run_policy.py --run-dir results/hoop/runs/<your run folder> --speed-scale 0.5
```

(or `--model hardware/flights/bc/my_clone/bc_model.zip`, or the 500k-step PPO `--model results/hoop/best_model.zip`.)
It prints the hoop from step 16 and asks you to confirm. Press **T**; after takeoff the drone hovers 1.5 s, then
the policy takes over, speed ramping up over one second. When the drone is within 15 cm of the hoop, it hovers for a
second and lands by itself.

* **You can take over at any moment:** press any movement key and you fly it by hand for as long as you hold it.
  Add `--record` and your corrections are saved as new demonstrations.
* Do three flights at `--speed-scale 0.5`, then three at `0.75`, then `1.0`.
* If it misbehaves: **L** to land, **X** if it does not respond.

*What to expect on the first real flights:* the simulator always starts the drone at 10 cm above the floor, but a
Tello takeoff settles about 0.8 m up, so the first moments are outside what the policy practised. Some wobble is
normal; see "Known gaps" in Part 3 for what to do about it.

### Step 20. After a flight: how did it compare, and can the estimate be tuned?

Every flight writes a log to `hardware/flights/`. Two tools read it:

```powershell
# replay the actions the real drone executed in the SIMULATOR and plot both paths
& $PY hardware/sim_check/sim_vs_real.py --log hardware/flights/policy_<time>.jsonl

# re-run the position filter with other settings and see which fits the camera best
& $PY hardware/state_estimation/tune_replay.py --log hardware/flights/policy_<time>.jsonl
```

`sim_vs_real.py` prints how far apart the sim and the real flight drift after 1, 3 and 5 seconds and saves a picture next to
the log: that gap is the sim-to-real mismatch. `tune_replay.py` is explained in `state_estimation/TUNING.md`.

---

## Key cheat sheet

| key | action | | key | action |
|---|---|---|---|---|
| **T** | take off | | **arrow keys** | move in x / y (left/right = x-/x+, up/down = y+/y-) |
| **L** | safe land | | **W / S** | up / down |
| **X**, **BACKSPACE** | **motors off** | | **1 / 2 / 3** | speed 40 / 70 / 100 % |
| **SPACE** | hold / release | | **R** | start / stop recording |
| **ESC** | land and quit | | **Ctrl+Shift+X / L / H** | kill / land / hold from any window |

## Troubleshooting

| what you see | what it means and what to do |
|---|---|
| `takeoff refused: axes not calibrated` | run `calibrate_axes.py` (step 13) |
| `takeoff refused: state estimate not ready` | no camera frames or no Tello packets yet. Is the drone marker visible, top edge toward the nose? Is the PC on `TELLO-xxxx`? |
| `takeoff refused: need 5 battery samples` | wait a second for telemetry |
| `takeoff refused: battery below 30 %` | change the battery |
| `takeoff refused: ... a LAND/KILL fault is latched` | something already stopped this session; restart the program |
| status shows `HOLD  axis_mismatch` | the drone moved opposite to what was commanded: re-run step 13 |
| status shows `policy_fights_fence` | the policy keeps trying to leave the cage; the hoop is too near a wall |
| drifts slowly toward a wall in hover | normal slow drift; the fence pushes it back inside. Persistent drift: check `health` and the camera |
| position jumps around | marker confusion or glare: see `state_estimation/TUNING.md` |
| `djitellopy is not installed` / `No module named cv2` / `pygame` | step 1 |
| `cannot open camera` | wrong camera index: `--camera-index 1`, or set `camera.index` in `config.local.yaml` |
| `no hoop position` | pass `--hoop X Y Z` or do step 16 |

---

# Part 3. Full explanation

## 3.1 How it fits together

The simulator trains a policy that maps a list of 75 numbers to 4 numbers, 30 times per second. The hardware
folder's job is to produce the same 75 numbers from the real world and to turn the 4 numbers back into Tello commands.

```
                    +---------------------- one control tick, 30 times per second (common/flight_session.py) ----------------------+
                    |                                                                                                              |
 Tello telemetry --> time_sync (clean timestamps, drop bad packets) --+                                                            |
 (10 Hz, via        |                                               v                                                              |
  djitellopy)       |                                      state_estimation: Kalman filter + yaw estimator + latency tracker        |
 Overhead camera -->|  aruco_tracker (marker -> cage x,y,yaw)  ------+  -> StateEstimate [x y z vx vy vz yaw pitch roll]            |
 (30 Hz, ArUco)     |                                                v                                                             |
                    |   obs_builder: 75 numbers, same as the simulator  --> policy (SB3 model)   or   keyboard action                |
                    |                                                v                                                             |
                    |   geofence (trim the action near walls) --> safety monitor (hold / land / kill) --> action_mapper --> rc      |
                    |                                                v                                                             |
                    |                                           Tello (rc 30 Hz, takeoff, land, emergency)                         |
                    |   every tick is logged (flights/*.jsonl): inputs, state, observation, action, rc, safety events              |
                    +--------------------------------------------------------------------------------------------------------------+
```

Everything a script can do to the drone goes through `FlightSession`. Keyboard flying, running a policy, and the
calibration scripts are thin front ends around it, so none of them can skip the geofence, the safety monitor, the takeoff
checks or the logging.

## 3.2 What is inside

```
hardware/
  README.md                this file
  config.yaml              every setting, with comments; calibration writes config.local.yaml beside it
  requirements-hardware.txt  the extra packages (step 1)
  conftest.py              makes pytest find the code
  common/                  the core: constants, observation, action, safety, the session, the simulated Tello
  state_estimation/        where is the drone? (camera, filter, timestamps, latency, tuning)
  control/                 the programs you run: keyboard_teleop, run_policy
  imitation/               record demonstrations, behaviour cloning, simulator demonstrations
  calibration/             measure your drone and camera; locate the hoop; make markers
  sim_check/               compare with the simulator
  tests/                   99 automated checks, no drone needed
  flights/                 logs and demonstrations (git-ignored), hoop.json
```

**`common/`**

| file | what it is |
|---|---|
| `constants.py` | the simulator's numbers copied here (cage size, 30 Hz, 15-action buffer, 75-number observation, 0.25 m/s). Plain numbers on purpose, so the hardware code never needs `pybullet`. `sim_check/obs_parity.py` asserts they still equal the simulator's |
| `obs_builder.py` | builds the 75-number observation from the estimated state: position, roll/pitch/yaw, velocity, angular velocity, the last 15 executed actions (oldest first, zeros at takeoff), and hoop minus position |
| `action_mapper.py` | the simulator's action (a direction plus a speed fraction of 0.25 m/s) to Tello rc: rotate from cage to body frame by the current yaw, divide by `rc_per_mps`, clamp to `rc_max`; plus the damped yaw-hold loop (gain 0.6, damping 0.15, 3 degree dead-band, clamp 20) |
| `geofence.py` | trims actions near the walls (see 3.5) |
| `safety.py` | the safety monitor, the landing sequence, the pre-takeoff checks, the global hotkeys |
| `flight_session.py` | the 30 Hz tick tying everything together (phases: ground, takeoff, warm-up, flying, landing, landed, or killed) |
| `fake_tello.py` | the simulated Tello and simulated overhead camera used by `--dry-run` and by the tests |
| `tello_link.py` | the real Tello (through `djitellopy`): connect, a thread that samples telemetry, rc at 30 Hz, takeoff/land on worker threads so the safety loop never blocks, emergency sent fire-and-forget three times |
| `smoothing.py` | median, EMA, running-min battery filter, hold timers (used by safety and estimator) |
| `logger.py`, `config.py` | per-flight JSONL log; config loading (`config.yaml`, then `config.local.yaml`) |

**`state_estimation/`**

| file | what it is |
|---|---|
| `aruco_tracker.py` | camera geometry and the camera thread (see 3.5) |
| `fusion_kf.py` | the Kalman filter and the yaw estimator |
| `time_sync.py` | cleans incoming Tello packets and assigns trustworthy timestamps |
| `latency.py` | online estimate of the camera's delay |
| `estimator.py` | combines the above and outputs the state; also the axis-mismatch monitor |
| `est_types.py` | the data types (`CameraDetection`, `StateEstimate`, health states) |
| `tune_replay.py`, `TUNING.md` | tune the filter's noise settings from a logged flight, and how |

**`control/`**: `keyboard_teleop.py` and `run_policy.py` (the two programs), `loop.py` (the shared loop: keys, recording, status panel), `inputs.py` (keyboard window, and scripted input for tests), `session_setup.py` (shared command-line options and building the real or pretend drone).

**`imitation/`**: `record_demo.py` (demonstration files), `bc_train.py` (behaviour cloning), `make_sim_demos.py` (simulator as expert).

**`calibration/`**: `calibrate_camera.py`, `calibrate_axes.py`, `calibrate_speed.py` (the two flying ones share `calib_flight.py` for the supervised flight and `analysis.py` for turning recordings into results), `gen_markers.py`, `hoop_locator.py`, `validate_hoop.py`.

**`sim_check/`**: `obs_parity.py` (is our observation identical to the simulator's?), `sim_vs_real.py` (replay a real flight's actions in the sim).

## 3.3 The simulator contract

The first thing to establish was exactly what the policy expects. Reading `scripts/hoop_env.py` and the installed
`BaseRLAviary` gave:

* **Observation**, shape (1, 75), unnormalised: position (3), roll/pitch/yaw (3), velocity (3), angular velocity
  (3), then the last 15 actions (15 x 4, oldest first; at 30 Hz that is half a second), then `hoop - position` (3).
  No normalisation wrapper is used in training, so there are no statistics files to load.
* **Action**, shape (1, 4), the `vel` type: dims 0 to 2 give a **direction in the world frame** (only the direction
  matters, its length is thrown away), dim 3 is a **speed fraction** (its sign is ignored) of 0.25 m/s. There is no yaw
  action: the simulator's controller holds the yaw it started with.
* Control rate 30 Hz; an episode ends when the drone is within 0.15 m of the hoop, or after 15 s, or if it leaves the room
  by 0.3 m, or tilts past 0.5 rad.
* One quirk: the simulator's 15-action buffer is not cleared between episodes, so a new episode starts with the previous
  episode's last actions in it. A real flight starts from rest, so the hardware starts with zeros instead.

The Tello gives less than the simulator's ground truth, so some of the 75 numbers are measured and some estimated:

| quantity | where it comes from on hardware |
|---|---|
| x, y | overhead ArUco camera, fused with dead-reckoning (the filter) |
| z | Tello's height reading |
| velocity | Tello velocity rotated to the cage frame, fused with the camera |
| roll, pitch | Tello attitude |
| yaw | Tello yaw plus a slowly estimated offset from the camera |
| angular velocity | **estimated** from the 10 Hz attitude (low-passed d/dt), not measured |
| action buffer | what was actually commanded (after the geofence), zero at takeoff |
| hoop minus position | the hoop position (from `hoop.json` or `--hoop`) minus the estimated position |

Checked by `sim_check/obs_parity.py`: it drives the real `HoopFindAviary` with random actions, and our `obs_builder` with
the identical state and actions, and compares all 75 numbers each step. Result: maximum difference **0.0**.
The same script asserts the cage size, speed limit, control rate, buffer length and reach radius still match.

## 3.4 Why it is built this way

| decision | why | what was rejected |
|---|---|---|
| Position from an overhead ArUco camera fused with Tello dead-reckoning in a Kalman filter | the Tello reports height and velocity but **no x/y position**. Integrating its velocity drifts; the camera never drifts but is noisy, late and drops out. The filter takes the good part of each. This was the user's specified design | dead-reckoning alone (drifts); camera alone (blackouts, latency) |
| A small Kalman filter written in numpy, not FilterPy | FilterPy is unmaintained and fragile with numpy 2.x, which this environment uses; the maths is short and the noise settings map one to one onto FilterPy's | FilterPy |
| Yaw treated as a known input to the filter | keeps every measurement linear, so the filter is exactly the EKF one would write here, with no linearisation error to tune | a full nonlinear EKF with yaw in the state |
| Demonstrations and cloned policies in the **simulator's own formats**, cloned into a PPO model with identical observation/action spaces | one demonstration file can train or fine-tune the same policy the simulator trains, and the result loads in `replay_hoop.py`, the dashboard and `run_policy.py` | a separate hardware-only policy format |
| The geofence edits the **action**, not the velocity | the policy never saw walls in training; a hidden velocity clamp would fight it invisibly and the policy's next input would not match what happened. Editing the action means the action buffer records what was really commanded | clamping velocities after the fact |
| Three safety levels (hold, land, kill) and promotion of a stuck hold to a landing | a hover that lasts forever is not a safe state; "kill" must stay a deliberate human choice when the drone is high | one "stop" |
| Only monotonic clocks for control timing (wall-clock time is used just for file names and the hoop file's age) | wall-clock time can jump (clock adjustments); the filter's timing must not | `time.time()` in the control path |
| One `FlightSession` that every script goes through | there is exactly one place where takeoff checks, safety, geofence and logging happen, so no script can skip them | each script talking to the drone itself |
| `cv2`, `pygame`, `djitellopy`, `pynput` imported only where used | the whole test suite (except 8 tests) and every `--dry-run` run without them, on any machine | importing everything up front |
| Logs/demos in `flights/`, not `runs/` | the project `.gitignore` already ignores any folder called `runs/` at any depth, which would have silently excluded them for the wrong reason | `hardware/runs/` |
| Calibration results in `config.local.yaml` (git-ignored) | they describe one drone and one camera; a friend cloning the repository must not inherit your numbers | committing them into `config.yaml` |
| Everything under `CMS/` | the user wants to push the whole thing to GitHub | files elsewhere on the machine |

## 3.5 Deep dives

### Where is the drone? (`state_estimation/`)

**Camera geometry (`aruco_tracker.py`).** Four markers lie flat on the floor at known cage positions. Their pixel
positions give a *homography* (a flat-plane mapping, fitted by least squares from the four or more markers, and refined
as the median over recent frames so one bad frame cannot spoil it) from image to floor coordinates. The drone's marker is
about 0.8 m above the floor, so on the floor plane it looks pushed away from the camera's nadir; the code undoes this with
`true_xy = nadir + (floor_xy - nadir) * (H - z) / H`, using the filter's own height estimate `z` and the camera height `H`.
The yaw of the drone is the direction from the marker's bottom edge to its top edge. The camera thread stamps each frame
the moment it is grabbed, before decoding, because that is the timestamp the latency correction needs. Because the drone
can hide a floor marker from above when it flies high near a corner, the floor mapping is learned while all markers are
visible and then retained; a frame missing a marker does not degrade it. (A test checks this.)

**The filter (`fusion_kf.py`).** State: position and velocity, `[x y z vx vy vz]`. Between measurements it assumes
constant velocity with random acceleration (`q_acc` = 0.8 m/s^2). Three kinds of measurement update it:
Tello's velocity rotated into the cage frame (`sigma_vel` 0.06 m/s), Tello's height (`sigma_z` 0.03 m), and the camera's
x/y (`sigma_cam_xy` 0.03 m, divided by the marker's quality, and inflated by `(speed x latency jitter)^2`). A camera
position is checked against a gate first (a 99.9 % chi-squared test); one bad detection is thrown out. If five in a
row are thrown out, the filter concludes that it is the one that is wrong and re-acquires.
Late measurements are handled properly: the filter keeps 1.5 s of past states and measurements, and a camera frame
stamped earlier than "now" rewinds to the state just before it and replays everything after it in time order.
While the camera is gone, the uncertainty grows; the estimator reports `pos_health`: `OK`, `DEGRADED` (no camera for
0.5 s), or `LOST` (position uncertainty above 0.35 m), and `LOST` makes the safety layer hold, then land.

**Yaw.** Tello's yaw is relative to its takeoff heading and drifts slowly; the camera's yaw is absolute but noisy. The
cage-frame yaw is `Tello yaw + offset`, where the offset is learned from the camera slowly (at most 5 degrees per second,
low gain) and only from readings that agree with the current estimate. If camera and Tello disagree by more than 15
degrees for a second, the offset is frozen (`DEGRADED`); beyond 30 degrees, or on a sudden 20 degree jump, it is `LOST`.
The camera yaw is never used to steer directly.

**Telemetry timestamps (`time_sync.py`).** Tello state packets arrive about 10 times a second with no sequence number,
sometimes in bursts after a delay, sometimes duplicated. Each batch is cleaned: exact duplicates within 50 ms are dropped;
packets arriving together are spread back across the 100 ms grid so the newest keeps the arrival time; times are forced
strictly increasing with at least 5 ms between samples (this minimum came from a bug, see 3.7); and each packet is checked
(fields present and finite, battery 0 to 100, tilt under 90 degrees, speed under 3 m/s, implied acceleration under
20 m/s^2, yaw rate under 400 degrees per second). Rejected packets are counted by reason. If nothing valid arrives for 0.5 s
the safety layer holds; for 2 s, it lands.

**Camera latency (`latency.py`).** The camera image is slightly older than its timestamp. The configured baseline
(`cam_latency_s` 0.08) is corrected online: while the drone is moving, the tracker slides the Tello's velocity against the
camera's velocity over 5 s to find the delay with the smallest squared error (searching 20 to 200 ms), smooths it, and
reports how jittery it is; fast motion with jittery timing makes the filter trust the camera less. If the estimate is
unstable it falls back to the baseline. (Plain correlation turned out to be nearly flat across delays for smooth motion,
which is why squared error is used; see 3.7.)

How to tune all of this for your lighting and room: `state_estimation/TUNING.md`.

### The safety layer (`common/safety.py`, `common/geofence.py`, `common/flight_session.py`)

Every tick, the safety monitor looks at the estimate and the session state and returns a level: none, **hold**
(zero command, hover), **land**, or **kill** (motors off). Land and kill are *latched*: once raised, they stay for the rest of
the session. A hold that lasts 2 s becomes a land.

| trigger | level | threshold (from `config.yaml`) |
|---|---|---|
| human: SPACE | hold (toggle) | |
| human: L, ESC, Ctrl+C | land | |
| human: X, BACKSPACE | **kill** | |
| no Tello packets | hold, then land | 0.5 s, then 2.0 s |
| position estimate lost | hold (then land after 2 s) | `pos_lost_std` 0.35 m |
| heading estimate lost | hold (then land after 2 s) | 30 degrees or a 20 degree jump |
| drone tilted | land | 0.5 rad on roll or pitch |
| beyond a wall | land | 0.15 m past the cage wall |
| battery low | land | smoothed under 20 % for 3 s |
| battery critical | land | raw under 10 % twice in a row |
| overheating | land | smoothed over 85 C for 3 s |
| flight too long | land | 120 s |
| NaN or out-of-range action | hold | |
| loop running slow | hold | a tick over 0.1 s, three times |
| commanded and observed direction disagree | hold | opposite directions for 1 s |
| the policy keeps being trimmed by the fence | hold | trimmed over 30 % of the last 2 s |

Before takeoff the session refuses unless: the axes are calibrated (real drone only), the estimate is ready, camera and heading
are healthy, 5 battery samples have been seen, and the average battery is at least 30 %.

**Battery and temperature** are smoothed (median of 7 readings; the battery estimate can only go down while flying,
so a glitchy upward jump is ignored and a momentary sag under load cannot trigger a landing on its own).

**Landing** is a small state machine so that a landing cannot silently fail. It sends `land`, re-sends every 3 s while
the drone is still up, and watches the *smoothed height*. "Landed" means height under 10 cm and flat (changing less than
6 cm over 0.6 s) for 0.3 s. It only cuts the motors itself if the descent has stalled low: 5 s after landing began and
under 0.3 m and not descending, or 10 s in and under 0.5 m. It never cuts the motors above 0.5 m on its own. If the height
reading is unavailable, time alone decides (15 s). Only a human pressing X can cut the motors higher up.

**Geofence.** The cage is 2x2 m wide and 0.15 to 1.9 m high. Within 0.30 m of a wall the outward part of the commanded
speed tapers down, reaching zero 0.10 m inside the wall (the drone needs room to coast), and past that line a gentle inward
push (up to 0.08 m/s) fights slow drift. The result is written back as a normal simulator-format action and goes into the
action buffer. Beyond the wall by 0.15 m the safety layer lands. A test confirms a drone pushed continuously into a wall
stops inside the cage.

### Imitation learning (`imitation/`)

> Step-by-step tutorial, including the onboard-camera drone policy: [`imitation/README.md`](imitation/README.md).
> The description below is of the older simulator point-reaching task.

A demonstration is, for each 1/30 s while you fly: the 75-number observation, what you asked for (direction plus speed),
and what was actually executed (after the geofence). Saved as `.npz` with the flight's metadata. Behaviour cloning trains a
PPO policy (same network shape the simulator trains) to reproduce your action from the observation. Details that matter:

* Only the **actor** half of the network is trained; the critic is left alone, or optionally warmed up (`--warm-critic`) on
  returns recomputed from the demonstration (the reward depends only on the observation), because a random critic ruins a
  cloned actor when PPO fine-tuning starts.
* Loss: a **cosine loss on direction** (the simulator's controller discards the magnitude of the direction vector, so only
  its direction matters) plus squared error on the speed fraction. Idle samples (speed near zero) do not train the direction.
* After cloning, the policy's exploration noise (`log_std`) is set small so a later PPO run does not swamp it.
* The saved zip has exactly the simulator's observation/action shapes, so it loads in `replay_hoop.py`, the dashboard (with
  `--as-run`) and `run_policy.py`.
* Keyboard demonstrations use few directions at a nearly constant speed; the clone learns that style.

**Measured in the simulator** (not on hardware): a trivial "fly straight at the hoop at full speed" expert reaches the hoop in
**60 of 60** random episodes; a clone trained on 7,433 of its samples reached the hoop in **30 of 30** fresh episodes
(mean final distance 0.15 m). The 500k-step PPO from the project README: 9 of 20. So, in this task, a clean demonstration is a
stronger starting point than the current RL result. Caveat: that is the simulator, with perfect state.

### Calibration (`calibration/`)

* `calibrate_camera.py`: standard chessboard calibration, writes the lens parameters.
* `calibrate_axes.py` and `calibrate_speed.py` fly **raw rc hops** (they bypass the action mapper so the assumptions under
  test are not in the loop) and use the camera, never the fused estimate (whose velocity input depends on the very signs being
  measured). `analysis.py` turns the recordings into results: for forward/back and left/right it projects the camera
  displacement onto the drone's body axes and checks direction and that the right axis moved; for up/down it uses the Tello's
  height reading (the overhead camera cannot see height); for yaw it compares camera heading change with the Tello's own yaw
  reading. It also finds the unit of the Tello's velocity numbers by comparing them with the camera's speed. If any channel
  cannot be established, it refuses to set `axes_calibrated`.
* The calibration scripts are allowed to take off even though axes are not yet calibrated (they are how that gets fixed); a
  normal session still refuses. (This was a bug found late, see 3.7.)
* `hoop_locator.py`: averages 30 camera frames of the hoop marker, discards stray detections, rejects a noisy reading (over 2 cm
  spread), and range-checks the result against what the policy trained on, then writes `flights/hoop.json`, which expires after 10
  minutes.

### The simulated Tello (`common/fake_tello.py`)

Used by every `--dry-run` and by the tests. Three presets: `ideal`, `typical`, `harsh`. It models: a small dead-band on the rc
commands, different speed-up and slow-down times (it brakes slower than it accelerates), acceleration limits, command delay with
jitter and occasional lost commands, biased and rounded velocity readings, height rounded to the centimetre with extra noise near
the ground, slow yaw drift and random wind-like drift, takeoff to about 0.8 m by itself, a battery that sags under load, and
telemetry at 10 Hz with delays, bursts, duplicates and dropped packets. There is also a simulated overhead camera (delay, noise,
dropouts, blackouts) and an option to mirror left/right (to see the axis checks work).

**What it can and cannot prove.** It is *my model* of how a Tello behaves, written from the Tello documentation and common
knowledge, not measured from your drone. It proves the software logic works (that the safety triggers fire, that landing
completes, that the filter survives blackouts and late data). It says nothing about how your Tello will actually respond.
Once `calibrate_speed.py` has measured your drone, put the numbers into a profile.

## 3.6 How it was made

**Starting point.** The project already had a simulator task (`HoopFindAviary`: fly a randomly placed hoop in a 2x2x2 m cage)
with trained PPO policies and a web dashboard, and an open question in `PROGRESS.md`: which real drone, and could the policy
ever run on it. The user supplied a transcript of a Tello programming course as a reference. It covers: connecting with
`djitellopy`; the four-number `send_rc_control` command (-100 to 100 for left/right, forward/back, up/down, turn); a pygame
keyboard module (hold keys to move, `e` to take off, `q` to land); saving camera frames; dead-reckoning "odometry" from commanded
speed; and face tracking. Two of its lessons shaped the design: speeds do not match the numbers (rc 15 gave about 12 cm/s),
and a keyboard land key is a necessary safety feature.

**What had to be found out.** Reading the simulator code (3.3) showed that the policy needs the full 75-number input,
including position the Tello cannot report, and that its actions are world-frame velocities with yaw held. That produced the
list of mismatches the rest of the work addresses: no x/y position, no angular rates, body-frame (not world-frame) rc commands,
10 Hz telemetry instead of 30 Hz, and different dynamics from the simulator's controller.

**What the user asked for.** (1) connect the RL system to the Tello for sim-to-real checks; (2) three ways to drive it:
keyboard, importing sim-trained policies, and imitation learning from recorded runs; (3) an emergency stop and safe-landing
buttons; (4) a position estimate combining an overhead ArUco camera (absolute position) with Tello telemetry
(dead-reckoning) in a filter that handles camera dropouts and outputs `[x y z vx vy vz yaw pitch roll]`, with guidance on tuning
the noise to trust each source (`StateEstimate.vector()` returns exactly that vector; `TUNING.md` is the guidance); (5)
behaviour cloning as the imitation method; (6) everything inside `CMS/` so it can be pushed to GitHub; (7) this README in the
three-part shape you are reading.

**Process.** The work was planned first (read-only: explore the simulator and the repository, design, then a written plan).
The plan was sent back by the user twice before it was approved and any code was written. Before that, two clarifying questions fixed how position is obtained (camera fused with dead-reckoning) and how imitation is done (behaviour cloning). The first review produced ten points; each was folded into the
design and is mapped below. The second asked for everything to be kept inside `CMS/`. Then it was built bottom-up (constants,
smoothing, timestamps, filter, estimator, action/observation, safety, simulated Tello, session, scripts, calibration, imitation,
tools, docs), writing the tests alongside and running them as each piece landed.

### The user's ten review points and what was done

| # | point | what was implemented | where |
|---|---|---|---|
| 1 | Yaw drift, oscillation, inconsistent axes | slow rate-limited yaw offset (5 deg/s), yaw health flag (degraded at 15 deg for 1 s, lost at 30 deg or a 20 deg jump); damped yaw-hold with dead-band and clamp; runtime axis-mismatch monitor (commanded vs camera-observed direction opposite for 1 s) | `fusion_kf.py`, `action_mapper.py`, `estimator.py` |
| 2 | Action buffer on takeoff | buffer cleared to zeros at takeoff; 1.5 s hover warm-up before the policy runs; speed ramps in over 1 s; the buffer stores executed (post-geofence) actions | `flight_session.py`, `obs_builder.py` |
| 3 | Bursty, delayed, duplicated telemetry | monotonic time only; duplicate removal; burst spreading; 5 ms minimum spacing; per-packet sanity checks; stale detection | `time_sync.py` |
| 4 | Fixed camera latency | baseline plus online squared-error estimate with jitter; noise inflated by speed times jitter; capture stamped at grab | `latency.py`, `aruco_tracker.py`, `fusion_kf.py` |
| 5 | Noisy height near the ground | smoothed height; "landed" from flat height rather than noisy velocity; stall-aware time fallback; blind-height time rule; never cut high up | `safety.py` (`LandingSequence`) |
| 6 | Geofence fighting the policy | fence acts on the action, tapers with a zero-speed line inside the wall and an inward push; recorded in the buffer; "fighting the fence" fault | `geofence.py` |
| 7 | BC model choice | actor only, cosine direction loss, small exploration noise, optional critic warm-up, PPO and A2C; sim rollout as a fidelity check | `bc_train.py` |
| 8 | FakeTello realism | three profiles with dead-band, asymmetric acceleration, latency, quantisation, drift, battery sag, telemetry bursts/duplicates/drops | `fake_tello.py` |
| 9 | Hoop detection | marker under the hoop; averaged, outlier-rejecting `hoop_locator`; `validate_hoop` overlay; freshness limit; confirmation prompt; range check | `calibration/` |
| 10 | Battery jumps | median plus monotone filter; land only after 3 s below threshold; critical needs two readings | `smoothing.py`, `safety.py` |

## 3.7 What testing found

The end-to-end tests flying the simulated Tello found real bugs that unit tests would not have. Each now has a regression test.

| symptom | cause | fix |
|---|---|---|
| the pretend drone spun at about 24 degrees/s while commanded to hold heading | two telemetry samples got timestamps one floating-point epsilon apart, so a yaw difference was divided by a tiny time and the yaw-rate estimate reached about 1e10; that fed the yaw damping and saturated the command | at least 5 ms between accepted samples; the yaw-rate estimate ignores gaps under 20 ms and is clamped to 4 rad/s |
| a drone that had landed was declared "not landed" and had its motors cut | "landed" required the filter's vertical velocity to be small, which is noisy exactly near the ground | landed is decided from the flatness of the smoothed height |
| the first fix made it worse | the height history was only fed once below 10 cm, and the fixed 5 s escalation hit drones still legitimately descending | history fed every tick; escalation only when the descent has stalled |
| a drone pushed at a wall passed through it | speed fell to zero only at the wall, but the drone coasts about 0.16 m (command delay plus slow braking) and also drifts | zero-speed line 10 cm inside the wall, an inward push past it, and a hard-land margin beyond |
| the camera latency tracker never updated | correlation between smooth velocity signals is almost the same at every delay, so the "clear peak" test failed | squared-error minimum with parabolic refinement |
| in a synthetic scene the top corner markers were not detected | a 70 degree webcam at 3 m sees only about +-1.15 m on its short side | recommended camera height raised to about 3.5 m |
| a drone flying high near a corner hid that corner's marker | seen from above, a high drone projects onto the floor right over the corner marker | floor mapping learned once and retained |
| **calibrate_axes.py could not take off on a real drone** | real takeoff needs calibrated axes, but calibration is how they get calibrated; the dry run could not show this because the pretend drone is not treated as real hardware | the calibration scripts' session skips only that one check; a test asserts a normal session still refuses |

## 3.8 Verification record

What was run, and the results:

| check | result |
|---|---|
| automated tests, with OpenCV and pygame installed (scratch environment) | **99 passed** |
| automated tests in the project's `drones` environment | 91 passed, 1 skipped (the file needing OpenCV and pygame) |
| `obs_parity.py` against the live simulator | maximum difference **0.0** over 60 random steps; all mirrored constants match |
| scripted expert in the simulator | reached the hoop in 60 / 60 episodes |
| behaviour clone of that expert, in the simulator | reached the hoop in 30 / 30 episodes |
| the 500k-step PPO (`results/hoop/best_model.zip`) on the pretend Tello | reached the hoop and landed on `ideal`, `typical` and `harsh` |
| calibration scripts against the pretend Tello | axes correct (and a mirrored left/right detected and fixed); rc scale recovered to within about 3.4 % (120.8 against a true 125) |
| the camera pipeline on rendered ArUco markers | position within 2 cm and yaw within 4 degrees at four poses; a camera thread reading a video file produced correct detections |

Tests by file: `test_safety` 19 (every trigger, landing escalation, preflight, battery spikes), `test_fusion` 10
(filter accuracy, blackouts, late data, outlier gate, yaw health, latency tracker, end-to-end estimator on the pretend
drone), `test_control` 9 (keyboard and policy programs, recording, kill key, takeover, out-of-range hoop, wrong-shape model, the real
trained model), `test_session` 9 (flights through the whole session: takeoff refusal, geofence, kill, hold, telemetry loss, flipped
axes, hoop reached, action buffer, speed ramp), `test_cv_paths` 8 (OpenCV and pygame), `test_action_mapper` 7, `test_imitation` 7,
`test_time_sync` 7, `test_calibration` 6, `test_geofence` 6, `test_aruco_math` 5, `test_tools` 4 (log tuning, sim-vs-real, hoop
aggregation), `test_obs_parity` 2.

**Not verified, and what that means for you:**

* **No real Tello flight.** Everything about how your drone actually responds is open until you do Part 2, Stage C.
* **`djitellopy`** (the Tello library) is not installed here. `tello_link.py` is written to its documented interface
  (`connect`, `send_rc_control`, `takeoff`, `land`, `send_command_without_return`, `get_current_state`) but has never been run against it.
* **The global emergency hotkeys** (`pynput`) have not been run. The in-window keys have (event handling and drawing, headless).
* **A physical camera** was never used: the camera thread was tested reading a video file. `calibrate_camera.py`,
  `hoop_locator.py` and `validate_hoop.py` need a real camera and have not been run end to end (their logic is tested separately).
* **The Tello's axis signs and velocity units** are assumptions in `config.yaml` marked UNVERIFIED.
  One more such assumption: that the Tello's `h` reading is height above where it took off, so **take off from the cage floor**.

## 3.9 Known gaps and next steps

1. **Takeoff height.** The simulator always starts the drone at 0.1 m, still, but a Tello takeoff settles at about 0.8 m. The first
   observations are outside what the policy practised. The code softens this (1.5 s warm-up, speed ramp); the real fix is to
   randomise the start height in `HoopFindAviary` and retrain.
2. **Dynamics.** The simulator's drone follows velocity commands through a PID with almost no lag; the Tello is slower to start
   and stop and has command delay. `calibrate_speed.py` aligns the steady speed and reports the response time; the shape of the
   response remains different. A fix on the simulator side: add command latency and action noise when training.
3. **Angular velocity** is estimated, not measured. In a stress test with random full-range actions in the simulator the estimate
   was rough (median error about 1.45 rad/s against signals of about 0.85); in smooth flight these values are small.
4. **Policy quality.** The best RL policy reaches the hoop 9 times in 20 in the simulator; the cloned expert does better, so the
   imitation route is worth taking seriously. The cloned policy was slower than the PPO on the pretend Tello (about three times
   as many ticks); the cause is not investigated.
5. **Fine-tuning a clone with PPO.** The clone saves a warmed-up critic so PPO can continue from it, but `scripts/train_hoop_multi.py`
   has no "start from this model" option yet (not added, to avoid changing your simulator scripts).
6. **Hoop height** always comes from configuration, because an overhead camera cannot measure it.
7. **The Tello's built-in obstacle sensing and its video camera** are not used.

## 3.10 Glossary

| term | meaning |
|---|---|
| **action (simulator format)** | `[dx, dy, dz, s]`: a direction in the cage frame and a speed fraction `s` of 0.25 m/s; yaw is held |
| **ArUco** | a family of printed square markers detected by OpenCV; each has an id number |
| **behaviour cloning (BC)** | teaching a policy by supervised learning to copy demonstrated actions |
| **body frame / FLU** | axes attached to the drone: x forward, y left, z up |
| **cage frame** | axes attached to the room: origin at floor centre, x/y on the floor, z up |
| **demonstration** | recorded observations and actions from a human or expert flight |
| **dry run** | running with the simulated Tello and camera instead of hardware |
| **executed action** | the action actually sent after the geofence and speed ramp; it is what goes into the observation's buffer |
| **geofence** | the virtual walls that trim actions near the cage boundary |
| **Kalman filter** | an estimator that combines noisy measurements with a motion model, weighting each by how much it is trusted |
| **observation** | the 75 numbers the policy sees |
| **policy** | the trained network mapping observation to action |
| **rc** | the Tello's remote-control command: left/right, forward/back, up/down, turn, each -100 to 100 |
| **health (OK / DEGRADED / LOST)** | the estimator's own verdict on whether position and heading can be trusted |
| **homography** | the flat-plane mapping from camera image to floor coordinates |
| **latched** | a safety level that stays raised for the rest of the session once triggered |
