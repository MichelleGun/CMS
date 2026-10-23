# Hand-over: training the hoop-looping Tello policy together

Written 2026-10-15 for everyone helping to train, updated 2026-10-23. Read this first, then the
illustrated version in [docs/Onboard Hoop Policy Anatomy.html](docs/Onboard%20Hoop%20Policy%20Anatomy.html)
(§18–23). **Section 7 is the running log of what changed, why, and what it did.**

## 1. Where we are

**Goal.** A Tello flies through a 30 cm hoop using only its own camera, then loops round the outside
and re-enters, as many times as possible. We train in our simulator (`hardware/onboard/sim.py`) and
fly the result on the real drone. Score = **laps per episode** (passes in a 90 s episode, same-side
laps, starts from all 9 zones of the cage).

**Best policies so far** (sim scores from `hardware/onboard/export_policy.py`; full ranking in
[hardware/policies/README.md](hardware/policies/README.md)):

| Policy | Laps / 90 s | Episodes with 2+ passes | Oblique entry* | Notes |
|---|---|---|---|---|
| `w10_mg6_C9_loop_practice` (final) | **2.33** | **64 %** | 57 % | **current best** (optimiser, 2026-10-19): O + stress drone, then more slanted and near-hoop starts |
| `w10_mg2_E9_stress` (best) | 2.00 | 64 % | 53 % | O + a slower, driftier drone in 30 % of episodes |
| `w10_n9_O_oblique` (best) | 1.58 | 53 % | 67 % | trained with slanted starts |
| `w10_n9_OZ_oblique_clearance` (best) | 0.94 | 19 % | 57 % | slanted starts + line-up reward |
| `n8_dapg_nodrain_cont2` (best) | 1.03 | 22 % | 70 % | **the one flown on the real drone** |
| best policy *without* memory | 0.78 | 0 % | – | never loops |

\* Oblique entry = share of 30 slanted starts near the hoop (35–80° off its axis, like the cage
corners) that end in a clean pass.

**The open problem: corners.** The policies loop more and more (1.03 → 1.58 → 2.33 laps/ep), but
clean entry from slanted starts went the other way (70 % → 67 % → 57 %). That slanted entry is the
corner clipping seen on the real drone. So since 2026-10-23 the optimiser scores corners as heavily
as loops (section 7).

**What the real drone did** (n8_cont2, 2026-10-14). From the bottom-right corner it got in 2 times
out of 3 but rarely looped. From the other three corners it found the hoop but **clipped the edge
going in**. Height and hoop detection were good. Some flights were cut short by our own safety
checks (ceiling, a bad range-sensor reading, mainboard over 80 °C). The first two are fixed in
`fly_hoop.py`.

**What worked** (details and numbers in the anatomy page):
1. PPO with the "trust-region brake" (target_kl 0.02, batch 512, 5 epochs).
2. The ring tracker (4 inputs: where the ring is while it is out of view).
3. Switching off imitation of human demos: they never loop, so copying them stops the loop.
4. **Memory**: the policy sees the last ~1 s of frames (history 0,1,2,4,8,16), the critic gets
   privileged sim inputs (asymmetric critic), and it starts from a DAgger clone of the privileged
   teacher, with PPO keeping a constant imitation term (DAPG). This line (`n8_*`, `w10_n9_*`) is the
   only one that loops.

**What didn't work** (so you don't repeat it):

| Tried | Result | Why |
|---|---|---|
| gamma 0.995, a bigger streak bonus | worse | not the bottleneck |
| near-pass starts (start just before the hoop) | no better | more practice of the loop doesn't help without memory |
| teacher labels on the loop leg for a policy **without** memory | collapsed to ~0 | to a memoryless policy "just passed, ring behind" looks like "blind before a pass", so the labels average out |
| bigger reward for re-finding the ring | no change | |

**Lighting matters (2026-10-17).** Night flights worked; daytime flights locked onto the window.
Calibrate the camera per lighting in the dashboard (Drone tab → *Calibrate camera…*, see
[webapp/README.md](webapp/README.md)) and pick the matching **Camera preset** when flying.

## 2. Setup

1. Follow [SETUP.md](SETUP.md) (conda env `drones`, Python 3.12, CPU-only PyTorch is fine).
2. `pytest hardware/onboard/tests -q`: expect 4 known failures (detector.json in
   `test_fly_hoop.py` / `test_human.py`); everything else should pass.
3. The demos a memory run needs are in the repo:
   `hardware/flights/demos/dagger_n1_r*.npz` (14 files, 442k teacher labels).

## 3. Training one experiment

Everything goes through `hardware/onboard/loop_night.py`, which runs a set of experiments (defined in
`hardware/onboard/week10_experiments.py`, `EXPERIMENTS`). It stops clear losers, extends the best,
and exports the results. Start it **detached**, so closing a terminal or a stray Ctrl+C can't kill it:

```bash
python hardware/onboard/launch_detached.py results/hoop/loop_<yourname> -- hardware/onboard/loop_night.py --vec subproc --keys O --parent results/hoop/runs/w10_n9_O_oblique --tag <yourname> --timesteps 3000000 --n-envs 4 --parallel 1 --resume-crashed
```

- `--tag <yourname>` names your run folders `results/hoop/runs/w10_<yourname>_<key>_<name>`, so
  nobody's folders clash.
- `--parent` is the model you continue from (a run folder continues from its latest model).
- `--n-envs` is the number of parallel simulators per run (each about 1 CPU core and ~0.7 GB of
  memory); `--parallel` is the number of runs at once. Training runs at below-normal priority.
- To stop a run cleanly, create a file `stop.flag` in its folder. It saves `final_model.zip`.
- **Faster training** ([TRAINING_SPEEDUPS.md](TRAINING_SPEEDUPS.md)): evaluation runs on several copies
  with `--vec subproc` (it was ~60 % of the wall clock), `latest_model.zip` is saved every 15 min, and
  `rescore_checkpoints.py` re-scores a run's policies on 100 episodes before you export one.
- **Watch:** `results/hoop/loop_night.log` (one line per check), the run's `passes.json` (one entry
  per evaluation: mean laps/ep, max, per zone), and the report
  `results/hoop/overnight_loop_report_<tag>.md`.
- **Export** (score + copy a Tello-ready file into `hardware/policies/`):
  `python hardware/onboard/export_policy.py results/hoop/runs/w10_<yourname>_O_oblique`
- Windows: keep the PC awake and plugged in (sleep killed a run). If a run dies with a memory
  error, lower `--n-envs` / `--parallel`. `--resume-crashed` continues it from its last save.

## 4. Next experiments: pick one, put your name next to it

All continue from `results/hoop/runs/w10_n9_O_oblique` (the current best). Success means beating O
on **laps/ep** and **2+ passes %** without dropping **oblique entry**, ideally oblique entry ≥ 90 %.

| # | Experiment | Hypothesis | How | Who |
|---|---|---|---|---|
| a | **O, longer** | O was still improving when stopped (best eval 1.64) | `--keys O`, 5M steps | |
| b | **Z / OZ: line up first** | a reward for lining up on the axis and keeping clear of the noodle stops the corner clipping seen on the real drone | `--keys Z OZ` (`w_rim` 1.0, `rim_margin_m` 0.15, `axis_pull` 2.0, `r_struck` -15) | |
| c | **More slant + loop practice** | more slanted starts, plus starts just before the hoop, give more loop practice now that the policy has memory | new key: O's flags with `--oblique-p 0.6 --near-pass-p 0.3` | |
| d | **Second DAgger pass with memory** | the teacher relabelling O's own states (DAgger) then DAPG again lifts the loop rate, as `dagger_n1` → n8 did | `hardware/privileged/dagger_hist.py` from O, then DAPG as in `_N8` | Michelle's PC (`loop_forever.py`, tag `mg`: key **D9**) |
| e | **Sim-to-real: corners** | real drift and command delay make slanted entries fail; more delay/drift randomisation in training makes the policy line up earlier | raise the delay/drift ranges in `sim.py` profiles (`sample_profile`), then `--keys O` | Michelle's PC (key **E9**: `--stress-p 0.3`, +0.06 s delay, drift x2) |

**Running continuously on Michelle's PC** (since 2026-10-17; round 2 since 2026-10-23):
`hardware/onboard/loop_forever.py` races the best policy so far against the next idea, round after
round, and carries the winner forward. Pool now: CZ, CE, CD, CL, CO (all built on C9; section 7).
Runs are `results/hoop/runs/w10_mg<round>_*`; the log is `results/hoop/mg_loop.log`. It starts by
itself at logon (`CMS loop optimiser.cmd` in the Startup folder, `Win+R` → `shell:startup`); start it by
hand with `hardware\onboard\start_loop_optimiser.cmd`, and stop it after the current round with
`type nul > results\hoop\mg_loop.stop`. To avoid duplicate work, pick other keys or tags.

To add an experiment, add a line to `EXPERIMENTS` in `week10_experiments.py` (copy `"O"` and change
the flags), and use its key with `--keys`.

**Hierarchical RL (2026-10-23).** `hardware/onboard/hrl_pipeline.py` runs unattended: skills SE and SL
(`loop_night.py --tag hrl`), then the manager (`train_hrl.py --tag v1`), then a fair comparison
(`results/hoop/hrl_compare.md`). A hierarchy flies like any policy: `fly_hoop.py --mode policy --model
results/hoop/runs/hrl_v1/best_model.zip` (learned) or `.../hrl_v1/phase_bundle.json` (fixed switch).

**Drone Quidditch website (2026-10-23).** The dashboard and the website are one frontend
(`webapp/frontend`). Its **Overview / Policies / Training** tabs show the milestone policies flying in 3D
(recorded in the simulator), the leaderboard, learning curves, the optimiser's rounds and the HRL
comparison, from `webapp/frontend/data/`. GitHub Pages (https://michellegun.github.io/CMS/) serves the same
page read-only: `scripts/build_pages.py` sets `DQ_STATIC`, and the dashboard tabs then read a snapshot of the
API (`data/api/`, a few runs only) instead of a server; anything that would train or fly points to the repo.
Refresh after new results: `python scripts/build_site.py --record`, commit `webapp/frontend/data` and push
(the Pages workflow republishes). The drone in every 3D view is `webapp/frontend/js/dronemodel.js`.

## 5. Sharing results

- Work on your own branch: `git checkout -b train/<yourname>`.
- Commit **your run folders** (`results/hoop/runs/w10_<yourname>_*`; `.gitignore` already keeps
  them to models + meta + passes) and your exported files in `hardware/policies/`, plus any code
  change. Pull before you push, then push the branch. Michelle merges.
- Write 2–3 lines in your branch's commit message: what you ran, laps/ep, 2+ passes %, oblique entry.
- Michelle flies the candidates on the real Tello (`fly_hoop.py --mode policy --model <file>`).
  Real-flight notes go back into this file.

## 6. Gotchas

- The 4 detector.json test failures are known, not yours.
- `best_model.zip` is picked on 36 evaluation episodes; check `passes.json` and the export, not
  only the best number.
- Memory policies (history + asymmetric critic) need `--ring-tracker --history-lags 0,1,2,4,8,16
  --asym-critic` and the `dagger_n1` demos. The `_N8` flag list in `week10_experiments.py` has it all.
- Don't run `loop_night.py` attached to a terminal you might close: use `launch_detached.py`.

## 7. What changed, why, and what it did (running log)

Newest first. "laps/ep" = passes per 90 s sim episode (same-side laps, all 9 start zones); "oblique"
= share of slanted starts near the hoop that end in a clean pass (the cage-corner approach).

| Date | Change | Why | Impact (measured) |
|---|---|---|---|
| 2026-10-23 | **Hierarchical RL** (`hrl.py`, options framework): skills **SE** (enter, corners) and **SL** (loop) trained from C9, a **learned manager** (PPO over skills, decides every 0.53 s or at a pass) and a **fixed switch** baseline (enter until the first pass, then loop) | One flat policy traded corners against loops (loop practice: corners 67 % -> 57 %; corner reward: fewer loops). Separate skills don't have to trade | pending: `results/hoop/hrl_compare.md` compares flat C9, the fixed switch and the learned manager on the optimiser's referee |
| 2026-10-23 | **Optimiser fixes** (`loop_forever.py`): a new seed every round; every candidate re-scored on the same layouts with 72 laps episodes + 60 slanted starts; a challenger must win by **> 0.10**; score = laps/ep **+ 1.0 x oblique** (was 0.5); restarts at logon; one copy at a time (lock file) | Rounds 4 and 5 trained the *identical* run (seed 0 both times: both 2.155). One 36-episode number wobbles about +-0.15, so some "new best" calls were noise. Corners got worse while loops improved. The PC shutdown on 19 Oct killed the loop | Fewer wasted rounds; corners now count as much as loops. Results from round 9 on are in the table below this one |
| 2026-10-23 | **New ideas on top of C9**: CZ (line-up / clearance reward), CE (+ stress drone), CD (+ DAgger relabel), CL (lr 2e-5), CO (oblique starts 0.8) | C9 loops best but enters corners worst; these target corners (CZ, CO, CE), loops (CD) or stability (CL) | CZ (round 9): no gain, oblique unchanged at 58 %, and it degraded late in training; more C9 training (seed 9) also lost (2.35 vs 2.75). Others pending |
| 2026-10-19 | Optimiser round 6: **C9 continued** (seed 0) | – | **2.33 laps/ep, 64 % 2+ passes, oblique 57 %**: the current best (`w10_mg6_C9_loop_practice`) |
| 2026-10-18 | Optimiser round 3: **C9 = more slanted starts (0.6) + starts just before the hoop (0.3)**, from E9 | More practice of the loop itself, now that the policy has memory (without memory this didn't help, round 4 of 2026-10-14) | 1.83 → **2.19 laps/ep** (E9 continued vs C9), oblique 57 % → 60 % |
| 2026-10-18 | Optimiser round 2: **E9 = stress drone** (`--stress-p 0.3`: 30 % of episodes with +0.06 s command delay and 2x drift), from O | The real drone clipped the hoop from the corners; a slower, driftier drone in training should teach earlier lining-up | **1.61 → 2.00 laps/ep**, 2+ passes 47 % → 64 %; oblique 63 % → 53 % (worse) |
| 2026-10-17..19 | Optimiser rounds 1, 4, 5, 6: **D9 / DE9** (DAgger relabel of the policy's own states), **OF** (zone focus), **OS** (stall cut 30 s) | DAgger made n8; zone focus and stall cut are the speed-up options | none beat the parent: DE9 2.06, OF 2.21, OS 2.06 vs C9 2.16-2.49; D9 crashed in round 1 |
| 2026-10-17 | **Camera calibration** in the dashboard + saved day/night presets; `fly_hoop.py --detector` | Daytime flights locked onto the window / a glass door (pale blue glare) | On the recorded daytime frames the fitted preset ignores the window and glass door and still finds the hoop at night. Real check pending |
| 2026-10-15 | **Training speed-ups** (Cedric): parallel evaluation, timed checkpoints, zone focus, stall cut, reward normalisation, rescore and sweep tools | Evaluation was ~60 % of wall-clock time | Faster runs; evaluation numbers now about +-0.15 noisy across copy counts (TRAINING_SPEEDUPS.md) |
| 2026-10-14 | **Slanted starts** (O: `--oblique-p 0.4`) on the memory policy | Corners clipped on the real drone | 1.03 → **1.58 laps/ep**; oblique 70 % → 67 % |
| 2026-10-14 | `fly_hoop.py`: soft ceiling, invalid range sensor no longer aborts, soft floor, start at 80 cm | Real flights were aborted by our own checks | Fewer aborted test flights |
| 2026-10-14 | **Memory + DAgger + PPO (DAPG)** (n8, Cedric's line) | A one-frame policy can't tell "just passed" from "before a pass" | 0.78 → **1.28 laps/ep**: the first policy that loops (44 % 2+ passes) |
| 2026-10-14 | Teacher loop labels for a policy *without* memory (L, Q) | Show it the way back | Collapsed to 0.02-0.04: the labels average out without memory |
| 2026-10-13/14 | **No imitation of demos** (W); near-pass starts (N, M) | The demos never loop | W: 0.16 → **0.45 laps/ep**; near-pass starts: no gain |

**Optimiser rounds since the 2026-10-23 fixes** (re-scored; updated as they finish):

| Round | Parent | Raced | Result | Kept |
|---|---|---|---|---|
| 9 | mg6 C9 (re-scored: 2.17 laps/ep, 58 % oblique -> **2.75**) | C9 (seed 9) vs CZ | C9: 1.78 / 57 % -> 2.35; CZ best: 2.11 / 58 % -> 2.69, CZ final 1.87 (it fell apart late) | parent (no win by > 0.10) |
| 10 | mg6 C9 | C9 (seed 10) vs CE (+ stress drone) | C9 best 2.55; CE: 1.86 / 63 % -> 2.49 | parent |
| 11 | mg6 C9 | C9 (seed 11) vs CD (DAgger relabel of C9's states) | C9 final 2.55; CD best: 1.86 / 63 % -> 2.49 | parent |
| 12 | mg6 C9 | C9 (seed 12) vs **CL** (lr 2e-5) | C9 best 2.60; **CL final: 2.58 laps/ep, 61 % 2+ passes, 60 % oblique -> 3.18** | **CL** (first win since C9) |
| 13 | mg12 CL (3.18) | CL (seed 13) vs CO (oblique 0.8, lr 1e-4) | **CL final: 2.92 laps/ep, 64 % 2+ passes, 52 % oblique -> 3.43**; CO best 2.76 | **CL** again (but corner entry fell 60 -> 52 %) |
| 14 | mg13 CL (3.43) | CL (seed 14) vs LZ (clearance reward, lr 2e-5) | running (first round with the low-lr pool) | – |

**Why CL matters (2026-10-23).** A continued run trains at lr 1e-4 by default (`train_onboard.py`). Every
round-2 idea at that rate came out *worse* than its parent: fine-tuning a good policy that hard knocks it
off its optimum before the new idea can help. At 2e-5 the same C9 recipe improved it (2.17 -> 2.58 laps/ep).
So from round 14 the optimiser retries the ideas at the gentle rate (`L*` keys in `week10_experiments.py`),
corner ideas first (LZ clearance reward, LE stress drone). The pool can now be changed without a restart:
write `["LZ", ...]` to `results/hoop/mg_loop_pool.json`. Fly this one first:
`hardware/policies/LOOP_20261024_w10-mg13-CL-loop-lowlr_final_3.03lpe.zip` (FLY_THIS_FIRST.txt; round 13).

**Hierarchical RL result (2026-10-23).** No gain over the flat policy: flat C9 2.75, learned manager (best)
2.74, fixed switch 2.51 (`results/hoop/hrl_compare.md`). The manager learned to pick C9 about 90 % of the
time, i.e. that the flat generalist is already the best skill. The specialised skills were trained for
2 M steps each; the generalist had about 30 M. HRL is parked; it would need skills trained as long as C9.
