# Hand-over: training the hoop-looping Tello policy together

Written 2026-10-15 for everyone helping to train. Read this first, then the illustrated version in
[docs/Onboard Hoop Policy Anatomy.html](docs/Onboard%20Hoop%20Policy%20Anatomy.html) (§18–22 cover
the last two days).

## 1. Where we are

**Goal.** A Tello flies through a 30 cm hoop using only its own camera, then loops round the outside
and re-enters, as many times as possible. We train in our simulator (`hardware/onboard/sim.py`) and
fly the result on the real drone. Score = **laps per episode** (passes in a 90 s episode, same-side
laps, starts from all 9 zones of the cage).

**Best policies so far** (sim scores from `hardware/onboard/export_policy.py`; full ranking in
[hardware/policies/README.md](hardware/policies/README.md)):

| Policy | Laps / 90 s | Episodes with 2+ passes | Oblique entry* | Notes |
|---|---|---|---|---|
| `w10_n9_O_oblique` (best) | **1.58** | **53 %** | 67 % | **current best**; trained with slanted starts |
| `w10_n9_OZ_oblique_clearance` (best) | 0.94 | 19 % | 57 % | slanted starts + line-up reward |
| `n8_dapg_nodrain_cont2` (best) | 1.03 | 22 % | 70 % | **the one flown on the real drone** |
| best policy *without* memory | 0.78 | 0 % | – | never loops |

\* Oblique entry = share of 30 slanted starts near the hoop (35–80° off its axis, like the cage
corners) that end in a clean pass.

**The open problem: corners.** O loops much more often than the flown policy, but its oblique
entry (67 %) is no better than the flown policy's 70 %. Slanted starts taught more loops, not cleaner
slanted entries. That is why experiments (b) and (e) below matter most.

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
| d | **Second DAgger pass with memory** | the teacher relabelling O's own states (DAgger) then DAPG again lifts the loop rate, as `dagger_n1` → n8 did | `hardware/privileged/dagger_hist.py` from O, then DAPG as in `_N8` | |
| e | **Sim-to-real: corners** | real drift and command delay make slanted entries fail; more delay/drift randomisation in training makes the policy line up earlier | raise the delay/drift ranges in `sim.py` profiles (`sample_profile`), then `--keys O` | |

To add an experiment, add a line to `EXPERIMENTS` in `week10_experiments.py` (copy `"O"` and change
the flags), and use its key with `--keys`.

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
