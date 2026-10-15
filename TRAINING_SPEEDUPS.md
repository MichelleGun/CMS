# Training speed-ups (2026-10-15)

Faster, steadier training for both trainers: the onboard-camera one (`hardware/onboard/train_onboard.py`,
`loop_night.py`) and the PyBullet one (`scripts/train_hoop_multi.py`). Some changes are on by default
and don't change what is learned. The rest are options you switch on per run, so you can race them
against O like any other experiment.

## 1. What it changes

**On by default (nothing to do):**

| Change | Why | Effect on results |
|---|---|---|
| **Parallel evaluation** (`--n-eval-envs`, default: `--n-envs` up to 4 with `--vec subproc`, 1 with `--vec dummy`) | Measured on the O setup: one 36-episode evaluation took **33 s on one core**; the 20k training steps between evaluations took ~22 s. Evaluation was **~60 % of the wall clock**, and more on a faster PC, because training gets faster with more envs but evaluation doesn't | Same layouts. Scores match up to float rounding (see section 4) |
| **Fixed evaluation layouts** (seed and start zone rewound before every evaluation) | Before, only the seed was rewound, so each evaluation paired the layouts with zones shifted by one | Evaluations within a run are exactly comparable |
| **Timed checkpoints** (`--checkpoint-min 15`, `--checkpoint-keep 5`) | `latest_model.zip` was only written at evaluations. Now it's also written every 15 min, plus `checkpoints/step_<n>.zip` (newest 5, each with its `meta.json` so the flight loader can read it) | A crash or sleep loses minutes, not the whole stretch since the last evaluation. `--resume-crashed` continues from it |
| **Exploration watch** (`--min-std-warn 0.05`) | Logs `explore/std` (ClearML, `progress.csv`) and warns once if the action noise collapses (v1 reached 0.06), writing `exploration_warning.txt` in the run folder | None |
| **PyBullet: tipping over is a failure** | It used to count as *truncated* (like a time limit) with no penalty, so PPO valued a crash as if the flight carried on. In budget mode it was cheaper than the net | Ends the episode with `tip_penalty` 5 (shaped) or `budget_tip` + the rest of the budget (budget mode). Rare with the velocity controller |

**Options (off unless you set them):**

| Option | Trainer | What it does |
|---|---|---|
| `--zone-focus 2` | both | Training starts favour the zones that fail: weight 1 + 2 × shortfall, so the worst zone comes up to 3× as often. No zone drops out. Evaluation still does all 9 equally |
| `--stall-sec 30` | onboard | Cuts a training episode after 30 s without a pass, as a time-limit style cut so PPO still bootstraps. A parking policy then wastes seconds, not 90 s, and gets more fresh starts |
| `--norm-reward` | both | Normalises training rewards (VecNormalize, reward only). Steadier critic; nothing to carry to replays or the drone |
| `--post-pass-p 0.3` | PyBullet | 30 % of training episodes start just past the ring: one pass counted, not re-armed. Practises the turn-around that v3–v5 rarely reached |
| `--budget-start 3 --budget-ramp-steps 2000000` | PyBullet | Budget mode: small budget first (more, shorter episodes), rising to `--budget`. Evaluation uses the full budget |
| `--action-repeat 2` | PyBullet | The policy decides at 15 Hz; the PID still runs at 30 Hz. Fresh runs get gamma 0.995² so the look-ahead stays the same in seconds |
| `--act-hist 4` | PyBullet, fresh runs | Only the last 4 actions in the observation: 84 → 40 inputs |
| `--sde`, `--batch-size` | PyBullet | gSDE exploration (fresh runs) and the PPO minibatch. The onboard trainer already has both |

**New tools:**

| Tool | What it's for |
|---|---|
| `hardware/onboard/rescore_checkpoints.py <run>` | Re-scores every saved policy of a run (best, latest, final, `checkpoints/`) on 100 fresh episodes through the flight loader, in parallel, with a ~95 % range. Writes `<run>/rescore.json` |
| `hardware/onboard/sweep_onboard.py` | Optuna sweep of PPO settings on top of an experiment (default O). Prunes losers early through `stop.flag`, resumable, writes `results/hoop/sweeps/<name>_best.md` |
| New experiment keys `OF`, `OS`, `ON` | O plus `--zone-focus 2`, `--stall-sec 30` or `--norm-reward`, ready for `loop_night.py --keys` |
| `loop_night.py --n-eval-envs N --extra-args "..."` | Sets the evaluation copies, or adds any trainer flags, for every run of the night |
| Dashboard Config card → **Speed** group | Evaluation copies, checkpoint interval, zone focus, stall cut and reward normalisation for onboard runs |

## 2. How to add it

You need the `CMS` repo on the latest `main` (with HANDOVER.md, commit `da9b5b8` or later) and the
patch file `training-speedups.patch`.

1. **Put the patch in the `CMS` folder**, open the Anaconda Prompt there and start a branch:
   ```
   cd <path to>\CMS
   git checkout main
   git pull
   git checkout -b speedups
   ```
2. **Apply it:**
   ```
   git am training-speedups.patch
   ```
   This makes one commit with all the changes. If git reports a conflict (someone changed the same
   lines since `da9b5b8`), run `git am --abort`, then `git apply --3way training-speedups.patch`,
   fix the files it marks, and `git commit -am "Training speed-ups"`.
3. **Install optuna** (only needed for the sweep):
   ```
   conda activate drones
   pip install optuna
   ```
4. **Run the tests:**
   ```
   python -m pytest hardware/onboard/tests scripts/tests -q
   python -m pytest webapp/backend/tests -q
   ```
   Expect the same known failures as before (`test_fly_hoop.py` / `test_human.py`, detector.json),
   and nothing else. The new tests are `test_speedups_onboard.py`, `test_rescore_sweep.py` and
   `scripts/tests/test_train_speedups.py`.
5. **Restart the dashboard** (`Ctrl+C`, then the `uvicorn` command again) and press **Ctrl+F5** in the
   browser.
6. **Push the branch** for Michelle to merge: `git push -u origin speedups`.

To undo, either `git checkout main`, or turn the defaults off per run: `--n-eval-envs 1 --checkpoint-min 0`.

## 3. Using it

**A night of experiments: O against the three new options**
```
python hardware/onboard/launch_detached.py results/hoop/loop_<yourname> -- hardware/onboard/loop_night.py --vec subproc --keys O OF OS ON --parent results/hoop/runs/w10_n9_O_oblique --tag <yourname> --timesteps 3000000 --n-envs 4 --parallel 1 --resume-crashed
```
With `--vec subproc --n-envs 4`, each run evaluates on 4 copies automatically.

**Memory.** Each evaluation copy is one more process, measured at ~0.5 GB (~0.3 GB of it its own).
With 4 training envs plus 4 evaluation copies, budget about 4 GB per run. If memory is tight, add
`--n-eval-envs 2`. `loop_night`'s default `--vec dummy` keeps evaluation on 1 copy, as before.

**Before you export or fly a run, re-score it**
```
python hardware/onboard/rescore_checkpoints.py results/hoop/runs/w10_<yourname>_O_oblique --episodes 100 --workers 4
```
If the top candidates' ranges overlap, they aren't really different: pick on oblique entry, or score
again with more episodes.

**A hyper-parameter sweep overnight**
```
pip install optuna
python hardware/onboard/launch_detached.py results/hoop/sweep_<yourname> -- hardware/onboard/sweep_onboard.py --name <yourname>_ppo --key O --trials 20 --timesteps 1000000 --n-envs 4 --vec subproc
```
Short trials favour settings that learn fast. Treat the winner (in `results/hoop/sweeps/<name>_best.md`)
as a candidate: add it as a new key in `week10_experiments.EXPERIMENTS` and race it full length.

**PyBullet: a fresh run with the learning changes**
```
python scripts/train_hoop_multi.py --algo ppo --until-stopped --n-envs 6 --reward budget --budget 10 --reward-weights "{\"budget_pass\": 5}" --budget-start 3 --post-pass-p 0.3 --zone-focus 2 --action-repeat 2 --act-hist 4 --sde --run-id ppo_speed --out results/hoop/runs/ppo_speed
```
Continuing an older PyBullet run keeps its action repeat and observation. `--sde` and `--act-hist`
need a fresh run.

## 4. What to know

- **Parallel evaluation flies the same layouts, but the scores aren't bit-identical.** With 2+ copies
  the network computes a batch of observations, which rounds about 1e-7 differently from one at a time,
  and a 90 s closed-loop flight amplifies that.
  - On the same 36 layouts, the best O model scored **1.25** laps/ep on 1 copy and **1.08** on 2, 3 or 4
    copies (those three agree exactly with each other).
  - That spread is the evaluation's own noise. Expect a ±0.15 laps/ep wobble between any two
    36-episode numbers, which is why `rescore_checkpoints.py` exists.
  - Within one run the number of copies is fixed, so its evaluations stay comparable.
- **Comparing with older runs.** Their evaluations were on 1 copy, with zones drifting by one each
  evaluation. A gap under ~0.2 laps/ep between an old curve and a new one isn't evidence either way.
  `loop_night`'s stop/extend rule compares against the old reference run T, so treat its borderline
  calls with care.
- **Evaluation cadence is unchanged** (every 20k steps in `loop_night`), so the race rules still mean
  what they did.
- **The camera-based PyBullet env from 2026-10-12** (`--env cam`) isn't part of this. The onboard
  simulator line (memory, asymmetric critic, DAgger, DAPG) has overtaken it.
