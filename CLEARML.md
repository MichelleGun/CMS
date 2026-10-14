# ClearML: tracking every run and flight

ClearML (https://app.clear.ml) is a website that keeps a record of every training run and Tello
flight: settings, reward curves, model files and logs. This project syncs to it automatically, so
you can compare runs without digging through folders.

## What gets synced

| ClearML project | What is in it | Synced when |
|---|---|---|
| `Tello/onboard-rl` | Camera policies (`hardware/onboard/train_onboard.py`, dashboard Training with "Onboard camera") | live, from start to stop |
| `Tello/pybullet` | PyBullet state policies (`scripts/train_hoop_multi.py`, dashboard Training) | live, from start to stop |
| `Tello/clones` | Behaviour clones (`hardware/imitation/bc_train.py`, dashboard Imitation) | when the clone is saved |
| `Tello/flights` | Real and dry-run Tello flights (`hardware/onboard/fly_hoop.py`, dashboard Drone tab) | after landing |

Flights upload only after the drone has landed, so nothing slows the 15 Hz flight loop.
Runs started from the dashboard sync too, because the dashboard starts the same scripts.

The code is in `scripts/clearml_sync.py` (shared helper) and `scripts/clearml_upload_runs.py` (upload of
older runs).

## One-time setup

Use the `drones` conda environment (SETUP.md). In PowerShell, from the `CMS` folder:

```powershell
$PY = "C:/Users/miche/miniconda3/envs/drones/python.exe"
& $PY -m pip install clearml            # also in webapp/requirements.txt
```

1. Create a free account at https://app.clear.ml.
2. In the web app click your avatar (top right), then **Settings → Workspace → Create new credentials**.
   Copy the block it shows.
3. Run the setup and paste that block when asked:
   ```powershell
   & $PY -m clearml.cli.config           # same as `clearml-init`, but uses the drones environment
   ```
   This writes `clearml.conf` into your user folder (`C:\Users\<you>\clearml.conf`). The credentials
   are not stored in this repo.

You do **not** add any ClearML code (`Task.init(...)`) to your own scripts: it is already in the
trainers and the flight script.

## Upload the runs you already have

```powershell
& $PY scripts/clearml_upload_runs.py --dry-run      # lists what would be uploaded; uploads nothing
& $PY scripts/clearml_upload_runs.py                # uploads everything (about 170 folders, ~160 MB)
& $PY scripts/clearml_upload_runs.py --only onboard-rl clones     # just some projects
```

Safe to re-run: a run already in ClearML is skipped (its `meta.json` holds a `clearml_task_id`).
If the dry run prints `ClearML: OFF - no ClearML credentials`, the setup step did not take.

## Where to look

In the web app, **Projects → Tello**, then open a project and click a run.

| Tab | What you find |
|---|---|
| **Configuration → Hyperparameters** | `run` (learning rate, envs, parent run), `reward` (every reward weight), `robust` (randomisation), `bc` (demos and cloning strength), `architecture` (PPO settings, layer shapes, activation, learned noise `log_std_values`) |
| **Configuration → meta.json** | The run's whole `meta.json` |
| **Scalars** | `eval/*` (reward, success rate), `passes/*` and `passes_by_zone/*`, and for live runs `train/*` (losses, entropy, `std`), `rollout/*`, `bc/*`. For flights: `ring/*` (range, cx, cy, axis_ratio, found), `drone/*`, `policy_action/*` (the 4 policy outputs) and `rc_sent/*` (the sticks sent) |
| **Artifacts** | `best_model.zip`, `latest_model.zip`, `final_model.zip`, `meta.json`, `evaluations.npz`, `passes.json`, `train.log`. Flights: `ticks.jsonl`, `events.jsonl`, `summary.json`, `demo.npz` |

To compare runs: tick several in a project's list and click **Compare**. A flight carries a
`model:<run>` tag, so you can find which policy flew it.

## Turning it off

| How | Effect |
|---|---|
| `--no-clearml` on `train_onboard.py`, `train_hoop_multi.py`, `bc_train.py` or `fly_hoop.py` | that one run is not synced |
| environment variable `CLEARML_OFF=1` | nothing is synced |

If `clearml` is not installed, the credentials are missing or the internet is down, runs and
flights carry on normally and print a single line `[clearml] sync off: <reason>`.
Under pytest nothing is synced.

## Troubleshooting

| What you see | What it means |
|---|---|
| `[clearml] sync off: no ClearML credentials` | Run `& $PY -m clearml.cli.config` and paste the credentials. |
| `[clearml] sync off: clearml is not installed` | `& $PY -m pip install clearml` in the `drones` environment. |
| Project `Tello` is empty in the web app | Check you are in the same workspace as the credentials, then run the upload script. |
| `[clearml] ... failed (run continues)` | One upload or report failed. The run itself is unaffected; run `clearml_upload_runs.py` later to fill in what is missing. |
| A run started from the dashboard is not in ClearML | Check the run's `train.log` for the `[clearml]` line. |
