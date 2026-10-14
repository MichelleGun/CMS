"""
Hyper-parameter sweep (Optuna) on top of one week-10 experiment: each trial continues the parent with
that experiment's flags plus sampled PPO settings, for a short budget, and is scored on its last
evaluations. Hopeless trials are stopped early (stop.flag, so they still save cleanly).

    pip install optuna                                   # once, in the drones environment
    python hardware/onboard/sweep_onboard.py --name lr_ent --key O --trials 20 --timesteps 1000000 \\
        --parent results/hoop/runs/w10_n9_O_oblique --n-envs 4 --vec subproc
    python hardware/onboard/sweep_onboard.py --name lr_ent --dry-run        # print one trial's command

Start it detached for a night: hardware/onboard/launch_detached.py results/hoop/sweep_lr_ent -- \\
    hardware/onboard/sweep_onboard.py --name lr_ent ...

The study lives in results/hoop/sweeps/<name>.db, so stopping and starting again continues it (and
--parallel > 1 runs trials side by side on one study). Trial runs are results/hoop/runs/sweep_<name>_t<k>.
The best settings so far are written to results/hoop/sweeps/<name>_best.md as train_onboard.py flags;
add them as a new key in week10_experiments.EXPERIMENTS to race them properly against O.

Score = mean laps/ep over the trial's last 3 evaluations (one evaluation is noisy; see
TRAINING_SPEEDUPS.md). Short trials favour settings that learn fast, not necessarily ones that end
best: treat the winner as a candidate for a full-length run, not a result.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMS = HERE.parents[1]
for p in (HERE, HERE.parent, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RUNS = CMS / "results" / "hoop" / "runs"
SWEEPS = CMS / "results" / "hoop" / "sweeps"
EVAL_STEPS = 20_000          # as loop_night: an evaluation every 20k steps whatever the env count
LAST_N = 3

# name -> (kind, low, high[, choices]); "log" = log-uniform
SPACE = {
    "--lr": ("log", 1e-5, 2e-4),
    "--ent-coef": ("log", 1e-4, 1e-2),
    "--target-kl": ("float", 0.005, 0.03),
    "--gae-lambda": ("float", 0.90, 1.0),
    "--gamma": ("float", 0.98, 0.997),
    "--n-epochs": ("cat", None, None, [3, 5, 10]),
    "--batch-size": ("cat", None, None, [256, 512, 1024]),
}


def sample_flags(trial, space=SPACE) -> list:
    """Optuna trial -> train_onboard.py flags (later flags win over the experiment's own)."""
    out = []
    for flag, spec in space.items():
        name = flag.lstrip("-").replace("-", "_")
        kind = spec[0]
        if kind == "log":
            v = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            v = trial.suggest_float(name, spec[1], spec[2])
        else:
            v = trial.suggest_categorical(name, spec[3])
        out += [flag, f"{v:.6g}" if isinstance(v, float) else str(v)]
    return out


def trial_dir(name: str, number: int) -> Path:
    return RUNS / f"sweep_{name}_t{number:03d}"


def trial_command(key, parent, timesteps, n_envs, vec, out: Path, flags, no_clearml=True, seed=0,
                  eval_steps=EVAL_STEPS):
    from week10_experiments import command
    cmd, _ = command("laps", key, parent, timesteps, n_envs, seed, no_clearml)
    i = cmd.index("--out")
    cmd[i + 1] = str(out)
    j = cmd.index("--run-id")
    cmd[j + 1] = out.name
    return cmd + ["--vec", vec, "--eval-freq", str(max(1, eval_steps // n_envs))] + list(flags)


def score_of(run: Path, last_n=LAST_N):
    """(mean laps/ep over the last `last_n` evaluations, number of evaluations) or (None, 0)."""
    try:
        hist = json.loads((run / "passes.json").read_text())
    except (OSError, ValueError):
        return None, 0
    means = [h["mean"] for h in hist if h.get("mean") is not None]
    if not means:
        return None, 0
    return float(sum(means[-last_n:]) / len(means[-last_n:])), len(means)


def _extra(a):
    return ["--n-eval-episodes", str(a.n_eval_episodes)] if a.n_eval_episodes else []


def _run_trial(trial, a):
    import optuna
    flags = sample_flags(trial)
    out = trial_dir(a.name, trial.number)
    out.mkdir(parents=True, exist_ok=True)
    cmd = trial_command(a.key, a.parent, a.timesteps, a.n_envs, a.vec, out, flags + _extra(a), seed=a.seed,
                        eval_steps=a.eval_steps)
    trial.set_user_attr("flags", " ".join(flags))
    trial.set_user_attr("run", out.name)
    with open(out / "train.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(CMS))
        reported = 0
        while proc.poll() is None:
            time.sleep(a.check_s)
            s, n = score_of(out)
            if s is None or n <= reported:
                continue
            reported = n
            trial.report(s, n)
            if trial.should_prune():
                (out / "stop.flag").write_text("pruned by sweep_onboard.py\n")
                proc.wait()
                raise optuna.TrialPruned(f"{s:.3f} laps/ep after {n} evaluations")
    s, n = score_of(out)
    if s is None:
        raise RuntimeError(f"trial {trial.number} produced no evaluation (see {out / 'train.log'})")
    return s


def write_best(study, name):
    try:
        best = study.best_trial
    except ValueError:
        return
    SWEEPS.mkdir(parents=True, exist_ok=True)
    (SWEEPS / f"{name}_best.md").write_text(
        f"# Sweep {name}: best so far\n\nTrial {best.number} ({best.user_attrs.get('run')}): "
        f"{best.value:.3f} laps/ep (mean of the last {LAST_N} evaluations)\n\n"
        f"Flags to add to the experiment:\n\n    {best.user_attrs.get('flags')}\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="study name (results/hoop/sweeps/<name>.db)")
    ap.add_argument("--key", default="O", help="week10_experiments key whose flags every trial uses")
    ap.add_argument("--parent", default="results/hoop/runs/w10_n9_O_oblique")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    ap.add_argument("--parallel", type=int, default=1, help="trials at once (memory: ~0.5 GB per env)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-s", type=float, default=60.0, help="seconds between progress checks")
    ap.add_argument("--eval-steps", type=int, default=EVAL_STEPS,
                    help="timesteps between evaluations (as loop_night: 20k)")
    ap.add_argument("--n-eval-episodes", type=int, default=None,
                    help="evaluation episodes per trial (default: the laps stage's 36)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.dry_run:
        class _Fixed:                       # mid-range values, to show the command
            number = 0

            def suggest_float(self, n, lo, hi, log=False):
                return (lo * hi) ** 0.5 if log else (lo + hi) / 2

            def suggest_categorical(self, n, ch):
                return ch[len(ch) // 2]
        flags = sample_flags(_Fixed()) + _extra(a)
        print(subprocess.list2cmdline(trial_command(a.key, a.parent, a.timesteps, a.n_envs, a.vec,
                                                    trial_dir(a.name, 0), flags, seed=a.seed)))
        return
    try:
        import optuna
    except ImportError:
        sys.exit("[ERROR] optuna is not installed: pip install optuna (in the drones environment)")
    SWEEPS.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=a.name, storage=f"sqlite:///{(SWEEPS / (a.name + '.db')).as_posix()}",
        load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=a.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=10))
    study.optimize(lambda t: _run_trial(t, a), n_trials=a.trials, n_jobs=max(1, a.parallel),
                   callbacks=[lambda s, _t: write_best(s, a.name)], catch=(RuntimeError,))
    write_best(study, a.name)
    print((SWEEPS / f"{a.name}_best.md").read_text() if (SWEEPS / f"{a.name}_best.md").exists()
          else "no completed trial")


if __name__ == "__main__":
    main()
