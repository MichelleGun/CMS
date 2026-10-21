"""
Week 10 experiments: one PPO setting changed per run, everything else held equal, so the runs
can be compared (tutorial/week10_compare.ipynb offline, ClearML project Tello/onboard-rl online).

    A baseline   the settings every run so far used (SB3 defaults)
    B gamma      gamma 0.995, gae_lambda 0.97: the critic sees ~13 s ahead (a lap), not ~6.7 s
    C kl         target_kl 0.02, batch 512, 5 epochs: a trust-region brake on each update
    D entropy    ent_coef 0.005 and log_std reset to -1: exploration that cannot collapse
    E sde        gSDE: smooth, state-dependent exploration noise
    F lambda1    GAE lambda 1.0 (+ batch 512, 5 epochs): Monte Carlo advantages, so a wrong critic
                 can only add noise, never bias (Week 10 notes 5.7)

All start from the same parent, with the same demos, BC strength, seed, task and budget, on the
"measured" drone profile (onboard/sim.py: dynamics fitted to the real flight logs).

Default parent: onboard_20261011-032612. Since 2026-10-13 the policy gets no speed input (the
real Tello's vgx is direction-less jitter in the cage) and yaw follows the drone's clockwise
convention; in that corrected sim 032612 still passes 60 % on the measured profile, while
025518 / v4 / v5 collapse (0-7 %: they leaned on the speed input).

    # overnight: plug the laptop in (the launcher keeps it from sleeping), then
    python hardware/onboard/week10_experiments.py --stage front
    # only some, a second seed (folders get _s1), or print the commands without running them
    python hardware/onboard/week10_experiments.py --stage front --only A C --seed 1
    python hardware/onboard/week10_experiments.py --stage front --dry-run
    # stage 2: same-side laps from every start zone, from a stage-1 winner (or --then-laps)
    python hardware/onboard/week10_experiments.py --stage laps --parent results/hoop/runs/w10_front_C_kl/best_model.zip

Each run writes results/hoop/runs/w10_<stage>_<X>_<name>[_s<seed>]/ (train.log, progress.csv,
meta.json, evaluations.npz, passes.json, best/latest/final_model.zip). When the batch ends,
results/hoop/runs/w10_<stage>_summary.txt lists every run's status and passes. A run that is
already "completed" is skipped; one that crashed or was stopped is moved to
results/hoop/runs_archive/ and started again.

evaluate() and critic_check() below are what the notebook uses.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CMS = HERE.parents[1]
for p in (HERE.parent, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RUNS = CMS / "results" / "hoop" / "runs"
ARCHIVE = CMS / "results" / "hoop" / "runs_archive"
DEFAULT_PARENT = "results/hoop/runs/onboard_20261011-032612/best_model.zip"
DEMOS = ["hardware/flights/tello_*/demo.npz", "hardware/flights/demos/teacher_ppo_20261010-011316.npz"]

_C_TRK = ["--target-kl", "0.02", "--batch-size", "512", "--n-epochs", "5", "--ring-tracker"]
_NO_BC = ["--bc-coef", "0", "--bc-coef-min", "0"]
# the teacher's loop-leg labels only (41-wide, recorded live; every relabelling pass adds a file)
_LOOP_BC = ["--demos", "hardware/flights/demos/teacher_loop_*.npz", "--bc-coef", "0.2", "--bc-coef-min", "0.02",
            "--bc-decay-steps", "2000000"]
_REFIND = ["--reward-weights", '{"r_found": 3.0}']
# n8_dapg_nodrain's own settings (a memory policy: history stack + asymmetric critic, DAgger BC kept on)
_N8 = ["--ring-tracker", "--history-lags", "0,1,2,4,8,16", "--asym-critic",
       "--demos", "hardware/flights/demos/dagger_n1_r*.npz", "--bc-coef", "0.3", "--bc-coef-min", "0.3",
       "--lr", "5e-5", "--target-kl", "0.01", "--batch-size", "512", "--n-epochs", "5",
       "--critic-warmup-steps", "0", "--sway-deg", "2"]
_MG_DEMOS = ["--demos", "hardware/flights/demos/dagger_n1_r*.npz", "hardware/flights/demos/dagger_mg_r*.npz"]
_N8_RW = '{"r_wrong_way": -20, "drain": false, "r_struck": -10, "w_dwell": 0.05}'
# + line up on the axis before going in, and keep clear of the noodle
_N8_CLEAR_RW = ('{"r_wrong_way": -20, "drain": false, "r_struck": -15, "w_dwell": 0.05, '
                '"w_rim": 1.0, "rim_margin_m": 0.15, "axis_pull": 2.0}')

EXPERIMENTS = {
    "A": ("baseline", []),
    "B": ("gamma", ["--gamma", "0.995", "--gae-lambda", "0.97"]),
    "C": ("kl", ["--target-kl", "0.02", "--batch-size", "512", "--n-epochs", "5"]),
    "D": ("entropy", ["--ent-coef", "0.005", "--log-std-reset", "-1"]),
    "E": ("sde", ["--sde"]),
    "F": ("lambda1", ["--gae-lambda", "1.0", "--batch-size", "512", "--n-epochs", "5"]),
    # round 2 (2026-10-13): C won by a clear margin, so the small B and F gains are tried ON TOP of C
    # (they can't both go in: they set lambda differently). Raced against C by w10_race.py.
    "G": ("kl_gamma", ["--target-kl", "0.02", "--batch-size", "512", "--n-epochs", "5",
                       "--gamma", "0.995", "--gae-lambda", "0.97"]),
    "H": ("kl_lambda1", ["--target-kl", "0.02", "--batch-size", "512", "--n-epochs", "5",
                         "--gae-lambda", "1.0"]),
    # C + the ring tracker (4 extra inputs: where the ring is while it is out of view, obs.RingTracker).
    # A 37-input parent is widened with zero weights, so it starts out flying exactly as C.
    "T": ("kl_tracker", ["--target-kl", "0.02", "--batch-size", "512", "--n-epochs", "5",
                         "--ring-tracker"]),
    # round 3 (2026-10-13 night, loop_night.py): what makes T keep its laps instead of decaying?
    "U": ("trk_gamma", _C_TRK + ["--gamma", "0.995", "--gae-lambda", "0.97"]),       # longer horizon
    "V": ("trk_streak", _C_TRK + ["--streak", "1.0", "--streak-cap", "4"]),           # re-entry pays more
    "W": ("trk_pure", _C_TRK + ["--bc-coef", "0", "--bc-coef-min", "0"]),             # no pull to the demos
    "X": ("trk_combo", _C_TRK + ["--gamma", "0.995", "--gae-lambda", "0.97", "--streak", "1.0",
                                 "--streak-cap", "4", "--bc-coef", "0", "--bc-coef-min", "0"]),
    # X's settings, but the demo term only sees the human flight(s) that really looped (--demos wins:
    # argparse keeps the last value)
    # round 4 (2026-10-14): W (no BC) won, but never re-enters: after its pass it hits the ring or
    # the net on the way round. Continue W with near-pass starts (loop practice) vs plain W.
    "P": ("pure_more", _C_TRK + ["--bc-coef", "0", "--bc-coef-min", "0"]),
    "N": ("np50", _C_TRK + ["--bc-coef", "0", "--bc-coef-min", "0", "--near-pass-p", "0.5"]),
    "M": ("np80", _C_TRK + ["--bc-coef", "0", "--bc-coef-min", "0", "--near-pass-p", "0.8"]),
    # round 5 (2026-10-14): P passes in ~2/3 of episodes, but after the pass it sees the ring on only
    # 0-3 % of ticks: it never turns back to find it. L: the privileged teacher's labels on P's own
    # loop-leg states (privileged/loop_labels.py) as a small decaying BC term inside PPO;
    # R: re-finding the ring after a pass pays 3 (was 0.5); Q: both; K: P continued (control).
    "K": ("pure_cont", _C_TRK + _NO_BC),
    "L": ("teacher_loop", _C_TRK + _LOOP_BC),
    "R": ("refind", _C_TRK + _NO_BC + _REFIND),
    "Q": ("teacher_loop_refind", _C_TRK + _LOOP_BC + _REFIND),
    # n9 (2026-10-14 night), after the first real flights of the memory policy (n8): from three corners
    # of the cage it found the hoop but clipped its edge going in. The sim scored every zone well, so
    # this is a sim-to-real gap at slanted approaches. All continue n8 with its own settings (memory,
    # asymmetric critic, DAgger BC 0.3); O: 40 % slanted starts; Z: line up first / keep clear of the
    # noodle; OZ: both; K9: n8 continued (control).
    "K9": ("n8_cont", _N8 + ["--reward-weights", _N8_RW]),
    "O": ("oblique", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.4"]),
    "Z": ("clearance", _N8 + ["--reward-weights", _N8_CLEAR_RW]),
    "OZ": ("oblique_clearance", _N8 + ["--reward-weights", _N8_CLEAR_RW, "--oblique-p", "0.4"]),
    # TRAINING_SPEEDUPS.md: O plus one training change each, so each can be raced against O
    "OF": ("oblique_zonefocus", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.4",
                                       "--zone-focus", "2"]),
    "OS": ("oblique_stall30", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.4",
                                     "--stall-sec", "30"]),
    "ON": ("oblique_normreward", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.4",
                                        "--norm-reward"]),
    # 2026-10-17 continuous optimiser (loop_forever.py), all from the best policy so far (O line):
    # D9: DAPG with a fresh DAgger relabelling of the policy's OWN states (privileged/
    #     dagger_from_policy.py, files dagger_mg_r*.npz) pooled with n8's dagger_n1 labels;
    # E9: 30 % of episodes with a slower, driftier drone (sim.STRESS_*): the real corner clipping;
    # C9: more slanted starts (0.6) plus starts just before the hoop (0.3): more loop practice;
    # DE9: D9 + E9.
    "D9": ("dagger_relabel", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.4"] + _MG_DEMOS),
    "E9": ("stress", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.4", "--stress-p", "0.3"]),
    "C9": ("loop_practice", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.6", "--near-pass-p", "0.3"]),
    "DE9": ("dagger_stress", _N8 + ["--reward-weights", _N8_RW, "--oblique-p", "0.4", "--stress-p", "0.3"]
            + _MG_DEMOS),
    "Y": ("trk_loopdemo", _C_TRK + ["--gamma", "0.995", "--gae-lambda", "0.97", "--streak", "1.0",
                                    "--streak-cap", "4", "--bc-coef", "0.1", "--bc-coef-min", "0.05"]),
}

ROUND1 = "ABCDEF"     # the default batch; round-2 keys (G, H) run via --only or w10_race.py

STAGES = {
    # one pass ends the episode; starts on the front runway; full randomisation from the start
    "front": ["--laps", "1", "--start-mode", "ring", "--dr", "1", "--dr-steps", "0",
              "--profile", "measured", "--n-eval-episodes", "30"],
    # the "over and over" task: endless same-side laps from all 9 start zones
    "laps": ["--laps", "0", "--lap-dir", "same", "--start-mode", "zones", "--dr", "1", "--dr-steps", "0",
             "--profile", "measured", "--n-eval-episodes", "36"],
}
STAGE_ENV = {"front": dict(laps=1, start_mode="ring", dr=1.0, profile="measured"),
             "laps": dict(laps=0, lap_dir="same", start_mode="zones", dr=1.0, profile="measured")}


def run_dir(stage: str, key: str, seed: int = 0) -> Path:
    return RUNS / (f"w10_{stage}_{key}_{EXPERIMENTS[key][0]}" + (f"_s{seed}" if seed else ""))


def command(stage, key, parent, timesteps, n_envs, seed, no_clearml):
    out = run_dir(stage, key, seed)
    cmd = [sys.executable, str(HERE / "train_onboard.py"),
           "--init", parent, "--demos", *DEMOS,
           # the parent already decayed to the BC floor; restarting at 1.0 would yank it back
           "--bc-coef", "0.1", "--bc-coef-min", "0.1",
           # every run re-fits its critic first (B changes what the values mean; the speed input
           # and the dynamics changed under every parent on 2026-10-13)
           "--critic-warmup-steps", "20000",
           "--log-interval", "2",                 # ~90 diagnostic points per 3M-step run
           "--timesteps", str(timesteps), "--n-envs", str(n_envs), "--seed", str(seed),
           "--run-id", out.name, "--out", str(out),
           *STAGES[stage], *EXPERIMENTS[key][1]]
    if no_clearml:
        cmd.append("--no-clearml")
    return cmd, out


def _status(out: Path):
    try:
        return json.loads((out / "meta.json").read_text()).get("final_status")
    except (OSError, ValueError):
        return None


def _keep_awake(on: bool):
    """Windows: stop the PC sleeping while runs are going (the screen may still turn off)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception as e:                    # never stop a night's training over this
        print(f"[WARN] could not change the sleep setting: {e}")


def summarize(stage: str, outs) -> str:
    lines = [f"Week 10 experiments, stage '{stage}', summary written {datetime.now():%Y-%m-%d %H:%M}", "",
             f"{'run':34s} {'status':10s} {'steps':>10s} {'best passes':>12s} {'last 5 evals':>13s}"]
    for out in outs:
        status = _status(out) or "missing"
        steps, best, last = 0, float("nan"), float("nan")
        try:
            passes = json.loads((out / "passes.json").read_text())
            m = np.array([p["mean"] for p in passes])
            steps = int(passes[-1]["t"])
            best = float(np.convolve(m, np.ones(5) / 5, mode="valid").max()) if len(m) >= 5 else float(m.max())
            last = float(m[-5:].mean())
        except (OSError, ValueError, IndexError, KeyError):
            pass
        lines.append(f"{out.name:34s} {status:10s} {steps:>10d} {best:>12.3f} {last:>13.3f}")
    lines += ["", "best = best 5-evaluation moving average; last = mean of the final 5 evaluations.",
              "Open tutorial/week10_compare.ipynb for the curves and diagnostics."]
    return "\n".join(lines)


def launch(stage="front", only=None, parent=DEFAULT_PARENT, timesteps=3_000_000, n_envs=8,
           parallel=2, seed=0, no_clearml=False, dry_run=False, then_laps=False):
    keys = [k for k in EXPERIMENTS if (k in only if only else k in ROUND1)]
    queue = [command(stage, k, parent, timesteps, n_envs, seed, no_clearml) for k in keys]
    if dry_run:
        for cmd, _ in queue:
            print(subprocess.list2cmdline(cmd))
        return
    outs = [out for _, out in queue]
    running = []
    _keep_awake(True)
    try:
        while queue or running:
            while queue and len(running) < parallel:
                cmd, out = queue.pop(0)
                if out.exists():
                    if _status(out) == "completed":
                        print(f"[SKIP] {out.name} already completed")
                        continue
                    ARCHIVE.mkdir(parents=True, exist_ok=True)
                    dest = ARCHIVE / f"{out.name}_{datetime.now():%Y%m%d-%H%M%S}"
                    shutil.move(str(out), str(dest))
                    print(f"[RERUN] {out.name} was '{_status(dest)}'; old folder kept in {dest}")
                out.mkdir(parents=True, exist_ok=True)
                log = open(out / "train.log", "w", encoding="utf-8")   # a file, not a PIPE: never stalls
                running.append((subprocess.Popen(cmd, cwd=CMS, stdout=log, stderr=subprocess.STDOUT), out, log))
                print(f"[START] {out.name}  {datetime.now():%H:%M}")
            for item in list(running):
                proc, out, log = item
                if proc.poll() is not None:
                    log.close()
                    running.remove(item)
                    print(f"[END] {out.name} exit {proc.returncode} ({_status(out)})  {datetime.now():%H:%M}")
            time.sleep(5)
        text = summarize(stage, outs)
        (RUNS / f"w10_{stage}_summary.txt").write_text(text, encoding="utf-8")
        print("\n" + text)
        if then_laps and stage == "front":
            done = [o for o in outs if _status(o) == "completed" and (o / "best_model.zip").exists()]
            if done:
                def last5(o):
                    try:
                        return float(np.mean([p["mean"] for p in json.loads((o / "passes.json").read_text())][-5:]))
                    except (OSError, ValueError):
                        return -1.0
                best = max(done, key=last5)
                print(f"[THEN] laps stage from {best.name} (last-5 mean {last5(best):.3f})")
                launch("laps", only, str((best / "best_model.zip").relative_to(CMS)), timesteps, n_envs,
                       parallel, seed, no_clearml)
    finally:
        _keep_awake(False)


def _load_model(model_zip, stage):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from onboard import spec
    from onboard.gym_env import OnboardHoopEnv
    from onboard.policy import convert_old_yaw
    from onboard.train_onboard import _widen_index
    from vec_util import load_for_resume
    kw = STAGE_ENV[stage]
    model, _ = load_for_resume(PPO, model_zip, DummyVecEnv([lambda: OnboardHoopEnv(**kw)]),
                               widen_at=_widen_index)
    meta_p = Path(model_zip).parent / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    if spec.old_yaw_convention(meta):
        convert_old_yaw(model)
    return model, meta, kw


def evaluate(model_zip, deterministic=True, episodes=30, stage="front", seed=900_000_000):
    """Mean passes per episode of one saved model, with (deterministic=False) or without its
    exploration noise, on fixed layouts. Lecture 3.1 (aliasing): if adding noise helps, the
    observation is hiding something the deterministic policy needs."""
    from onboard.gym_env import OnboardHoopEnv
    model, _, kw = _load_model(model_zip, stage)
    env = OnboardHoopEnv(seed=seed, **kw)
    env.set_episode_seed(seed)
    passes = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, info = False, {}
        while not done:
            act, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
        passes.append(info.get("passes", 0))
    return float(np.mean(passes))


def critic_check(model_zip, episodes=20, stage="front", seed=910_000_000):
    """How wrong is the critic? Fly the policy WITH its exploration noise (the policy the critic
    was trained to evaluate), record every state and reward, and compare the critic's V(s) with the
    discounted return G_t that actually followed (gamma from the run's meta.json).

    A single return is an unbiased but noisy sample of V(s) (Week 10 notes 6.1, 7.8): the spread of
    points around the diagonal is noise, a systematic tilt or offset is critic error. Ticks within
    1/(1-gamma) of a time-limit cut are dropped, because their returns were cut short, not ended.
    -> dict(v, g, explained_variance, bias, rmse, gamma, n)."""
    import torch
    from onboard.gym_env import OnboardHoopEnv
    model, meta, kw = _load_model(model_zip, stage)
    gamma = float((meta.get("ppo") or {}).get("gamma") or model.gamma)
    horizon = int(math.ceil(1.0 / (1.0 - gamma)))
    env = OnboardHoopEnv(seed=seed, **kw)
    env.set_episode_seed(seed)
    V, G = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        states, rewards = [], []
        done, truncated = False, False
        while not done:
            states.append(obs)
            act, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, _ = env.step(act)
            rewards.append(r)
            done, truncated = term or trunc, trunc and not term
        g, ret = 0.0, np.zeros(len(rewards))
        for i in range(len(rewards) - 1, -1, -1):
            g = rewards[i] + gamma * g
            ret[i] = g
        keep = len(rewards) - horizon if truncated else len(rewards)
        if keep <= 0:
            continue
        with torch.no_grad():
            v = model.policy.predict_values(model.policy.obs_to_tensor(np.array(states[:keep]))[0])
        V.append(v.numpy().reshape(-1))
        G.append(ret[:keep])
    v, g = (np.concatenate(V), np.concatenate(G)) if V else (np.zeros(0), np.zeros(0))
    ev = float(1 - np.var(g - v) / np.var(g)) if len(g) > 1 and np.var(g) > 0 else float("nan")
    return dict(v=v, g=g, explained_variance=ev, bias=float(np.mean(v - g)) if len(g) else float("nan"),
                rmse=float(np.sqrt(np.mean((v - g) ** 2))) if len(g) else float("nan"), gamma=gamma, n=len(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), default="front")
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), default=None)
    ap.add_argument("--parent", default=DEFAULT_PARENT)
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=2, help="runs at once (each uses --n-envs cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clearml", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--then-laps", action="store_true",
                    help="after stage front, start stage laps from the run with the best last-5 mean")
    a = ap.parse_args()
    launch(a.stage, a.only, a.parent, a.timesteps, a.n_envs, a.parallel, a.seed, a.no_clearml, a.dry_run,
           a.then_laps)


if __name__ == "__main__":
    main()
