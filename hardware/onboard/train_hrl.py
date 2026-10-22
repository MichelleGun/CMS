"""
Train the MANAGER of the hierarchy (hrl.py): a PPO policy over frozen skills (the options framework).

    python hardware/onboard/train_hrl.py --tag v1 --skills results/hoop/runs/w10_hrl_SE_skill_enter \
        results/hoop/runs/w10_hrl_SL_skill_loop results/hoop/runs/w10_mg6_C9_loop_practice
    python hardware/onboard/train_hrl.py --tag smoke --timesteps 2000 --eval-every 1000 --eval-episodes 2

What it does:
  1. scores the fixed-switch baseline (hrl.PhaseSwitch: the first skill until the first pass, then the
     second) on the same fixed layouts the manager is evaluated on, and writes it as a bundle
     (<run>/phase_bundle.json, flyable: policy.load_policy takes it);
  2. trains PPO on hrl.OptionEnv (Discrete action = which skill flies the next 0.53 s; reward = the
     env's reward summed over the option; gamma 0.92 per decision, about 0.99 per tick);
  3. evaluates every --eval-every manager steps on fixed layouts (laps per 90 s episode, per zone,
     and which skill flew before / after the first pass), keeping best_model.zip by laps/episode.
Run folder results/hoop/runs/hrl_<tag>/: meta.json (with "hrl": kind learned, skills), passes.json
(t = environment ticks, as for the flat runs), best/latest/final_model.zip (the manager), report.md.
Ends cleanly on stop.flag in the run folder.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from onboard.hrl import CMS, OptionEnv, hrl_meta, write_phase_bundle  # noqa: E402

RUNS = CMS / "results" / "hoop" / "runs"
EVAL_SEED = 950_000_000          # export_policy's layouts


def evaluate(skills, manager=None, episodes=36, seed=EVAL_SEED, phase=False):
    """Laps per episode of the manager (or the phase switch) on fixed layouts, plus skill usage."""
    env = OptionEnv(skills, seed=seed, phase=phase)
    env.set_episode_seed(seed)
    passes, zones = [], {}
    for _ in range(episodes):
        obs, _ = env.reset()
        done, info = False, {}
        while not done:
            if manager is not None:
                a, _ = manager.predict(obs, deterministic=True)
                a = int(np.asarray(a).reshape(-1)[0])
            else:
                a = 0
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        passes.append(int(info.get("passes", 0)))
        zones.setdefault(str(info.get("start_type", "?")).replace("zone_", ""), []).append(passes[-1])
    u = env.usage.astype(float)
    share = (u / np.maximum(u.sum(axis=1, keepdims=True), 1)).round(3).tolist()
    p = np.array(passes)
    return dict(mean=float(p.mean()), max=int(p.max()), two_plus=float((p >= 2).mean()),
                zones={k: round(float(np.mean(v)), 2) for k, v in sorted(zones.items())},
                usage_before_after_first_pass=share)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skills", nargs="+", required=True,
                    help="skill run folders or zips; the FIRST is the enter skill and the SECOND the loop "
                         "skill for the phase-switch baseline")
    ap.add_argument("--timesteps", type=int, default=300_000, help="manager decisions (x8 = env ticks)")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    ap.add_argument("--eval-every", type=int, default=6_000, help="manager decisions between evaluations")
    ap.add_argument("--eval-episodes", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    run = RUNS / f"hrl_{a.tag}"
    run.mkdir(parents=True, exist_ok=True)
    skills = [str(Path(s).as_posix()) for s in a.skills]
    meta = dict(hrl_meta("learned", skills), run_id=run.name, env="onboard", created_at=datetime.now().isoformat(
        timespec="seconds"), final_status=None, n_envs=a.n_envs, seed=a.seed, timesteps_requested=a.timesteps)
    (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log = open(run / "train.log", "a", encoding="utf-8")

    def say(msg):
        line = f"{datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    t0 = time.time()
    if len(skills) >= 2:
        base = evaluate(skills, phase=True, episodes=a.eval_episodes)
        write_phase_bundle(run / "phase_bundle.json", skills[:2])
        meta["phase_baseline"] = base
        (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        say(f"phase-switch baseline: {base['mean']:.2f} laps/ep, 2+ passes {base['two_plus']:.0%}, "
            f"usage {base['usage_before_after_first_pass']} ({time.time() - t0:.0f} s)")

    make = lambda i: (lambda: OptionEnv(skills, seed=a.seed * 1000 + i * 101))       # noqa: E731
    vec = (SubprocVecEnv if a.vec == "subproc" and a.n_envs > 1 else DummyVecEnv)([make(i) for i in range(a.n_envs)])
    model = PPO("MlpPolicy", vec, n_steps=256, batch_size=256, n_epochs=5, gamma=0.92, gae_lambda=0.95,
                ent_coef=0.01, learning_rate=3e-4, policy_kwargs=dict(net_arch=[64, 64]), seed=a.seed,
                device="cpu", verbose=0)
    history, best = [], -1.0

    class Eval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next = a.eval_every

        def _on_step(self):
            nonlocal best
            if (run / "stop.flag").exists():
                say("stop.flag: ending")
                return False
            if self.num_timesteps < self.next:
                return True
            self.next += a.eval_every
            self.model.save(str(run / "latest_model.zip"))
            r = evaluate(skills, self.model, episodes=a.eval_episodes)
            ticks = int(self.num_timesteps * 8)
            history.append(dict(t=ticks, decisions=int(self.num_timesteps), **r))
            (run / "passes.json").write_text(json.dumps(history), encoding="utf-8")
            if r["mean"] > best:
                best = r["mean"]
                self.model.save(str(run / "best_model.zip"))
            (run / "live_status.json").write_text(json.dumps(dict(
                status="training", current_timestep=ticks, latest_mean_passes=r["mean"],
                best_mean_passes=best)), encoding="utf-8")
            say(f"{self.num_timesteps} decisions: {r['mean']:.2f} laps/ep (best {best:.2f}), 2+ passes "
                f"{r['two_plus']:.0%}, usage before/after first pass {r['usage_before_after_first_pass']}")
            return True

    status = "completed"
    try:
        model.learn(total_timesteps=a.timesteps, callback=Eval())
        if (run / "stop.flag").exists():
            status = "stopped"
    except Exception as e:
        status = "crashed"
        say(f"crashed: {e!r}")
        raise
    finally:
        model.save(str(run / "final_model.zip"))
        meta.update(final_status=status, finished_at=datetime.now().isoformat(timespec="seconds"),
                    best_mean_passes=best)
        (run / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        vec.close()
        say(f"{status}; best {best:.2f} laps/ep; {run}")


if __name__ == "__main__":
    main()
