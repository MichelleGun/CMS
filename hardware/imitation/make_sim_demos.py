"""
Generate demonstrations in the simulator with an expert, so the whole BC pipeline can be tested
(and a useful warm start produced) before any hardware flight.

    python hardware/imitation/make_sim_demos.py --expert scripted --episodes 40
    python hardware/imitation/make_sim_demos.py --expert model --run-dir results/hoop/runs/<good run> --episodes 40

`scripted`: fly straight at the hoop at full speed (what a human on the keyboard would do).
`model`   : a trained continuous-action policy (PPO/A2C/SAC/TD3; not DQN).
Needs pybullet + gym_pybullet_drones (the `drones` env).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
for p in (HW, HW.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from imitation.record_demo import DEMO_DIR, DemoRecorder  # noqa: E402


def scripted_action(obs, noise=0.0, rng=None):
    rel = obs[0, 72:75]
    a = np.array([*rel, 1.0], dtype=np.float32)
    if noise and rng is not None:
        a[:3] += rng.normal(0, noise, 3)
    return a


def make_demos(episodes=20, expert="scripted", model=None, seed=0, noise=0.0, out=None, verbose=True):
    from hoop_env import HoopFindAviary
    env = HoopFindAviary(gui=False)
    rng = np.random.default_rng(seed)
    rec = DemoRecorder({"source": f"sim_{expert}", "episodes": episodes})
    reached = 0
    try:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            t = 0.0
            while True:
                if expert == "scripted":
                    act = scripted_action(obs, noise, rng)
                else:
                    act = np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)[:4]
                rec.add(obs, act, act, t)
                obs, _r, te, tr, _ = env.step(np.asarray(act, dtype=np.float32).reshape(1, 4))
                t += env.unwrapped.CTRL_TIMESTEP
                if te or tr:
                    reached += int(te)
                    break
            rec.next_episode()
    finally:
        env.close()
    rec.meta["reached"] = reached
    path = rec.save(out)
    if verbose:
        print(f"[make_sim_demos] {len(rec)} samples, {reached}/{episodes} episodes reached the hoop -> {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", choices=["scripted", "model"], default="scripted")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--variant", choices=["best", "final"], default="best")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--noise", type=float, default=0.0, help="direction noise on the scripted expert")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    model = None
    if a.expert == "model":
        if not a.run_dir:
            sys.exit("--run-dir is required with --expert model")
        from stable_baselines3 import A2C, PPO, SAC, TD3
        meta = json.loads((a.run_dir / "meta.json").read_text())
        if meta.get("env_variant") == "discrete_vel":
            sys.exit("discrete (DQN) policies are not supported as demo experts")
        cls = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3}[meta.get("algo", "ppo")]
        model = cls.load(str(a.run_dir / f"{a.variant}_model.zip"))
    make_demos(a.episodes, a.expert, model, a.seed, a.noise, a.out)


if __name__ == "__main__":
    main()
