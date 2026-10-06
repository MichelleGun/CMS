"""
Behaviour cloning into a PPO/A2C policy that the existing sim tools can load.

    python hardware/imitation/bc_train.py --demos hardware/flights/demos/*.npz --out hardware/flights/bc/my_bc
    python hardware/imitation/bc_train.py --demos ... --warm-critic --eval-episodes 20     # needs pybullet

What it does, and why:
  * Builds PPO("MlpPolicy") with a dummy env whose spaces are exactly the sim's: obs (1, 75), action (1, 4).
    The saved zip therefore loads in replay_hoop.py / run_policy.py like any trained model.
  * Trains ONLY the actor path (mlp_extractor.policy_net + action_net). The sim's controller uses the
    action's direction (dims 0-2; magnitude is normalised away) and speed fraction |dim 3|, so:
      - direction: cosine loss against the demonstrated direction, only where the demo is moving
      - speed:     MSE on |dim 3|
      - a light penalty keeps |direction| near 1 so the policy mean stays in [-1, 1]
  * After BC, log_std is set small (-2): PPO's exploration noise would otherwise swamp a cloned policy.
  * --warm-critic regresses the value network on Monte-Carlo returns recomputed from the demos
    (reward is a function of the observation), because a random critic wrecks a BC actor when PPO
    fine-tuning starts. Fine-tune with a small learning rate regardless.
Caveat: keyboard demos are few-direction, near-constant-speed; BC reproduces that style.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
for p in (HW, HW.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from imitation.record_demo import load_demos, mc_returns  # noqa: E402


def make_dummy_env():
    import gymnasium as gym
    from gymnasium import spaces

    class SpaceEnv(gym.Env):
        """Only carries the spaces; never stepped."""
        observation_space = spaces.Box(-np.inf, np.inf, shape=(1, 75), dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, shape=(1, 4), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros((1, 75), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1, 75), dtype=np.float32), 0.0, True, False, {}
    return SpaceEnv()


def build_model(algo="ppo", seed=0, net=(64, 64)):
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    cls = {"ppo": PPO, "a2c": A2C}[algo]
    venv = DummyVecEnv([make_dummy_env])
    return cls("MlpPolicy", venv, seed=seed, device="cpu", policy_kwargs=dict(net_arch=list(net)), verbose=0)


def targets_from_demo(intent: np.ndarray):
    """(direction unit vectors, speed in [0,1], moving mask)."""
    d = intent[:, :3].astype(np.float64)
    n = np.linalg.norm(d, axis=1)
    speed = np.abs(intent[:, 3]).astype(np.float64)
    moving = (n > 1e-6) & (speed > 1e-6)
    unit = np.zeros_like(d)
    unit[moving] = d[moving] / n[moving, None]
    return unit.astype(np.float32), speed.astype(np.float32), moving


def train_actor(model, obs, intent, epochs=200, lr=3e-3, batch=256, val_frac=0.15, seed=0, verbose=True):
    import torch
    rng = np.random.default_rng(seed)
    unit, speed, moving = targets_from_demo(intent)
    n = len(obs)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac)) if n > 20 else 0
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    pol = model.policy
    params = list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    O = torch.as_tensor(obs.reshape(-1, 1, 75), dtype=torch.float32)
    U, S, M = torch.as_tensor(unit), torch.as_tensor(speed), torch.as_tensor(moving)

    def mean_action(o):
        return pol.get_distribution(o).distribution.mean.reshape(-1, 4)

    def losses(ids):
        mean = mean_action(O[ids])
        d, s = mean[:, :3], mean[:, 3]
        m = M[ids]
        if m.any():
            cos = torch.nn.functional.cosine_similarity(d[m], U[ids][m], dim=1, eps=1e-6)
            l_dir = (1 - cos).mean()
            l_mag = ((d[m].norm(dim=1) - 1.0) ** 2).mean()
        else:
            cos, l_dir, l_mag = torch.zeros(1), torch.zeros(()), torch.zeros(())
        l_spd = ((s - S[ids]) ** 2).mean()
        return l_dir, l_spd, l_mag, cos

    hist = []
    for ep in range(epochs):
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), batch):
            ids = torch.as_tensor(perm[i:i + batch])
            l_dir, l_spd, l_mag, _ = losses(ids)
            loss = l_dir + l_spd + 0.05 * l_mag
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            with torch.no_grad():
                ld, ls, _, c = losses(torch.as_tensor(val_idx if n_val else tr_idx))
            hist.append((ep, float(ld), float(ls), float(c.mean())))
            print(f"[bc] epoch {ep:4d}  val dir-loss {float(ld):.4f}  speed-mse {float(ls):.4f}  mean cos {float(c.mean()):.3f}")
    with torch.no_grad():
        ld, ls, _, c = losses(torch.as_tensor(val_idx if n_val else tr_idx))
        pol.log_std.data.fill_(-2.0)                          # small exploration after cloning
    return dict(val_dir_loss=float(ld), val_speed_mse=float(ls), val_mean_cos=float(c.mean()), history=hist,
                n_train=int(len(tr_idx)), n_val=int(n_val))


def warm_critic(model, obs, episode, epochs=100, lr=1e-3, batch=256, gamma=0.99, seed=0):
    import torch
    G = torch.as_tensor(mc_returns(obs, episode, gamma))
    G = (G - G.mean()) / (G.std() + 1e-6)                      # critic learns normalised returns shape; scale is relearned by PPO
    pol = model.policy
    params = list(pol.mlp_extractor.value_net.parameters()) + list(pol.value_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    O = torch.as_tensor(obs.reshape(-1, 1, 75), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    last = None
    for ep in range(epochs):
        perm = rng.permutation(len(obs))
        for i in range(0, len(perm), batch):
            ids = torch.as_tensor(perm[i:i + batch])
            v = pol.predict_values(O[ids]).reshape(-1)
            loss = ((v - G[ids]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss)
    return last


def evaluate_in_sim(model, episodes=20, seed=100):
    """Roll the cloned policy in HoopFindAviary: success rate and mean final distance. Needs pybullet."""
    from hoop_env import HoopFindAviary
    env = HoopFindAviary(gui=False)
    ok, dists = 0, []
    try:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            while True:
                a, _ = model.predict(obs, deterministic=True)
                obs, _r, te, tr, _ = env.step(np.asarray(a).reshape(1, 4))
                if te or tr:
                    break
            dists.append(float(np.linalg.norm(obs[0, 72:75])))
            ok += int(te)
    finally:
        env.close()
    return dict(success_rate=ok / episodes, mean_final_dist=float(np.mean(dists)), episodes=episodes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", nargs="+", required=True, help="demo .npz files (globs allowed)")
    ap.add_argument("--out", type=Path, required=True, help="output directory (gets bc_model.zip + meta.json)")
    ap.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min-speed", type=float, default=0.05, help="drop demo samples whose speed fraction is below this")
    ap.add_argument("--use", choices=["intent", "executed"], default="intent")
    ap.add_argument("--warm-critic", action="store_true")
    ap.add_argument("--eval-episodes", type=int, default=0, help="roll the result in the sim (needs pybullet)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--as-run", action="store_true",
                    help="also save as results/hoop/runs/bc_<ts>/ so replay_hoop.py / the dashboard can load it")
    a = ap.parse_args(argv)

    files = sorted({f for pat in a.demos for f in glob.glob(pat)})
    if not files:
        sys.exit(f"no demo files match {a.demos}")
    d = load_demos(files)
    keep = np.abs(d[a.use][:, 3]) >= a.min_speed
    keep |= np.linalg.norm(d[a.use][:, :3], axis=1) < 1e-6          # keep explicit hover samples too (action 0)
    obs, act, ep = d["obs"][keep], d[a.use][keep], d["episode"][keep]
    print(f"[bc] {len(files)} file(s), {len(d['obs'])} samples, {len(obs)} after filtering")

    model = build_model(a.algo, a.seed)
    stats = train_actor(model, obs, act, epochs=a.epochs, lr=a.lr, seed=a.seed)
    if a.warm_critic:
        stats["critic_loss"] = warm_critic(model, d["obs"], d["episode"], seed=a.seed)
        print(f"[bc] critic warmed up, final loss {stats['critic_loss']:.4f}")

    a.out.mkdir(parents=True, exist_ok=True)
    zip_path = a.out / "bc_model.zip"
    model.save(str(zip_path))
    meta = dict(algo=a.algo, env_variant="continuous_vel", source="bc", demos=[Path(f).name for f in files],
                n_samples=int(len(obs)), final_status="bc", saved=time.strftime("%Y-%m-%d %H:%M:%S"), **{
                    k: v for k, v in stats.items() if k != "history"})
    if a.eval_episodes:
        res = evaluate_in_sim(model, a.eval_episodes)
        meta["sim_eval"] = res
        print(f"[bc] sim rollout: success {res['success_rate']:.0%}, mean final distance {res['mean_final_dist']:.2f} m")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bc] saved {zip_path}")

    if a.as_run:
        run = HW.parent / "results" / "hoop" / "runs" / f"bc_{time.strftime('%Y%m%d-%H%M%S')}"
        run.mkdir(parents=True, exist_ok=True)
        model.save(str(run / "best_model.zip"))
        model.save(str(run / "final_model.zip"))
        (run / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[bc] also saved as sim run {run}")
    return meta


if __name__ == "__main__":
    main()
