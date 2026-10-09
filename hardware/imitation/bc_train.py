"""
Behaviour cloning into a PPO/A2C policy that the existing sim tools can load.

    python hardware/imitation/bc_train.py --demos hardware/flights/demos/*.npz --out hardware/flights/bc/my_bc
    python hardware/imitation/bc_train.py --demos ... --warm-critic --eval-episodes 20     # needs pybullet

What it does, and why:
  * Builds PPO("MlpPolicy") with a dummy env whose spaces match the demonstrated task.
    The saved zip therefore loads in replay_hoop.py / run_policy.py (simulator task) or
    onboard/policy.py (onboard-camera task) like any trained model.
  * Trains ONLY the actor path (mlp_extractor.policy_net + action_net).
  * After BC, log_std is set small (-2): PPO's exploration noise would otherwise swamp a cloned policy.
  * --warm-critic regresses the value network on Monte-Carlo returns recomputed from the demos
    (reward is a function of the observation), because a random critic wrecks a BC actor when PPO
    fine-tuning starts. Fine-tune with a small learning rate regardless.
Caveat: keyboard demos are few-direction, near-constant-speed; BC reproduces that style.

TWO TASKS, TWO LOSSES. The action means different things in each, so cloning it with one
loss would be wrong for the other:
  * sim_hoop ("sim_vel" loss) -- action is [dx, dy, dz, speed_fraction]. The simulator's
    controller normalises the direction away and ignores the speed's sign, so only the
    DIRECTION of dims 0-2 and the MAGNITUDE of dim 3 carry information. Hence a cosine
    loss on the direction (applied only where the demo is moving), MSE on |dim 3|, and a
    light penalty keeping |direction| near 1 so the policy mean stays in [-1, 1].
  * onboard_hoop ("mse" loss) -- action is four independent rc channels
    [lr, fb, ud, yaw]. Every channel's sign and magnitude matters and a zero action is a
    meaningful command (hover), so it is plain MSE over all four with no direction
    normalisation and no moving-only mask.
The loss defaults from the demo file's meta["task"], so existing simulator demos train
exactly as they did before this was generalised.
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

from imitation.record_demo import SIM_OBS_DIM, SIM_TASK, load_demos, mc_returns  # noqa: E402

SIM_OBS_SHAPE = (1, SIM_OBS_DIM)
SIM_ACT_SHAPE = (1, 4)


def make_dummy_env(obs_shape=SIM_OBS_SHAPE, act_shape=SIM_ACT_SHAPE):
    import gymnasium as gym
    from gymnasium import spaces

    obs_shape, act_shape = tuple(obs_shape), tuple(act_shape)

    class SpaceEnv(gym.Env):
        """Only carries the spaces; never stepped."""
        observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Box(-1.0, 1.0, shape=act_shape, dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(obs_shape, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(obs_shape, dtype=np.float32), 0.0, True, False, {}
    return SpaceEnv()


def build_model(algo="ppo", seed=0, net=(64, 64), obs_shape=SIM_OBS_SHAPE,
                act_shape=SIM_ACT_SHAPE):
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    cls = {"ppo": PPO, "a2c": A2C}[algo]
    venv = DummyVecEnv([lambda: make_dummy_env(obs_shape, act_shape)])
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


def train_actor(model, obs, intent, epochs=200, lr=3e-3, batch=256, val_frac=0.15, seed=0,
                verbose=True, loss="sim_vel", obs_shape=SIM_OBS_SHAPE):
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
    O = torch.as_tensor(obs.reshape(-1, *tuple(obs_shape)), dtype=torch.float32)
    U, S, M = torch.as_tensor(unit), torch.as_tensor(speed), torch.as_tensor(moving)
    A = torch.as_tensor(np.asarray(intent, dtype=np.float32))

    def mean_action(o):
        return pol.get_distribution(o).distribution.mean.reshape(-1, 4)

    def losses(ids):
        """-> (primary, secondary, regulariser, agreement). The three returned scalars
        are summed into the training loss; `agreement` is only reported."""
        mean = mean_action(O[ids])
        if loss == "mse":
            # Four independent rc channels: every sign and magnitude matters, and a zero
            # action is a real command, so there is nothing to mask or normalise.
            err = ((mean - A[ids]) ** 2).mean(dim=0)
            l_all = err.mean()
            cos = torch.nn.functional.cosine_similarity(
                mean, A[ids], dim=1, eps=1e-6)
            return l_all, torch.zeros((), device=l_all.device), \
                torch.zeros((), device=l_all.device), cos
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

    label = "action-mse" if loss == "mse" else "dir-loss"
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), batch):
            ids = torch.as_tensor(perm[i:i + batch])
            l_dir, l_spd, l_mag, _ = losses(ids)
            loss_val = l_dir + l_spd + 0.05 * l_mag
            opt.zero_grad()
            loss_val.backward()
            opt.step()
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            with torch.no_grad():
                ld, ls, _, c = losses(torch.as_tensor(val_idx if n_val else tr_idx))
            hist.append((ep, float(ld), float(ls), float(c.mean())))
            print(f"[bc] epoch {ep:4d}  val {label} {float(ld):.4f}  "
                  f"speed-mse {float(ls):.4f}  mean cos {float(c.mean()):.3f}")
    with torch.no_grad():
        ld, ls, _, c = losses(torch.as_tensor(val_idx if n_val else tr_idx))
        pol.log_std.data.fill_(-2.0)                          # small exploration after cloning
    return dict(val_dir_loss=float(ld), val_speed_mse=float(ls), val_mean_cos=float(c.mean()), history=hist,
                n_train=int(len(tr_idx)), n_val=int(n_val), loss=loss)


def warm_critic(model, obs, episode, epochs=100, lr=1e-3, batch=256, gamma=0.99, seed=0,
                obs_shape=SIM_OBS_SHAPE):
    import torch
    G = torch.as_tensor(mc_returns(obs, episode, gamma))
    G = (G - G.mean()) / (G.std() + 1e-6)                      # critic learns normalised returns shape; scale is relearned by PPO
    pol = model.policy
    params = list(pol.mlp_extractor.value_net.parameters()) + list(pol.value_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    O = torch.as_tensor(obs.reshape(-1, *tuple(obs_shape)), dtype=torch.float32)
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


def evaluate_onboard(model, episodes=20, seed=100, profile="typical"):
    """Roll the clone in the onboard-camera sim: pass rate is the number that matters."""
    from onboard.gym_env import rollout
    import numpy as _np

    def policy_fn(obs):
        a, _ = model.predict(_np.asarray(obs, dtype=_np.float32).reshape(-1),
                             deterministic=True)
        return _np.asarray(a, dtype=float).reshape(-1)[:4]
    return rollout(policy_fn, episodes=episodes, seed=seed, profile=profile)


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
    ap.add_argument("--loss", choices=["auto", "sim_vel", "mse"], default="auto",
                    help="auto picks sim_vel for simulator demos and mse for onboard-camera demos")
    ap.add_argument("--as-run", action="store_true",
                    help="also save as results/hoop/runs/bc_<ts>/ so replay_hoop.py / the dashboard can load it")
    a = ap.parse_args(argv)

    files = sorted({f for pat in a.demos for f in glob.glob(pat)})
    if not files:
        sys.exit(f"no demo files match {a.demos}")
    d = load_demos(files)

    # Which task do these demos describe? Older simulator demos have no meta["task"], so
    # the observation width decides, and the explicit meta wins when present.
    metas = d.get("meta") or [{}]
    task = next((m.get("task") for m in metas if isinstance(m, dict) and m.get("task")), None)
    obs_width = int(d["obs"].shape[1])
    if task is None:
        task = SIM_TASK if obs_width == SIM_OBS_DIM else "onboard_hoop"
    if len({m.get("task", task) for m in metas if isinstance(m, dict)}) > 1:
        sys.exit("[bc] demo files are from different tasks; train them separately")
    shape = next((m.get("obs_shape") for m in metas
                  if isinstance(m, dict) and m.get("obs_shape")), None)
    obs_shape = tuple(shape) if shape else (
        SIM_OBS_SHAPE if task == SIM_TASK else (obs_width,))
    act_shape = SIM_ACT_SHAPE if task == SIM_TASK else (4,)
    loss = a.loss if a.loss != "auto" else ("sim_vel" if task == SIM_TASK else "mse")
    print(f"[bc] task '{task}', obs {obs_shape}, action {act_shape}, loss '{loss}'")

    if loss == "sim_vel":
        keep = np.abs(d[a.use][:, 3]) >= a.min_speed
        keep |= np.linalg.norm(d[a.use][:, :3], axis=1) < 1e-6      # keep explicit hover samples too (action 0)
    else:
        # dim 3 is yaw here, not a speed fraction, so the speed filter would discard
        # perfectly good samples (and keep ones chosen by an unrelated criterion).
        keep = np.ones(len(d["obs"]), dtype=bool)
    obs, act, ep = d["obs"][keep], d[a.use][keep], d["episode"][keep]
    print(f"[bc] {len(files)} file(s), {len(d['obs'])} samples, {len(obs)} after filtering")

    model = build_model(a.algo, a.seed, obs_shape=obs_shape, act_shape=act_shape)
    stats = train_actor(model, obs, act, epochs=a.epochs, lr=a.lr, seed=a.seed,
                        loss=loss, obs_shape=obs_shape)
    if a.warm_critic:
        if task != SIM_TASK:
            sys.exit("[bc] --warm-critic needs the simulator's reward (mc_returns reads "
                     "hoop - position from the observation), which the onboard-camera "
                     "task has no equivalent of. Train its critic with RL instead.")
        stats["critic_loss"] = warm_critic(model, d["obs"], d["episode"], seed=a.seed,
                                           obs_shape=obs_shape)
        print(f"[bc] critic warmed up, final loss {stats['critic_loss']:.4f}")

    a.out.mkdir(parents=True, exist_ok=True)
    zip_path = a.out / "bc_model.zip"
    model.save(str(zip_path))
    meta = dict(algo=a.algo, env_variant="continuous_vel", source="bc", demos=[Path(f).name for f in files],
                task=task, obs_shape=list(obs_shape),
                n_samples=int(len(obs)), final_status="bc", saved=time.strftime("%Y-%m-%d %H:%M:%S"), **{
                    k: v for k, v in stats.items() if k != "history"})
    if a.eval_episodes:
        if task == SIM_TASK:
            res = evaluate_in_sim(model, a.eval_episodes)
            meta["sim_eval"] = res
            print(f"[bc] sim rollout: success {res['success_rate']:.0%}, mean final distance {res['mean_final_dist']:.2f} m")
        else:
            res = evaluate_onboard(model, a.eval_episodes)
            meta["sim_eval"] = res
            print(f"[bc] onboard sim rollout: passed {res['pass_rate']:.0%} "
                  f"({res['passed']}/{res['episodes']}), struck {res['struck']}, "
                  f"missed {res['missed']}, mean reward {res['mean_reward']:.1f}")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[bc] saved {zip_path}")

    if a.as_run and task != SIM_TASK:
        print("[bc] --as-run skipped: the dashboard's replay tools only load simulator-task models")
    elif a.as_run:
        run = HW.parent / "results" / "hoop" / "runs" / f"bc_{time.strftime('%Y%m%d-%H%M%S')}"
        run.mkdir(parents=True, exist_ok=True)
        model.save(str(run / "best_model.zip"))
        model.save(str(run / "final_model.zip"))
        (run / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[bc] also saved as sim run {run}")
    return meta


if __name__ == "__main__":
    main()
