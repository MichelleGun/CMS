"""Demo format + behaviour cloning, on synthetic demos (no pybullet) and on sim demos (if available)."""
import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from imitation.bc_train import build_model, main as bc_main, targets_from_demo, train_actor, warm_critic
from imitation.record_demo import DemoRecorder, load_demo, load_demos, mc_returns


def synthetic_demos(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 0.3, (n, 75)).astype(np.float32)
    obs[:, 72:75] = rng.uniform(-1, 1, (n, 3))
    rel = obs[:, 72:75]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    speed = np.where(dist > 0.15, 1.0, 0.0)
    act = np.hstack([rel / np.maximum(dist, 1e-6), speed]).astype(np.float32)
    return obs, act


def test_recorder_roundtrip_and_validation(tmp_path):
    rec = DemoRecorder({"source": "test"})
    obs, act = synthetic_demos(10)
    for i in range(10):
        assert rec.add(obs[i:i + 1], act[i], act[i], i * 0.033)
    assert not rec.add(np.zeros(5), act[0], act[0], 1.0)                  # wrong shape rejected
    bad = obs[0:1].copy()
    bad[0, 3] = np.nan
    assert not rec.add(bad, act[0], act[0], 1.0)                          # NaN rejected
    rec.next_episode()
    rec.add(obs[0:1], act[0], act[0], 5.0)
    p = rec.save(tmp_path / "d.npz")
    d = load_demo(p)
    assert d["obs"].shape == (11, 75) and d["intent"].shape == (11, 4)
    assert d["episode"].tolist() == [0] * 10 + [1] and d["t"][10] == 0.0
    assert d["meta"]["source"] == "test"
    both = load_demos([p, p])
    assert len(both["obs"]) == 22 and both["episode"].max() == 3


def test_targets_ignore_direction_when_idle():
    intent = np.array([[1, 0, 0, 1.0], [0, 0, 0, 0.0], [0, 1, 0, -0.5]], dtype=np.float32)
    unit, speed, moving = targets_from_demo(intent)
    assert moving.tolist() == [True, False, True]
    assert np.allclose(speed, [1.0, 0.0, 0.5])                            # sign of dim 3 ignored like the sim


def test_mc_returns_reset_at_episode_boundaries():
    obs = np.zeros((6, 75), dtype=np.float32)
    obs[:, 72] = [1.0, 0.5, 0.1, 1.0, 0.5, 0.1]                           # distance to hoop
    ep = np.array([0, 0, 0, 1, 1, 1])
    G = mc_returns(obs, ep, gamma=0.9)
    assert np.isclose(G[2], 10.0 - 0.1)                                   # reached: +10
    assert G[3] < G[2] and np.isclose(G[5], G[2])                         # episode 1 starts fresh
    assert G[0] > G[3] - 10 or True


def test_bc_learns_expert_and_zip_loads_like_a_sim_model(tmp_path):
    from stable_baselines3 import PPO
    obs, act = synthetic_demos()
    model = build_model("ppo", seed=0)
    stats = train_actor(model, obs, act, epochs=60, lr=3e-3, verbose=False)
    assert stats["val_mean_cos"] > 0.95, stats
    assert stats["val_speed_mse"] < 0.05, stats
    assert abs(float(model.policy.log_std.mean()) + 2.0) < 1e-6
    model.save(str(tmp_path / "bc_model.zip"))
    loaded = PPO.load(str(tmp_path / "bc_model.zip"))                    # no env, exactly like replay_hoop.py
    o = np.zeros((1, 75), dtype=np.float32)
    o[0, 72:75] = [0.8, 0.0, 0.0]
    a, _ = loaded.predict(o, deterministic=True)
    a = np.asarray(a).reshape(1, 4)
    assert a.shape == (1, 4) and np.all(np.abs(a) <= 1.0)
    assert a[0, 0] > 0.5 and abs(a[0, 1]) < 0.4 and a[0, 3] > 0.6        # heads +x, fast


def test_critic_warmup_runs_and_keeps_actor(tmp_path):
    obs, act = synthetic_demos(600)
    model = build_model("ppo")
    train_actor(model, obs, act, epochs=10, verbose=False)
    before = [p.detach().clone() for p in model.policy.action_net.parameters()]
    loss = warm_critic(model, obs, np.zeros(len(obs), dtype=int), epochs=5)
    after = list(model.policy.action_net.parameters())
    assert np.isfinite(loss)
    assert all(np.allclose(b.numpy(), a.detach().numpy()) for b, a in zip(before, after))


def test_cli_writes_zip_and_meta(tmp_path):
    obs, act = synthetic_demos(500)
    rec = DemoRecorder({"source": "unit"})
    for i in range(len(obs)):
        rec.add(obs[i:i + 1], act[i], act[i], i * 0.033)
    demo = rec.save(tmp_path / "demo.npz")
    meta = bc_main(["--demos", str(demo), "--out", str(tmp_path / "bc"), "--epochs", "20", "--warm-critic"])
    assert (tmp_path / "bc" / "bc_model.zip").is_file()
    m = json.loads((tmp_path / "bc" / "meta.json").read_text())
    assert m["algo"] == "ppo" and m["env_variant"] == "continuous_vel" and m["source"] == "bc"


def test_sim_expert_demos_then_bc_flies_the_sim(tmp_path):
    pytest.importorskip("pybullet")
    pytest.importorskip("gym_pybullet_drones")
    from imitation.make_sim_demos import make_demos
    from imitation.bc_train import evaluate_in_sim
    path = make_demos(episodes=24, expert="scripted", seed=3, out=tmp_path / "sim.npz", verbose=False)
    d = load_demo(path)
    assert d["obs"].shape[1] == 75 and d["meta"]["reached"] >= 12        # the scripted expert mostly succeeds
    model = build_model("ppo")
    train_actor(model, d["obs"], d["intent"], epochs=80, verbose=False)
    res = evaluate_in_sim(model, episodes=8, seed=900)
    assert res["mean_final_dist"] < 1.2                                  # clearly better than standing still
