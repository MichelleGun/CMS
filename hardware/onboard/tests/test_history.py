"""History observations and the asymmetric critic (2026-10-14): what the simulator trains on must be
exactly what policy.load_policy rebuilds on the drone, and the actor must ignore the critic's inputs."""
import json

import numpy as np
import pytest

from onboard import spec
from onboard.gym_env import OnboardHoopEnv
from onboard.history import PRIV_DIM, HistoryStack, parse_lags, stacked_width

LAGS = (0, 1, 2, 4)


def test_lags_are_parsed_and_must_start_at_zero():
    assert parse_lags("0,2,1") == (0, 1, 2)
    assert parse_lags(None) is None
    with pytest.raises(ValueError):
        parse_lags("1,2")


def test_the_stack_is_newest_first_and_starts_full():
    h = HistoryStack(2, (0, 1, 3))
    first = h.push([1, 1])
    assert first.tolist() == [1, 1, 1, 1, 1, 1]       # before the episode: the first frame
    for v in (2, 3, 4):
        out = h.push([v, v])
    assert out.tolist() == [4, 4, 3, 3, 1, 1]
    h.reset()
    assert h.push([9, 9]).tolist() == [9] * 6


def test_env_layout_frame_history_then_privileged():
    env = OnboardHoopEnv(profile="measured", seed=3, laps=0, lap_dir="same", start_mode="zones",
                         ring_tracker=True, history_lags=LAGS, privileged=True)
    obs, _ = env.reset(seed=3)
    w = spec.TRACKER_OBS_DIM
    assert obs.shape == (stacked_width(w, LAGS, True),) == (w * len(LAGS) + PRIV_DIM,)
    assert env.actor_dim == w * len(LAGS)
    frames = [obs[:w].copy()]
    for _ in range(5):
        obs, *_ = env.step(np.array([0.0, 0.3, 0.0, 0.2]))
        frames.append(obs[:w].copy())
    # lag 1 is the previous frame, lag 4 the one four ticks ago
    assert np.allclose(obs[w:2 * w], frames[-2]) and np.allclose(obs[3 * w:4 * w], frames[-5])


def _model(tmp_path, privileged=True):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from onboard.aac_policy import MaskedActorCriticPolicy
    kw = dict(profile="measured", laps=0, lap_dir="same", start_mode="zones", ring_tracker=True,
              history_lags=LAGS, privileged=privileged)
    env = OnboardHoopEnv(seed=0, **kw)
    model = PPO(MaskedActorCriticPolicy, DummyVecEnv([lambda: env]), n_steps=64, batch_size=32,
                device="cpu", policy_kwargs=dict(actor_dim=env.actor_dim, net_arch=[32, 32]))
    d = tmp_path / "run"
    d.mkdir()
    model.save(str(d / "best_model.zip"))
    (d / "meta.json").write_text(json.dumps({
        "algo": "ppo", "task": spec.TASK_NAME, "frame_dim": spec.TRACKER_OBS_DIM, "ring_tracker": True,
        "obs_history": list(LAGS), "asym_critic": True, "yaw_convention": spec.YAW_CONVENTION}))
    return model, env, d, kw


def test_the_actor_ignores_the_privileged_inputs(tmp_path):
    import torch
    model, env, _, _ = _model(tmp_path)
    obs, _ = env.reset(seed=1)
    x = torch.as_tensor(np.stack([obs, obs]))
    x[1, env.actor_dim:] = torch.randn(PRIV_DIM) * 10
    with torch.no_grad():
        mean = model.policy.get_distribution(x).distribution.mean
        values = model.policy.predict_values(x)
    assert torch.allclose(mean[0], mean[1])           # actor: same action
    assert not torch.allclose(values[0], values[1])  # critic: does read them


def test_the_drone_side_loader_flies_exactly_what_was_trained(tmp_path):
    """policy.load_policy fed plain frames (as fly_hoop does) == the model fed the env's stacked
    observation, tick by tick, including after a reset."""
    from onboard.policy import load_policy
    model, _, d, kw = _model(tmp_path)
    fn = load_policy(d / "best_model.zip")
    assert fn.obs_width == spec.TRACKER_OBS_DIM
    env_full = OnboardHoopEnv(seed=5, **kw)
    env_plain = OnboardHoopEnv(seed=5, **dict(kw, history_lags=None, privileged=False))
    for ep in range(2):
        o_full, _ = env_full.reset(seed=50 + ep)
        o_plain, _ = env_plain.reset(seed=50 + ep)
        fn.reset()
        for _ in range(30):
            a_full, _ = model.predict(o_full, deterministic=True)
            a_plain = fn(o_plain)
            assert np.allclose(np.clip(a_full, -1, 1), a_plain, atol=1e-5)
            o_full, *_ = env_full.step(a_full)
            o_plain, *_ = env_plain.step(a_plain)


def test_a_one_frame_parent_is_widened_onto_the_current_frame(tmp_path):
    """train_onboard --init <plain 41-input run> --history-lags ... --asym-critic: the parent's
    weights sit on the CURRENT frame, so it starts out acting exactly as before."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from vec_util import widen_policy_input
    plain_kw = dict(profile="measured", laps=0, lap_dir="same", start_mode="zones", ring_tracker=True)
    plain = PPO("MlpPolicy", DummyVecEnv([lambda: OnboardHoopEnv(seed=0, **plain_kw)]), device="cpu")
    plain.save(str(tmp_path / "p.zip"))
    wide_env = OnboardHoopEnv(seed=0, **plain_kw, history_lags=LAGS, privileged=True)
    widen_policy_input(tmp_path / "p.zip", wide_env.observation_space, spec.TRACKER_OBS_DIM,
                       tmp_path / "w.zip")
    wide = PPO.load(str(tmp_path / "w.zip"), device="cpu")
    obs, _ = wide_env.reset(seed=2)
    a_wide, _ = wide.predict(obs, deterministic=True)
    a_plain, _ = plain.predict(obs[:spec.TRACKER_OBS_DIM], deterministic=True)
    assert np.allclose(a_wide, a_plain, atol=1e-6)


def test_dwell_costs_loitering_in_the_opening_only_when_switched_on():
    env = OnboardHoopEnv(profile="ideal", seed=0, reward_weights={"w_dwell": 0.05})
    env0 = OnboardHoopEnv(profile="ideal", seed=0)
    for e in (env, env0):
        e.reset(seed=0)
        e.sim.pos = e.sim.ring.centre.copy()          # right in the opening
        e._reset_shaping()
    r_hover = env.step(np.zeros(4))[1]
    r_hover0 = env0.step(np.zeros(4))[1]
    assert r_hover < r_hover0
