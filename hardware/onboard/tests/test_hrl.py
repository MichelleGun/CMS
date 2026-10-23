"""Hierarchical RL (hrl.py): options last at most 8 ticks and stop at a pass, the manager's reward is
the env's summed reward, the phase switch changes skill after the first pass, and the flight's
HierarchicalPolicy makes the same choices and actions as OptionEnv in the simulator."""
import numpy as np
import pytest

from onboard import hrl, spec
from onboard.policy import load_policy

C9 = "results/hoop/runs/w10_mg6_C9_loop_practice"
O = "results/hoop/runs/w10_n9_O_oblique"

pytestmark = pytest.mark.skipif(not (hrl.CMS / C9 / "final_model.zip").is_file()
                                or not (hrl.CMS / O / "best_model.zip").is_file(),
                                reason="needs the trained memory policies (C9, O) in results/hoop/runs")


def frame_with_streak(v):
    f = np.zeros(spec.TRACKER_OBS_DIM, np.float32)
    f[hrl.STREAK_IDX] = v
    return f


def test_the_phase_switch_changes_skill_after_the_first_pass():
    assert hrl.phase_choice(frame_with_streak(0.0)) == 0
    assert hrl.phase_choice(frame_with_streak(0.25)) == 1


def test_an_option_lasts_at_most_8_ticks_and_its_reward_is_the_env_sum():
    env = hrl.OptionEnv([C9, O], seed=3)
    rewards = []
    inner = env.env.step

    def spy(a):
        out = inner(a)
        rewards.append(out[1])
        return out
    env.env.step = spy
    env.reset(seed=3)
    for _ in range(12):
        n0 = len(rewards)
        _, r, term, trunc, info = env.step(1)
        assert 1 <= info["option_ticks"] <= hrl.OPTION_TICKS
        assert info["option_ticks"] == len(rewards) - n0
        assert r == pytest.approx(sum(rewards[n0:]))
        if term or trunc:
            break


def test_the_flight_policy_flies_the_hierarchy_as_the_simulator_does(tmp_path):
    """Feed the frames OptionEnv saw to the flight's HierarchicalPolicy (phase switch): it must pick
    the same skill at the same ticks and send the same commands."""
    env = hrl.OptionEnv([O, C9], seed=5, phase=True)
    sent, frames = [], []
    inner = env.env.step

    def spy(a):
        sent.append(np.asarray(a, float).copy())
        out = inner(a)
        frames.append(out[0][:env.frame_dim].copy())
        return out
    env.env.step = spy
    obs0, _ = env.reset(seed=5)
    first = env._full[:env.frame_dim].copy()
    for _ in range(60):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    bundle = hrl.write_phase_bundle(tmp_path / "phase.json", [O, C9])
    fn = load_policy(bundle)
    assert isinstance(fn, hrl.HierarchicalPolicy) and fn.obs_width == spec.TRACKER_OBS_DIM
    fn.reset()
    feed = [first] + frames[:-1]
    got = [fn(f) for f in feed]
    assert np.allclose(np.array(got), np.array(sent), atol=1e-5)


def test_a_learned_manager_round_trips_through_load_policy(tmp_path):
    import json
    from stable_baselines3 import PPO
    env = hrl.OptionEnv([O, C9], seed=1)
    m = PPO("MlpPolicy", env, n_steps=16, batch_size=16, policy_kwargs=dict(net_arch=[8]), device="cpu")
    run = tmp_path / "hrl_x"
    run.mkdir()
    m.save(str(run / "best_model.zip"))
    (run / "meta.json").write_text(json.dumps(hrl.hrl_meta("learned", [O, C9])))
    fn = load_policy(run / "best_model.zip")
    assert isinstance(fn, hrl.HierarchicalPolicy) and fn.kind == "learned"
    fn.reset()
    a = fn(np.zeros(spec.TRACKER_OBS_DIM, np.float32))
    assert a.shape == (spec.ACT_DIM,) and np.all(np.abs(a) <= 1)
