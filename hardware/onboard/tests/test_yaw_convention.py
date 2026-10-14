"""The clockwise yaw convention (2026-10-13) and the conversion of everything made before it."""
import json

import numpy as np
import pytest

from onboard import sim as onboard_sim
from onboard import spec
from onboard.sim import OnboardSim, sample_profile


def _turn(profile="typical", action=1.0, ticks=30):
    s = OnboardSim(seed=0, profile=profile, start_pos=[0.0, -0.5, 0.8], start_yaw_deg=0.0)
    y0 = s.state()["yaw"]
    for _ in range(ticks):
        s.step([0.0, 0.0, 0.0, action])
    return s.state()["yaw"] - y0


def test_a_right_turn_reads_clockwise_positive_like_the_tello():
    assert _turn(action=1.0) > 0
    assert _turn(action=-1.0) < 0


def test_yaw_columns_per_layout():
    for width in spec.LAYOUT_FIELDS:
        sl = spec.layout_slices(width)
        cols = spec.yaw_columns(width)
        assert cols[0] == sl["att"].start + 2
        assert len(cols) == (2 if "pass_heading" in sl else 1)


def test_flip_touches_only_the_yaw_columns_and_is_its_own_inverse():
    rng = np.random.default_rng(0)
    for width in spec.LAYOUT_FIELDS:
        o = rng.normal(size=(5, width)).astype(np.float32)
        f = spec.flip_yaw_obs(o)
        cols = spec.yaw_columns(width)
        others = [i for i in range(width) if i not in cols]
        np.testing.assert_array_equal(f[:, others], o[:, others])
        np.testing.assert_array_equal(f[:, cols], -o[:, cols])
        np.testing.assert_array_equal(spec.flip_yaw_obs(f), o)


@pytest.mark.parametrize("width", sorted(spec.LAYOUT_FIELDS))
def test_a_converted_old_model_acts_exactly_as_before(width):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from onboard.policy import convert_old_yaw

    class E(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (spec.ACT_DIM,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(width, np.float32), {}

        def step(self, a):
            return np.zeros(width, np.float32), 0.0, True, False, {}

    model = PPO("MlpPolicy", E(), seed=0, device="cpu")
    obs_old = np.random.default_rng(1).normal(size=(16, width)).astype(np.float32)
    act_old, _ = model.predict(obs_old, deterministic=True)
    v_old = model.policy.predict_values(model.policy.obs_to_tensor(obs_old)[0]).detach().numpy()
    assert convert_old_yaw(model)
    obs_new = spec.flip_yaw_obs(obs_old)
    act_new, _ = model.predict(obs_new, deterministic=True)
    v_new = model.policy.predict_values(model.policy.obs_to_tensor(obs_new)[0]).detach().numpy()
    np.testing.assert_allclose(act_new, act_old, atol=1e-5)
    np.testing.assert_allclose(v_new, v_old, atol=1e-5)


def _demo(tmp_path, name, meta):
    from imitation.record_demo import load_demo
    obs = np.random.default_rng(2).normal(size=(4, spec.OBS_DIM)).astype(np.float32)
    p = tmp_path / name
    np.savez_compressed(p, obs=obs, intent=np.zeros((4, 4), np.float32), executed=np.zeros((4, 4), np.float32),
                        t=np.arange(4, dtype=np.float64), episode=np.zeros(4, np.int64),
                        meta=json.dumps(dict(task="onboard_hoop", **meta)))
    # loading also refreshes the memory fields (obs.refresh_obs), so compare the yaw-rate column,
    # which only the yaw conversion touches
    c = spec.OBS_SLICES["att"].start + 2
    return obs[:, c], load_demo(p)["obs"][:, c]


def test_old_sim_demos_are_flipped_on_load(tmp_path):
    raw, got = _demo(tmp_path, "sim.npz", {"source": "onboard_scripted_expert"})
    np.testing.assert_array_equal(got, -raw)


def test_real_flight_demos_are_left_alone(tmp_path):
    raw, got = _demo(tmp_path, "real.npz", {"source": "human", "dry_run": False})
    np.testing.assert_array_equal(got, raw)


def test_new_demos_are_left_alone(tmp_path):
    raw, got = _demo(tmp_path, "new.npz", {"source": "onboard_scripted_expert", "yaw_convention": "cw"})
    np.testing.assert_array_equal(got, raw)


def test_typical_is_unchanged_and_measured_survives_randomisation():
    typ = onboard_sim.PROFILES["typical"]
    assert (typ.gain_fb, typ.gain_lr, typ.gain_ud, typ.gain_yaw, typ.yaw_deadband) == (1, 1, 1, 1, None)
    meas = onboard_sim.PROFILES["measured"]
    for seed in range(5):
        p = sample_profile(np.random.default_rng(seed), 1.0, meas)
        assert (p.gain_fb, p.gain_ud, p.gain_yaw, p.yaw_deadband) == (
            meas.gain_fb, meas.gain_ud, meas.gain_yaw, meas.yaw_deadband)
    # and the env really uses it with dr on: measured turns slower than typical
    assert abs(_turn("measured")) < abs(_turn("typical"))


def test_measured_yaw_dead_zone_ignores_small_turns():
    # 2 s at 0.15 stick: typical turns ~9 deg; measured ignores it (only yaw drift, ~1 deg)
    assert abs(_turn("measured", action=0.15)) < 4
    assert abs(_turn("typical", action=0.15)) > 6


def test_velocity_is_blank_and_pushed_comes_from_commands():
    """The real Tello's speed readings carry no direction in the cage, so the policy never gets
    them: `vel` is 0 in the sim too, and `pushed` integrates the executed forward command."""
    from onboard.expert import ExpertPolicy
    from onboard.gym_env import OnboardHoopEnv
    from onboard.obs import refresh_obs
    env = OnboardHoopEnv(seed=3, dr=1.0)
    pol = ExpertPolicy()
    rows, ts = [], []
    obs, _ = env.reset()
    pol.reset()
    for _ in range(300):
        rows.append(obs)
        ts.append(env.sim.t)
        obs, _, term, trunc, _ = env.step(pol(obs))
        if term or trunc:
            break
    rows = np.array(rows, np.float32)
    s = spec.OBS_SLICES
    assert np.all(rows[:, s["vel"]] == 0)
    assert rows[:, s["pushed"]].max() > 0          # the expert's blind push is counted
    # replaying a recording gives exactly what the live builder built
    again = refresh_obs(rows, np.zeros(len(rows), np.int64), np.array(ts))
    np.testing.assert_allclose(again, rows, atol=1e-5)
