"""The two entry-point scripts, driven by scripted key events against FakeTello (no window, no drone)."""
import json
from pathlib import Path

import numpy as np
import pytest

from common.config import CMS_ROOT
from control import keyboard_teleop, run_policy
from control.inputs import ScriptedInput
from control.loop import movement_action
from imitation import record_demo
from imitation.record_demo import load_demo

BASE = ["--dry-run", "--headless", "--no-log", "--no-hotkeys", "--yes", "--profile", "typical", "--seed", "4"]


def test_movement_action_combines_keys_in_cage_frame():
    a = movement_action({"x+", "y+"}, 0.7)
    assert np.allclose(a[:3] / np.linalg.norm(a[:3]), [np.sqrt(.5), np.sqrt(.5), 0]) and a[3] == 0.7
    assert np.all(movement_action({"x+", "x-"}, 1.0) == 0)                 # opposite keys cancel -> hover
    assert np.all(movement_action(set(), 1.0) == 0)


def teleop_script():
    # takeoff is retried because the estimator needs ~1 s of data before the preflight check passes
    ev = {45: {"press": ["takeoff"]}, 90: {"press": ["takeoff"]}}
    ev[300] = {"hold": ["x+"]}
    ev[420] = {"hold": ["x+", "y+"]}
    ev[520] = {"hold": []}
    ev[560] = {"press": ["land"]}
    return ScriptedInput(ev)


def test_teleop_flight_records_demo_and_lands(tmp_path, monkeypatch):
    monkeypatch.setattr(record_demo, "DEMO_DIR", tmp_path)
    summary, rec, session = keyboard_teleop.main(
        BASE + ["--record", "--hoop", "0.6", "0.3", "1.0", "--max-seconds", "90"],
        inputs=teleop_script(), realtime=False)
    assert summary["phase"] == "landed"
    assert len(rec) > 100
    files = list(tmp_path.glob("demo_*.npz"))
    assert len(files) == 1
    d = load_demo(files[0])
    assert d["obs"].shape[1] == 75 and d["intent"].shape[1] == 4
    # recorded only while flying and moving: leading idle is skipped, movement present
    assert np.linalg.norm(d["intent"][0, :3]) > 0
    dirs = {tuple(np.sign(r[:2]).astype(int)) for r in d["intent"] if np.linalg.norm(r[:3]) > 0}
    assert (1, 0) in dirs and (1, 1) in dirs
    assert np.all(d["obs"][0, 12:72] == 0) or np.any(d["obs"][5, 12:72] != 0)   # buffer fills with executed actions
    assert d["meta"]["source"] == "keyboard" and d["meta"]["hoop"] == [0.6, 0.3, 1.0]


def test_teleop_kill_key_cuts_motors(tmp_path, monkeypatch):
    monkeypatch.setattr(record_demo, "DEMO_DIR", tmp_path)
    ev = {45: {"press": ["takeoff"]}, 90: {"press": ["takeoff"]}, 330: {"press": ["kill"]}}
    summary, rec, session = keyboard_teleop.main(BASE + ["--max-seconds", "60"], inputs=ScriptedInput(ev), realtime=False)
    assert summary["phase"] == "killed"
    assert any(k == "emergency" for k, _ in session.backend.log)


def test_teleop_refuses_takeoff_before_estimate_is_ready():
    ev = {0: {"press": ["takeoff"]}, 5: {"press": ["land"]}, 40: {"press": ["quit"]}}
    summary, rec, session = keyboard_teleop.main(BASE + ["--max-seconds", "3"], inputs=ScriptedInput(ev), realtime=False)
    assert all(k != "takeoff" for k, _ in session.backend.log)


def test_policy_run_with_custom_policy_reaches_hoop_and_lands():
    def go_to_hoop(obs):
        return np.array([*obs[0, 72:75], 1.0])
    summary, rec, session = run_policy.main(
        BASE + ["--hoop", "0.5", "-0.3", "0.9", "--speed-scale", "1.0", "--max-seconds", "120"],
        policy_fn=go_to_hoop)
    assert summary["reached"] and summary["phase"] == "landed"


def test_policy_human_takeover_is_recorded_as_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(record_demo, "DEMO_DIR", tmp_path)

    def lazy_policy(obs):
        return np.zeros(4)                                              # a policy that just hovers

    ev = {45: {}, 400: {"hold": ["y+"]}, 480: {"hold": []}, 520: {"press": ["land"]}}
    summary, rec, session = run_policy.main(
        BASE + ["--hoop", "0.0", "0.5", "1.0", "--record", "--max-seconds", "90"],
        inputs=ScriptedInput(ev), policy_fn=lazy_policy, realtime=False)
    assert summary["phase"] == "landed"
    files = list(tmp_path.glob("demo_*.npz"))
    assert files and load_demo(files[0])["meta"]["source"] == "dagger"


def test_policy_rejects_out_of_range_hoop_unless_forced():
    with pytest.raises(SystemExit):
        run_policy.main(BASE + ["--hoop", "1.5", "0", "1.0"], policy_fn=lambda o: np.zeros(4))


def test_policy_rejects_model_with_wrong_observation_shape(tmp_path):
    from stable_baselines3 import PPO
    import gymnasium as gym
    from gymnasium import spaces

    class E(gym.Env):
        observation_space = spaces.Box(-1, 1, (12,), np.float32)
        action_space = spaces.Box(-1, 1, (4,), np.float32)
        def reset(self, *, seed=None, options=None):
            return np.zeros(12, np.float32), {}
        def step(self, a):
            return np.zeros(12, np.float32), 0.0, True, False, {}
    PPO("MlpPolicy", E(), device="cpu").save(str(tmp_path / "m.zip"))
    with pytest.raises(SystemExit):
        run_policy.load_policy(model_path=tmp_path / "m.zip")


REAL = CMS_ROOT / "results" / "hoop" / "best_model.zip"


@pytest.mark.skipif(not REAL.is_file(), reason="no trained sim model in results/hoop")
def test_real_trained_sim_model_loads_and_produces_valid_actions():
    fn, desc = run_policy.load_policy(model_path=REAL)
    a = fn(np.zeros((1, 75), dtype=np.float32))
    assert a.shape == (4,) and np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0 + 1e-6)
    summary, rec, session = run_policy.main(
        BASE + ["--model", str(REAL), "--hoop", "0.4", "-0.3", "1.0", "--speed-scale", "1.0", "--max-seconds", "60"])
    assert summary["phase"] in ("landed", "killed")                    # whatever the policy does, it ends safely
    assert session.monitor.latched in (0, 2, 3)
