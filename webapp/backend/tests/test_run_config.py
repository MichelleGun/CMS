"""
The Config card's API: GET /api/runs/{id}/config, POST /api/runs/command, and the new onboard fields
of POST /api/runs (PPO hyperparameters, ring tracker, reward weights). Training is never started:
subprocess.Popen is replaced by a recorder.

    python -m pytest webapp/backend/tests/test_run_config.py -q
"""
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# flake8: noqa: E402
import main
import run_config
import run_manager as rm_mod
from fastapi.testclient import TestClient
from run_manager import PYTHON_EXE, RunManager

ONBOARD_META = {
    "run_id": "w10_x", "algo": "ppo", "env": "onboard", "task": "onboard_hoop", "profile": "measured",
    "seed": 3, "timesteps_requested": 32_630_000, "until_stopped": False, "n_envs": 8, "vec": "subproc",
    "eval_freq": 2500, "n_eval_episodes": 36, "init_timesteps": 29_630_000, "learning_rate": 0.0001,
    "critic_warmup_steps": 20000, "laps": 0, "lap_dir": "same", "streak": 0.5, "streak_cap": 3.0,
    "robust": {"dr": 1.0, "sway_deg": 5.0, "hard_start_p": 0.0, "wash_deg": 15.0, "start_mode": "zones",
               "dr_steps": 0, "bc_augment": True},
    "obs_dim": 37, "final_status": "completed", "final_timesteps": 32_644_656,
    "reward": dict(run_config.REWARD_DEFAULTS, r_passed=25.0, streak=0.5, streak_cap=3.0),
    "bc": {"demos": ["demo.npz"], "demo_paths": ["tello_1/demo.npz"], "samples": {"human": 10},
           "source_weights": {"human": 3.0}, "coef": 0.1, "coef_min": 0.1, "decay_steps": 300000},
    "ppo": {"gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0, "target_kl": 0.02, "batch_size": 512,
            "n_epochs": 5, "n_steps": 2048, "use_sde": False, "sde_sample_freq": None, "log_std_reset": None},
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    spawned = []

    class Proc:
        def __init__(self, argv, **kw):
            self.argv, self.pid = [str(a) for a in argv], 999999

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        spawned.append(Proc(argv))
        return spawned[-1]
    monkeypatch.setattr(rm_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rm_mod, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path)
    rm = RunManager()
    monkeypatch.setattr(main, "run_manager", rm)
    d = tmp_path / "w10_x"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(ONBOARD_META))
    (d / "latest_model.zip").write_bytes(b"not a real model: only its existence is checked here")
    return dict(client=TestClient(main.app), runs=tmp_path, spawned=spawned, rm=rm)


def test_the_reward_table_matches_the_trainers_own():
    sys.path.insert(0, str(BACKEND.parents[1] / "hardware"))
    from onboard.gym_env import REWARD_WEIGHTS
    expected = {k: v for k, v in REWARD_WEIGHTS.items() if k not in ("streak", "streak_cap")}
    assert run_config.REWARD_DEFAULTS == expected


def test_the_old_panels_payload_builds_the_command_it_always_did(env):
    r = env["client"].post("/api/runs", json=dict(
        algo="ppo", env="onboard", timesteps=500_000, n_envs=4, eval_freq=2500, n_eval_episodes=18, seed=1,
        profile="typical", onboard_dr=1.0, onboard_sway_deg=5.0, onboard_wash_deg=15.0, hard_start_p=0.5,
        dr_steps=500_000, bc_augment=True, same_side=True, start_zones=True, streak=0.5, vec="auto"))
    assert r.status_code == 201, r.text
    argv = env["spawned"][-1].argv
    run_dir = str(env["runs"] / r.json()["run_id"])
    assert argv == [PYTHON_EXE, str(run_config.ONBOARD_TRAIN),
                    "--timesteps", "500000", "--n-envs", "4", "--vec", "auto", "--eval-freq", "2500",
                    "--n-eval-episodes", "18", "--run-id", r.json()["run_id"], "--out", run_dir, "--seed", "1",
                    "--profile", "typical", "--dr", "1.0", "--sway-deg", "5.0", "--wash-deg", "15.0",
                    "--lap-dir", "same", "--start-mode", "zones", "--streak", "0.5", "--hard-start-p", "0.5",
                    "--dr-steps", "500000", "--laps", "0", "--bc-augment"]


def test_new_fields_become_trainer_flags(env):
    r = env["client"].post("/api/runs", json=dict(
        algo="ppo", env="onboard", profile="measured", gamma=0.995, gae_lambda=1.0, ent_coef=0.005,
        target_kl=0.02, batch_size=512, n_epochs=5, learning_rate=3e-5, critic_warmup_steps=10000,
        log_std_reset=-1.0, sde=True, ring_tracker=True, streak_cap=4.0, onboard_laps=0, init_run="w10_x",
        reward_weights={"r_passed": 25.0, "w_rim": 0.3, "drain": 0.0}))
    assert r.status_code == 201, r.text
    a = env["spawned"][-1].argv

    def after(flag):
        return a[a.index(flag) + 1]
    assert after("--profile") == "measured" and after("--gamma") == "0.995" and after("--gae-lambda") == "1.0"
    assert after("--target-kl") == "0.02" and after("--batch-size") == "512" and after("--n-epochs") == "5"
    assert after("--lr") == "3e-05" and after("--critic-warmup-steps") == "10000"
    assert after("--log-std-reset") == "-1.0" and "--sde" in a and "--ring-tracker" in a
    assert after("--streak-cap") == "4.0" and after("--init").endswith("w10_x")
    # only values that differ from the defaults are sent (w_rim 0.3 is the default)
    assert json.loads(after("--reward-weights")) == {"r_passed": 25.0, "drain": 0.0}


@pytest.mark.parametrize("bad", [dict(gamma=1.5), dict(target_kl=0.0), dict(batch_size=3),
                                 dict(n_epochs=0), dict(learning_rate=1.0), dict(profile="wild")])
def test_out_of_range_values_are_refused(env, bad):
    r = env["client"].post("/api/runs", json=dict(algo="ppo", env="onboard", **bad))
    assert r.status_code == 422 and not env["spawned"]


def test_an_unknown_reward_weight_is_refused(env):
    r = env["client"].post("/api/runs", json=dict(algo="ppo", env="onboard", reward_weights={"r_pased": 1}))
    assert r.status_code == 400 and "r_pased" in r.json()["detail"] and not env["spawned"]


def test_the_config_of_a_run_lists_its_values_and_what_is_editable(env):
    c = env["client"].get("/api/runs/w10_x/config").json()
    assert c["editable"] and c["status"] == "completed"
    v = {f["key"]: f["value"] for f in c["fields"]}
    assert v["timesteps"] == 3_000_000                      # the run added 3M, not 32.6M
    assert v["profile"] == "measured" and v["gamma"] == 0.99 and v["target_kl"] == 0.02
    assert v["batch_size"] == 512 and v["same_side"] is True and v["start_zones"] is True
    assert v["ring_tracker"] is False and v["bc_coef"] == 0.1 and v["human_weight"] == 3.0
    assert {f["group"] for f in c["fields"]} == {"Run", "Environment", "PPO", "Imitation"}
    rw = {r["key"]: r for r in c["reward"]}
    assert rw["r_passed"]["value"] == 25.0 and rw["r_passed"]["default"] == 20.0
    assert rw["drain"]["type"] == "bool"
    assert c["demo_paths"] == ["tello_1/demo.npz"]
    assert "--profile measured" in c["command"] and "--reward-weights" in c["command"]
    assert "--target-kl 0.02" in c["command"]


def test_an_old_run_without_ppo_numbers_falls_back_to_the_models_or_the_trainers(env):
    meta = {k: v for k, v in ONBOARD_META.items() if k != "ppo"}
    meta["run_id"] = "old"
    d = env["runs"] / "old"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps(meta))
    v = {f["key"]: f["value"] for f in env["client"].get("/api/runs/old/config").json()["fields"]}
    assert v["gamma"] == 0.99 and v["batch_size"] == 64 and v["target_kl"] is None


def test_a_net_run_has_a_config_view_but_nothing_editable(env):
    d = env["runs"] / "net1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"run_id": "net1", "algo": "ppo", "env": "net"}))
    c = env["client"].get("/api/runs/net1/config").json()
    assert c["editable"] is False and c["fields"] == [] and c["info"]


def test_an_unknown_run_is_a_404(env):
    assert env["client"].get("/api/runs/nope/config").status_code == 404


def test_the_command_preview_starts_nothing(env):
    r = env["client"].post("/api/runs/command", json=dict(
        algo="ppo", env="onboard", gamma=0.995, ring_tracker=True, reward_weights={"r_passed": 30.0}))
    assert r.status_code == 200
    cmd = r.json()["command"]
    assert cmd.startswith("python hardware/onboard/train_onboard.py")
    assert "--gamma 0.995" in cmd and "--ring-tracker" in cmd and "r_passed" in cmd
    assert not env["spawned"]
    assert env["client"].post("/api/runs/command", json=dict(algo="ppo", env="net")).status_code == 400


def test_the_command_shows_project_paths_relative_to_cms(env):
    from config import CMS_DIR
    demo = CMS_DIR / "hardware" / "flights" / "tello_x" / "demo.npz"
    argv = run_config.onboard_argv(PYTHON_EXE, "r1", CMS_DIR / "results" / "hoop" / "runs" / "r1",
                                   dict(timesteps=1000, n_envs=1, eval_freq=100, n_eval_episodes=1,
                                        profile="typical", onboard_dr=0.0, onboard_sway_deg=5.0,
                                        onboard_wash_deg=15.0, same_side=True, start_zones=True, streak=0.5,
                                        hard_start_p=0.0, dr_steps=0, onboard_laps=0, bc_augment=True,
                                        human_weight=3.0, bc_coef=1.0),
                                   init_dir=CMS_DIR / "results" / "hoop" / "runs" / "parent", demos=[demo])
    line = run_config.command_line(argv)
    assert str(CMS_DIR) not in line
    assert "--out results/hoop/runs/r1" in line and "--init results/hoop/runs/parent" in line
    assert "--demos hardware/flights/tello_x/demo.npz" in line


def test_the_architecture_is_read_in_a_subprocess_so_the_dashboard_never_imports_torch(tmp_path):
    """A real model zip (saved by SB3) gives its layer shapes; torch is not imported in this process."""
    import subprocess
    z = tmp_path / "latest_model.zip"
    subprocess.run([sys.executable, "-c",
                    "import gymnasium as gym, numpy as np; from stable_baselines3 import PPO; "
                    "e = gym.make('Pendulum-v1'); PPO('MlpPolicy', e, seed=0).save(r'%s')" % z],
                   check=True, capture_output=True)
    had_torch = "torch" in sys.modules
    arch = run_config.architecture(tmp_path)
    assert arch["model_file"] == "latest_model.zip"
    assert arch["layers.mlp_extractor.policy_net.0.weight"] == "64x3"
    assert arch["ppo.gamma"] == 0.99
    assert ("torch" in sys.modules) == had_torch
