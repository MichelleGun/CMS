"""
Training-run and showcase endpoints, against a temporary runs folder. subprocess.Popen is replaced
by a fake that records argv, so nothing is actually trained or played.

    python -m pytest webapp/backend/tests -q
"""
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# flake8: noqa: E402
import main
import run_manager as rm_mod
from fastapi.testclient import TestClient
from run_manager import RunManager


class FakeProc:
    def __init__(self, argv, **kwargs):
        self.argv, self.pid, self.code = [str(a) for a in argv], 999999, None

    def poll(self):
        return self.code


@pytest.fixture
def env(tmp_path, monkeypatch):
    spawned = []

    def fake_popen(argv, **kwargs):
        p = FakeProc(argv)
        spawned.append(p)
        return p
    monkeypatch.setattr(rm_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rm_mod, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path)
    rm = RunManager()
    monkeypatch.setattr(main, "run_manager", rm)
    return dict(client=TestClient(main.app), runs=tmp_path, spawned=spawned, rm=rm)


def make_run(runs: Path, run_id="ppo_x", models=("latest_model.zip",), env="net"):
    d = runs / run_id
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"run_id": run_id, "algo": "ppo", "env": env}))
    for m in models:
        (d / m).write_bytes(b"zip")
    return d


def test_start_run_passes_the_net_env_options(env):
    r = env["client"].post("/api/runs", json={"algo": "ppo", "timesteps": 1000, "env": "net",
                                              "ring_jitter": 0.0, "radius_start": 2.0,
                                              "sway_deg": 35.0})
    assert r.status_code == 201, r.text
    argv = env["spawned"][0].argv
    assert argv[1].endswith("train_hoop_multi.py")
    assert argv[argv.index("--env") + 1] == "net"
    assert argv[argv.index("--ring-jitter") + 1] == "0.0"
    assert argv[argv.index("--radius-start") + 1] == "2.0"
    assert argv[argv.index("--sway-deg") + 1] == "35.0"


def test_legacy_run_gets_no_net_options(env):
    env["client"].post("/api/runs", json={"algo": "ppo", "timesteps": 1000, "env": "legacy"})
    argv = env["spawned"][0].argv
    assert argv[argv.index("--env") + 1] == "legacy"
    assert "--ring-jitter" not in argv and "--radius-start" not in argv
    assert "--sway-deg" not in argv


def test_default_env_is_net(env):
    env["client"].post("/api/runs", json={"algo": "ppo", "timesteps": 1000})
    argv = env["spawned"][0].argv
    assert argv[argv.index("--env") + 1] == "net"


def test_showcase_starts_the_player_and_truncates_old_output(env):
    d = make_run(env["runs"])
    (d / "showcase.jsonl").write_text('{"old": 1}\n')
    r = env["client"].post("/api/runs/ppo_x/showcase", json={"variant": "latest"})
    assert r.status_code == 202, r.text
    argv = env["spawned"][0].argv
    assert argv[1].endswith("showcase_hoop.py")
    assert argv[argv.index("--run-dir") + 1] == str(d)
    assert (d / "showcase.jsonl").read_text() == ""


def test_only_one_showcase_per_run_at_a_time(env):
    make_run(env["runs"])
    c = env["client"]
    assert c.post("/api/runs/ppo_x/showcase", json={}).status_code == 202
    assert c.post("/api/runs/ppo_x/showcase", json={}).status_code == 409
    env["spawned"][0].code = 0                      # finished
    assert c.post("/api/runs/ppo_x/showcase", json={}).status_code == 202


def test_showcase_needs_a_saved_model(env):
    make_run(env["runs"], models=())
    r = env["client"].post("/api/runs/ppo_x/showcase", json={})
    assert r.status_code == 409 and "no model" in r.json()["detail"]


def test_showcase_of_unknown_run_is_404(env):
    assert env["client"].post("/api/runs/nope/showcase", json={}).status_code == 404


def test_frontend_scripts_are_never_served_stale(env):
    r = env["client"].get("/js/app.js")
    assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache"
    assert env["client"].get("/").headers.get("cache-control") == "no-cache"


def test_config_carries_the_ring_spec(env):
    hoop = env["client"].get("/api/config").json()["hoop"]
    assert hoop["inner_radius"] == pytest.approx(0.15) and len(hoop["centre"]) == 3


def test_realtime_flag_is_created_and_removed(env):
    d = make_run(env["runs"])
    c = env["client"]
    assert c.post("/api/runs/ppo_x/realtime", json={"enabled": True}).status_code == 200
    assert (d / "realtime.flag").is_file()
    assert c.post("/api/runs/ppo_x/realtime", json={"enabled": False}).status_code == 200
    assert not (d / "realtime.flag").exists()
    assert c.post("/api/runs/nope/realtime", json={"enabled": True}).status_code == 404


def test_showcase_is_full_speed_unless_realtime_is_asked_for(env):
    make_run(env["runs"])
    c = env["client"]
    c.post("/api/runs/ppo_x/showcase", json={})
    assert "--realtime" not in env["spawned"][-1].argv
    env["spawned"][-1].code = 0
    c.post("/api/runs/ppo_x/showcase", json={"realtime": True})
    assert "--realtime" in env["spawned"][-1].argv


def test_a_new_run_counts_as_training_before_its_first_evaluation(env):
    r = env["client"].post("/api/runs", json={"algo": "ppo", "timesteps": 1000})
    run_id = r.json()["run_id"]
    d = env["runs"] / run_id
    (d / "meta.json").write_text(json.dumps({"run_id": run_id, "algo": "ppo", "env": "net",
                                             "final_status": None}))
    runs = env["client"].get("/api/runs").json()["runs"]
    assert runs[-1]["status"] == "training" and runs[-1]["current_timestep"] == 0
    assert env["client"].get(f"/api/runs/{run_id}").json()["status"] == "training"
    env["spawned"][0].code = 0                       # process ended without a final status
    assert env["client"].get("/api/runs").json()["runs"][-1]["status"] != "training"


def test_camera_frames_are_served_whole(env):
    d = make_run(env["runs"])
    (d / "camera_latest.jpg").write_bytes(b"\xff\xd8jpegdata")
    r = env["client"].get("/api/runs/ppo_x/camera/latest.jpg")
    assert r.status_code == 200 and r.content == b"\xff\xd8jpegdata"
    assert r.headers["content-type"] == "image/jpeg"


def test_until_stopped_run_passes_the_flag_and_has_no_end(env):
    r = env["client"].post("/api/runs", json={"algo": "ppo", "until_stopped": True})
    assert r.status_code == 201, r.text
    argv = env["spawned"][0].argv
    assert "--until-stopped" in argv
    run_id = r.json()["run_id"]
    (env["runs"] / run_id / "meta.json").write_text(json.dumps({
        "run_id": run_id, "algo": "ppo", "env": "net", "until_stopped": True,
        "timesteps_requested": None, "final_status": None}))
    row = env["client"].get("/api/runs").json()["runs"][-1]
    assert row["until_stopped"] is True and row["total_timesteps"] is None


def test_fixed_runs_are_unchanged(env):
    env["client"].post("/api/runs", json={"algo": "ppo", "timesteps": 1234})
    argv = env["spawned"][0].argv
    assert "--until-stopped" not in argv and argv[argv.index("--timesteps") + 1] == "1234"


def test_episode_variety_flags_reach_the_trainer(env):
    env["client"].post("/api/runs", json={"algo": "ppo", "timesteps": 1000, "env": "net",
                                          "carry_start": 0.0, "reverse_prob": 0.5,
                                          "lean_flip_prob": 0.3})
    argv = env["spawned"][0].argv
    assert argv[argv.index("--carry-start") + 1] == "0.0"
    assert argv[argv.index("--reverse-prob") + 1] == "0.5"
    assert argv[argv.index("--lean-flip-prob") + 1] == "0.3"


def test_a_pinned_start_may_be_near_a_corner_but_the_ring_may_not(env):
    c = env["client"]
    ok = c.post("/api/runs", json={"algo": "ppo", "timesteps": 1000,
                                   "start_pos": {"x": 0.85, "y": -0.85, "z": 0.03}})
    assert ok.status_code == 201, ok.text
    env["spawned"][0].code = 0
    bad = c.post("/api/runs", json={"algo": "ppo", "timesteps": 1000,
                                    "target_pos": {"x": 0.85, "y": 0.0, "z": 1.2}})
    assert bad.status_code == 400


# ---------------------------------------------------------- side-by-side runs, resume, onboard

def test_three_runs_side_by_side_then_refused(env):
    c = env["client"]
    for _ in range(3):
        r = c.post("/api/runs", json={"algo": "ppo", "timesteps": 1000, "n_envs": 2})
        assert r.status_code == 201, r.text
    ids = [p.argv[p.argv.index("--run-id") + 1] for p in env["spawned"]]
    assert len(set(ids)) == 3                       # same-second starts still get unique ids
    r = c.post("/api/runs", json={"algo": "ppo", "timesteps": 1000, "n_envs": 1})
    assert r.status_code == 409 and "limit" in r.json()["detail"]
    assert c.get("/api/runs").json()["capacity"]["live_runs"] == 3


def test_run_needing_more_cores_than_free_is_refused(env, monkeypatch):
    monkeypatch.setattr(rm_mod, "CPU_BUDGET", 6)
    c = env["client"]
    assert c.post("/api/runs", json={"algo": "ppo", "n_envs": 4}).status_code == 201
    r = c.post("/api/runs", json={"algo": "ppo", "n_envs": 4})
    assert r.status_code == 409 and "cores" in r.json()["detail"]
    assert c.post("/api/runs", json={"algo": "ppo", "n_envs": 2}).status_code == 201


def test_continue_from_a_run_passes_init(env):
    d = make_run(env["runs"], "ppo_parent")
    r = env["client"].post("/api/runs", json={"algo": "ppo", "env": "net", "init_run": "ppo_parent"})
    assert r.status_code == 201, r.text
    argv = env["spawned"][-1].argv
    assert argv[argv.index("--init") + 1] == str(d.resolve())
    assert argv[argv.index("--vec") + 1] == "auto"


def test_continue_refuses_another_env_or_algo(env):
    make_run(env["runs"], "ppo_parent", env="net")
    c = env["client"]
    assert c.post("/api/runs", json={"algo": "ppo", "env": "legacy", "init_run": "ppo_parent"}).status_code == 400
    assert c.post("/api/runs", json={"algo": "sac", "env": "net", "init_run": "ppo_parent"}).status_code == 400
    assert c.post("/api/runs", json={"algo": "ppo", "env": "net", "init_run": "nope"}).status_code == 400


def test_onboard_run_uses_the_onboard_trainer_with_clone_and_demos(env, tmp_path, monkeypatch):
    flights = tmp_path / "flights"
    (flights / "bc" / "mix").mkdir(parents=True)
    (flights / "bc" / "mix" / "bc_model.zip").write_bytes(b"zip")
    (flights / "demos").mkdir()
    (flights / "demos" / "t.npz").write_bytes(b"npz")
    monkeypatch.setattr(rm_mod, "FLIGHTS_DIR", flights)
    monkeypatch.setattr(rm_mod, "BC_DIR", flights / "bc")
    r = env["client"].post("/api/runs", json={
        "algo": "ppo", "env": "onboard", "init_clone": "mix", "demo_ids": ["demos/t.npz"],
        "human_weight": 5, "bc_coef": 0.5, "profile": "harsh"})
    assert r.status_code == 201, r.text
    argv = env["spawned"][-1].argv
    assert argv[1].endswith("train_onboard.py")
    assert r.json()["run_id"].startswith("onboard_")
    assert argv[argv.index("--init") + 1] == str((flights / "bc" / "mix").resolve())
    assert argv[argv.index("--demos") + 1] == str((flights / "demos" / "t.npz").resolve())
    assert "human=5.0" in argv and argv[argv.index("--bc-coef") + 1] == "0.5"
    assert argv[argv.index("--profile") + 1] == "harsh"
    bad = env["client"].post("/api/runs", json={"algo": "ppo", "env": "onboard",
                                                "demo_ids": ["../../etc.npz"]})
    assert bad.status_code == 400


def test_onboard_runs_have_no_pybullet_replay_or_showcase(env):
    make_run(env["runs"], "onboard_x", models=("best_model.zip",), env="onboard")
    c = env["client"]
    assert c.post("/api/runs/onboard_x/showcase", json={}).status_code in (400, 404)
    assert c.post("/api/runs/onboard_x/replay", json={}).status_code in (400, 404)


def test_start_run_passes_swing_ramp_and_reward(env):
    r = env["client"].post("/api/runs", json={
        "algo": "ppo", "timesteps": 1000, "env": "net", "wind_steps": 0,
        "reward_mode": "budget", "budget": 5, "laps": 0,
        "reward_weights": {"budget_pass": 2.0}})
    assert r.status_code == 201, r.text
    argv = env["spawned"][0].argv
    assert argv[argv.index("--wind-steps") + 1] == "0"
    assert argv[argv.index("--reward") + 1] == "budget"
    assert argv[argv.index("--budget") + 1] == "5.0"
    assert json.loads(argv[argv.index("--reward-weights") + 1]) == {"budget_pass": 2.0}


def test_default_run_sends_no_reward_overrides(env):
    env["client"].post("/api/runs", json={"algo": "ppo", "timesteps": 1000, "env": "net"})
    argv = env["spawned"][0].argv
    assert argv[argv.index("--wind-steps") + 1] == "1500000"
    assert argv[argv.index("--reward") + 1] == "shaped"
    assert "--reward-weights" not in argv


def test_config_lists_the_reward_terms(env):
    cfg = env["client"].get("/api/config").json()
    names = {n for terms in cfg["reward"]["terms"].values() for n, _ in terms}
    assert names <= set(cfg["reward"]["defaults"])
    assert cfg["reward"]["defaults"]["pass_bonus"] > 0


def test_rename_run_keeps_meta_untouched(env):
    d = make_run(env["runs"], "ppo_a")
    before = (d / "meta.json").read_text()
    r = env["client"].put("/api/runs/ppo_a/label", json={"label": "  wide   ring test "})
    assert r.status_code == 200 and r.json()["label"] == "wide ring test"
    assert (d / "meta.json").read_text() == before            # the trainer owns meta.json
    runs = env["client"].get("/api/runs").json()["runs"]
    assert runs[0]["label"] == "wide ring test"
    assert env["client"].get("/api/runs/ppo_a").json()["label"] == "wide ring test"
    env["client"].put("/api/runs/ppo_a/label", json={"label": ""})
    assert env["client"].get("/api/runs").json()["runs"][0]["label"] is None
    assert env["client"].put("/api/runs/nope/label", json={"label": "x"}).status_code == 404
    assert env["client"].put("/api/runs/ppo_a/label", json={"label": "x" * 61}).status_code == 422


def _evals(d: Path, successes, rewards):
    import numpy as np
    s = np.array(successes, dtype=float)
    np.savez(d / "evaluations.npz", timesteps=np.arange(len(s)) * 1000,
             results=np.array(rewards, dtype=float), ep_lengths=np.ones_like(s), successes=s)


def _net_run(runs: Path, run_id, obs_dim=82, reward_mode="shaped", best_reward=None):
    d = runs / run_id
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"run_id": run_id, "algo": "ppo", "env": "net",
                                             "obs_dim": obs_dim, "reward_mode": reward_mode}))
    (d / "latest_model.zip").write_bytes(b"zip")
    if best_reward is not None:
        (d / "live_status.json").write_text(json.dumps({"best_eval_reward": best_reward}))
    return d


def test_best_run_is_picked_by_success_not_reward(env):
    shaped = _net_run(env["runs"], "ppo_shaped", best_reward=13.0)
    _evals(shaped, [[0, 0], [0, 0]], [[13, 13], [12, 12]])
    budget = _net_run(env["runs"], "ppo_budget", reward_mode="budget", best_reward=2.0)
    _evals(budget, [[0, 0], [1, 0]], [[1, 1], [2, 2]])
    old = _net_run(env["runs"], "ppo_old", obs_dim=None, best_reward=50.0)   # older env version
    _evals(old, [[1, 1]], [[50, 50]])
    data = env["client"].get("/api/runs").json()
    assert data["best_run_id"] == "ppo_budget"
    rows = {r["run_id"]: r for r in data["runs"]}
    assert rows["ppo_budget"]["best_success_rate"] == 0.5
    assert rows["ppo_old"]["comparable"] is False


def test_star_and_name_live_side_by_side(env):
    d = make_run(env["runs"], "ppo_k")
    c = env["client"]
    assert c.put("/api/runs/ppo_k/star", json={"starred": True}).json()["starred"] is True
    c.put("/api/runs/ppo_k/label", json={"label": "keeper"})
    row = c.get("/api/runs").json()["runs"][0]
    assert row["starred"] is True and row["label"] == "keeper"
    c.put("/api/runs/ppo_k/label", json={"label": ""})
    assert c.get("/api/runs/ppo_k").json()["starred"] is True       # renaming keeps the star
    c.put("/api/runs/ppo_k/star", json={"starred": False})
    assert not (d / "label.json").exists()                           # nothing left to keep
    assert c.put("/api/runs/nope/star", json={"starred": True}).status_code == 404
