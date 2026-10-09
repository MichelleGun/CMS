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
