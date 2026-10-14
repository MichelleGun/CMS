"""
Dashboard API, against temporary folders. No subprocess is started: `spawn` is replaced
by a fake that records the argv, so these tests check what WOULD run.

    python -m pytest webapp/backend/tests -q
"""
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# flake8: noqa: E402
import drone_manager as dm_mod
import jobs as jobs_mod
import main
from drone_manager import DroneManager
from imitation_manager import ImitationManager
from jobs import JobManager
from fastapi.testclient import TestClient


class FakeProc:
    def __init__(self, argv):
        self.argv, self.pid, self.code = argv, 999999, None

    def poll(self):
        return self.code


@pytest.fixture
def env(tmp_path, monkeypatch):
    spawned = []

    def fake_spawn(argv, log_path, cwd):
        log_path.write_text("")
        p = FakeProc([str(a) for a in argv])
        spawned.append(p)
        return p
    monkeypatch.setattr(jobs_mod, "spawn", fake_spawn)
    monkeypatch.setattr(dm_mod, "spawn", fake_spawn)

    flights = tmp_path / "flights"
    flights.mkdir()
    jm = JobManager(flights / "jobs")
    dm = DroneManager(flights)
    im = ImitationManager(flights, jobs=jm)
    monkeypatch.setattr(main, "drone_manager", dm)
    monkeypatch.setattr(main, "imitation_manager", im)
    monkeypatch.setattr(main, "job_manager", jm)
    monkeypatch.setattr(main, "LOCAL_HOSTS", main.LOCAL_HOSTS | {"testclient"})
    return dict(client=TestClient(main.app), flights=flights, spawned=spawned, dm=dm,
                im=im, jm=jm)


def make_demo(path: Path, n=30, passed=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, obs=np.zeros((n, 33), np.float32),
                        intent=np.zeros((n, 4), np.float32),
                        executed=np.zeros((n, 4), np.float32), t=np.arange(n) / 15.0,
                        episode=np.zeros(n, np.int64),
                        meta=json.dumps({"task": "onboard_hoop", "n_samples": n,
                                         "source": "human", "passed": passed}))


def make_flight(flights: Path, fid="tello_20260101-120000", **meta):
    d = flights / fid
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"flight_id": fid, "mode": "human",
                                             "dry_run": False, "final_status": "ok",
                                             **meta}))
    return d


# ------------------------------------------------------------------ flights

def test_a_real_flight_needs_confirm(env):
    r = env["client"].post("/api/drone/flights", json={"dry_run": False})
    assert r.status_code == 400 and "confirm" in r.json()["detail"]
    assert env["spawned"] == []


def test_a_confirmed_real_flight_starts_without_dry_run(env):
    r = env["client"].post("/api/drone/flights",
                           json={"dry_run": False, "confirm": True, "mode": "hover"})
    assert r.status_code == 201, r.text
    argv = env["spawned"][0].argv
    assert "--dry-run" not in argv and argv[argv.index("--mode") + 1] == "hover"
    assert "--camera" in argv


def test_a_real_flight_from_another_computer_is_refused(env, monkeypatch):
    monkeypatch.setattr(main, "LOCAL_HOSTS", {"127.0.0.1"})
    r = env["client"].post("/api/drone/flights", json={"dry_run": False, "confirm": True})
    assert r.status_code == 403
    assert env["spawned"] == []


def test_a_second_flight_is_refused_while_one_is_running(env):
    c = env["client"]
    assert c.post("/api/drone/flights", json={}).status_code == 201
    r = c.post("/api/drone/flights", json={"dry_run": False, "confirm": True})
    assert r.status_code == 409
    assert len(env["spawned"]) == 1


def test_human_mode_flies_from_the_popup_window_by_default(env):
    """The dashboard's browser -> websocket -> UDP key path lagged; the terminal's pygame window
    did not. Human flights now open that window (click it, fly from it)."""
    r = env["client"].post("/api/drone/flights", json={"mode": "human", "record": True})
    assert r.status_code == 201
    argv = env["spawned"][0].argv
    assert argv[argv.index("--keys") + 1] == "pygame" and "--keys-port" not in argv
    assert argv[argv.index("--save-answer") + 1] == "file"


def test_the_flight_gets_the_height_window_and_starts_the_camera_early(env):
    r = env["client"].post("/api/drone/flights", json={"mode": "expert", "max_height": 185,
                                                       "clearml": False})
    assert r.status_code == 201
    argv = env["spawned"][0].argv
    assert argv[argv.index("--max-height") + 1] == "185"
    assert argv[argv.index("--start-height") + 1] == "80"
    assert float(argv[argv.index("--max-airborne") + 1]) >= 120
    assert "--preview-before-takeoff" in argv and "--no-clearml" in argv
    bad = env["client"].post("/api/drone/flights", json={"min_height": 200, "max_height": 190})
    assert bad.status_code == 400


def test_human_mode_takes_keys_and_answers_from_the_dashboard(env):
    r = env["client"].post("/api/drone/flights",
                           json={"mode": "human", "record": True, "human_gain": 0.5,
                                 "keys": "browser"})
    assert r.status_code == 201
    argv = env["spawned"][0].argv
    assert argv[argv.index("--keys") + 1] == "udp"
    assert int(argv[argv.index("--keys-port") + 1]) > 0
    assert argv[argv.index("--save-answer") + 1] == "file"
    assert "--record" in argv and argv[argv.index("--human-gain") + 1] == "0.5"


def test_a_policy_flight_needs_an_existing_model(env, tmp_path):
    r = env["client"].post("/api/drone/flights",
                           json={"mode": "policy", "model": str(tmp_path / "nope.zip")})
    assert r.status_code == 400


def test_keys_are_forwarded_to_the_flights_port(env):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2)
    port = sock.getsockname()[1]
    make_flight(env["flights"], keys_port=port, final_status=None)
    with env["client"].websocket_connect("/ws/drone/tello_20260101-120000/keys") as ws:
        ws.send_json({"held": ["fb+", "takeoff"]})
        msg = json.loads(sock.recv(4096))
    assert msg == {"held": ["fb+"]}, "unknown names must be dropped"
    assert json.loads(sock.recv(4096)) == {"held": []}, "disconnect must release the keys"
    sock.close()


def _udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2)
    return sock, sock.getsockname()[1]


def test_an_unreadable_meta_does_not_drop_a_key(env, monkeypatch):
    """meta.json lives on OneDrive and is read for every key message. A read that fails
    mid-flight (a sync lock, a half-written replace) used to drop that key silently."""
    monkeypatch.setattr(dm_mod, "KEYS_META_TTL_S", 0.0)       # re-read on every key
    sock, port = _udp_listener()
    d = make_flight(env["flights"], keys_port=port, final_status=None)
    assert env["dm"].send_keys(d.name, ["fb+"])
    (d / "meta.json").write_text("{")                          # unreadable for a moment
    assert env["dm"].send_keys(d.name, ["lr-"]), "the key was dropped"
    assert json.loads(sock.recv(4096)) == {"held": ["fb+"]}
    assert json.loads(sock.recv(4096)) == {"held": ["lr-"]}
    sock.close()


def test_a_finished_flight_stops_taking_keys(env, monkeypatch):
    monkeypatch.setattr(dm_mod, "KEYS_META_TTL_S", 0.0)
    sock, port = _udp_listener()
    d = make_flight(env["flights"], keys_port=port, final_status=None)
    assert env["dm"].send_keys(d.name, ["fb+"])
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps({**meta, "final_status": "ok"}))
    assert env["dm"].send_keys(d.name, ["fb+"]) is False
    sock.close()


def test_the_save_answer_is_written_for_the_flight(env):
    d = make_flight(env["flights"])
    r = env["client"].post(f"/api/drone/flights/{d.name}/save",
                           json={"save": True, "passed": False})
    assert r.status_code == 200
    assert json.loads((d / "save_answer.json").read_text()) == {"save": True,
                                                                "passed": False}


@pytest.mark.parametrize("fid", ["..", "tello_x/../../etc", "notaflight"])
def test_flight_ids_cannot_escape_the_flights_folder(env, fid):
    r = env["client"].post(f"/api/drone/flights/{fid}/save", json={"save": True})
    assert r.status_code in (404, 405)


def test_ticks_and_events_are_served_for_history(env):
    d = make_flight(env["flights"])
    (d / "ticks.jsonl").write_text(
        "\n".join(json.dumps({"type": "tick", "t": i / 15}) for i in range(10)) + "\n")
    (d / "events.jsonl").write_text(json.dumps({"event": "takeoff_done"}) + "\n")
    c = env["client"]
    assert len(c.get(f"/api/drone/flights/{d.name}/ticks").json()["ticks"]) == 10
    assert c.get(f"/api/drone/flights/{d.name}/events").json()["events"][0][
        "event"] == "takeoff_done"


def test_the_drone_camera_frame_is_served_whole(env):
    d = make_flight(env["flights"])
    (d / "camera_latest.jpg").write_bytes(b"\xff\xd8drone")
    r = env["client"].get(f"/api/drone/flights/{d.name}/camera/latest.jpg")
    assert r.status_code == 200 and r.content == b"\xff\xd8drone"
    assert r.headers["content-type"] == "image/jpeg"


def test_the_drone_camera_frame_is_read_into_memory_not_streamed(env):
    """The regression this pins: streaming the frame with FileResponse held the file open
    while fly_hoop tried to os.replace it, and on Windows that froze the dashboard camera a
    few seconds into every flight. The body must already be in memory when the route
    returns, so no handle on the file outlives the call."""
    d = make_flight(env["flights"])
    (d / "camera_latest.jpg").write_bytes(b"\xff\xd8first")
    r = main.get_drone_camera_frame(d.name)
    assert r.body == b"\xff\xd8first"
    os.replace(_write(d / "camera_latest.jpg.tmp", b"\xff\xd8second"),
               d / "camera_latest.jpg")
    assert env["client"].get(
        f"/api/drone/flights/{d.name}/camera/latest.jpg").content == b"\xff\xd8second"


def test_a_flight_without_a_camera_frame_yet_is_a_404(env):
    d = make_flight(env["flights"])
    r = env["client"].get(f"/api/drone/flights/{d.name}/camera/latest.jpg")
    assert r.status_code == 404


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


# ------------------------------------------------------------------ imitation

def test_demos_are_listed_with_their_labels(env):
    make_demo(env["flights"] / "tello_20260101-120000" / "demo.npz", n=45)
    make_demo(env["flights"] / "tello_20260101-120100" / "demo_failed.npz")
    make_demo(env["flights"] / "demos" / "sim.npz")
    demos = {d["id"]: d for d in env["client"].get("/api/imitation/demos").json()["demos"]}
    assert demos["tello_20260101-120000/demo.npz"]["label"] == "pass"
    assert demos["tello_20260101-120000/demo.npz"]["samples"] == 45
    assert demos["tello_20260101-120100/demo_failed.npz"]["label"] == "miss"
    assert demos["demos/sim.npz"]["real"] is False


def test_relabelling_renames_the_file(env):
    make_demo(env["flights"] / "tello_20260101-120000" / "demo_failed.npz")
    r = env["client"].post("/api/imitation/demos/label",
                           json={"demo_id": "tello_20260101-120000/demo_failed.npz",
                                 "passed": True})
    assert r.json()["demo_id"] == "tello_20260101-120000/demo.npz"
    assert (env["flights"] / "tello_20260101-120000" / "demo.npz").is_file()


def test_training_builds_the_bc_command(env):
    make_demo(env["flights"] / "tello_20260101-120000" / "demo.npz")
    r = env["client"].post("/api/imitation/train", json={
        "name": "my_clone", "demo_ids": ["tello_20260101-120000/demo.npz"],
        "epochs": 50, "eval_episodes": 5})
    assert r.status_code == 202, r.text
    argv = env["spawned"][0].argv
    assert argv[1].endswith("bc_train.py")
    assert argv[argv.index("--out") + 1].endswith(str(Path("bc") / "my_clone"))
    assert argv[argv.index("--epochs") + 1] == "50"
    job = env["client"].get(f"/api/jobs/{r.json()['job_id']}").json()
    assert job["status"] == "running" and job["kind"] == "train"


def test_one_job_at_a_time(env):
    c = env["client"]
    assert c.post("/api/imitation/demos/generate", json={"name": "a"}).status_code == 202
    assert c.post("/api/imitation/demos/generate", json={"name": "b"}).status_code == 409


@pytest.mark.parametrize("name", ["../evil", "a b", "", "x" * 41])
def test_bad_names_are_refused(env, name):
    r = env["client"].post("/api/imitation/demos/generate", json={"name": name})
    assert r.status_code == 400


def test_demo_ids_cannot_escape_the_flights_folder(env):
    r = env["client"].post("/api/imitation/train", json={
        "name": "x", "demo_ids": ["../../secret.npz"]})
    assert r.status_code == 400


def test_models_list_reads_meta_and_evals(env):
    d = env["flights"] / "bc" / "c1"
    (d / "evals").mkdir(parents=True)
    (d / "bc_model.zip").write_bytes(b"zip")
    (d / "meta.json").write_text(json.dumps({"task": "onboard_hoop", "n_samples": 10}))
    (d / "evals" / "20260101-000000.json").write_text(json.dumps(
        {"passed": 6, "episodes": 12, "render": True}))
    m = env["client"].get("/api/imitation/models").json()["models"][0]
    assert m["name"] == "c1" and m["flyable"] and m["evals"][0]["passed"] == 6


def test_job_log_is_tailed_by_offset(env):
    c = env["client"]
    job = c.post("/api/imitation/demos/generate", json={"name": "a"}).json()["job_id"]
    (env["jm"].dir / f"{job}.log").write_bytes(b"line one\nline two\n")
    first = c.get(f"/api/jobs/{job}/log").json()
    assert first["text"] == "line one\nline two\n"
    assert c.get(f"/api/jobs/{job}/log?offset={first['offset']}").json()["text"] == ""


def test_jobs_are_listed_newest_first_whatever_their_kind(env):
    d = env["jm"].dir
    d.mkdir(parents=True, exist_ok=True)
    for jid in ("train_20260101-120000", "demos_20260101-130000", "eval_20260101-110000"):
        (d / f"{jid}.json").write_text(json.dumps({"job_id": jid, "returncode": 0}))
    ids = [j["job_id"] for j in env["client"].get("/api/jobs").json()["jobs"]]
    assert ids == ["demos_20260101-130000", "train_20260101-120000", "eval_20260101-110000"]


def test_policy_labels_are_readable_and_never_undefined(tmp_path):
    """The Drone tab's picker: a run's dashboard label (else its id), what it was trained for, how,
    and a score -- an RL run without evals used to print '(undefined/undefined in sim)'."""
    from imitation_manager import describe_policy
    d = tmp_path / "n6_same_from_W"
    (d / "evals").mkdir(parents=True)
    meta = {"env": "onboard", "lap_dir": "same", "laps": 0, "asym_critic": True,
            "obs_history": [0, 1, 2], "ring_tracker": True}
    plain = describe_policy(d, meta, "rl", {})
    assert plain["title"] == "n6_same_from_W" and plain["score"] == "not evaluated yet"
    assert plain["task_label"] == "same-side laps" and "asymmetric critic" in plain["method"]
    (d / "label.json").write_text(json.dumps({"label": "laps v6", "starred": True}))
    (d / "evals" / "deploy_eval.json").write_text(json.dumps(
        {"laps_per_90s": 3.4, "min_zone": 1.5, "recommended": True}))
    best = describe_policy(d, meta, "rl", {})
    assert best["title"] == "laps v6" and best["recommended"] and "3.4 laps/90s" in best["score"]
    assert "undefined" not in json.dumps([plain, best]) and "None" not in best["score"]
