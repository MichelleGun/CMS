"""
The human-at-the-keyboard controller and the post-landing save prompt.

Key sources are scripted, so nothing here needs pygame or a window. The flight tests run
the real loop against sim_drone.SimDrone, like test_fly_hoop.py.
"""
import numpy as np
import pytest

from onboard import spec
from onboard.expert import DONE
from onboard.fly_hoop import Flight, parse
from onboard.human import HumanController
from onboard.kit import FlightError, Settings
from onboard.ring_detect import RingObs
from onboard.sim_drone import SimDrone

pytestmark = pytest.mark.usefixtures("fresh_telemetry", "fast_clock")

NO_RING = RingObs()


class ScriptedKeys:
    """poll() -> (held, pressed). `held` for `n` polls, then `then` pressed once."""

    def __init__(self, held=(), n=10 ** 9, then=None):
        self.held, self.n, self.then = set(held), n, then
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self.polls <= self.n:
            return set(self.held), []
        return set(), [self.then] if self.then and self.polls == self.n + 1 else []


def act(held, gain=1.0):
    return HumanController(ScriptedKeys(held), gain=gain)(NO_RING)[0]


# ------------------------------------------------------------- key -> action mapping

@pytest.mark.parametrize("key,channel,sign", [
    ("fb+", spec.ACT_FB, +1), ("fb-", spec.ACT_FB, -1),
    ("lr+", spec.ACT_LR, +1), ("lr-", spec.ACT_LR, -1),
    ("ud+", spec.ACT_UD, +1), ("ud-", spec.ACT_UD, -1),
    ("yaw+", spec.ACT_YAW, +1), ("yaw-", spec.ACT_YAW, -1),
])
def test_each_key_drives_exactly_its_own_channel(key, channel, sign):
    a = act({key})
    assert a[channel] == sign
    assert np.count_nonzero(a) == 1


def test_no_key_is_a_hover():
    assert not act(set()).any()


def test_opposing_keys_cancel():
    assert act({"fb+", "fb-", "lr+", "lr-"}).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_gain_scales_a_held_key():
    assert act({"fb+"}, gain=0.6)[spec.ACT_FB] == pytest.approx(0.6)


@pytest.mark.parametrize("gain", [0.0, -0.5, 1.5])
def test_a_nonsense_gain_is_refused(gain):
    with pytest.raises(ValueError):
        HumanController(ScriptedKeys(), gain=gain)


def test_the_rotate_keys_can_run_faster_than_the_others():
    c = HumanController(ScriptedKeys({"yaw+", "fb+"}), gain=0.6, yaw_gain=1.0)
    a, _ = c(NO_RING)
    assert a[spec.ACT_YAW] == pytest.approx(1.0) and a[spec.ACT_FB] == pytest.approx(0.6)
    assert spec.action_to_rc(a)[3] == spec.RC_MAX, "full yaw is the clamp, not beyond it"


def test_yaw_follows_the_main_gain_unless_told_otherwise():
    a, _ = HumanController(ScriptedKeys({"yaw-"}), gain=0.5)(NO_RING)
    assert a[spec.ACT_YAW] == pytest.approx(-0.5)


@pytest.mark.parametrize("g", [0.0, -1.0, 1.5])
def test_a_nonsense_yaw_gain_is_refused(g):
    with pytest.raises(ValueError):
        HumanController(ScriptedKeys(), yaw_gain=g)


def test_land_key_finishes_and_zeroes_the_action():
    c = HumanController(ScriptedKeys({"fb+"}, n=0, then="land"), gain=1.0)
    a, info = c(NO_RING)
    assert c.finished and info.phase == DONE
    c.keys = ScriptedKeys({"fb+"})
    assert not c(NO_RING)[0].any()


def test_kill_key_latches_and_zeroes_the_action():
    c = HumanController(ScriptedKeys({"fb+"}, n=0, then="kill"))
    a, _ = c(NO_RING)
    assert c.kill_requested and not a.any()


def test_a_key_pressed_before_takeoff_is_not_forgotten():
    c = HumanController(ScriptedKeys(n=0, then="land"))
    c.pump()
    assert c.finished


# ------------------------------------------------------------- the real flight loop

def flight(tmp_path, keys, *extra, answers=()):
    from imitation.record_demo import DemoRecorder
    a = parse(["--dry-run", "--mode", "human", "--countdown", "0", "--settle", "0.2",
               "--record", *extra])
    d = tmp_path / "flight"
    d.mkdir(parents=True, exist_ok=True)
    f = Flight(a, Settings(), d, keys=keys)
    f.recorder = DemoRecorder(meta={"source": "human"}, obs_dim=spec.OBS_DIM,
                              task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    asked = []
    queue = list(answers)

    def ask(prompt):
        asked.append(prompt)
        return queue.pop(0)
    f._ask = ask
    return f, SimDrone(profile=a.profile, seed=0, detector=f.detector), d, asked


def fly_forward(tmp_path, answers):
    keys = ScriptedKeys({"fb+"}, n=200, then="land")
    f, drone, d, asked = flight(tmp_path, keys, answers=answers)
    return f, drone, d, asked, f.run(drone)


@pytest.mark.parametrize("argv, expected", [
    (["--mode", "human"], True),                            # the case that was silently skipped
    (["--mode", "human", "--record"], True),
    (["--mode", "human", "--no-record"], False),            # a practice flight
    (["--mode", "human", "--record", "--no-record"], False),
    (["--mode", "expert"], False),
    (["--mode", "expert", "--record"], True),
])
def test_a_human_flight_is_recorded_unless_opted_out(argv, expected):
    from onboard.fly_hoop import wants_recording
    assert wants_recording(parse(["--dry-run", *argv])) is expected


def test_the_dashboard_chooses_recording_explicitly(tmp_path):
    """The dashboard's checkbox must still decide: human flights now record by default."""
    from drone_manager import DroneManager
    dm = DroneManager(tmp_path)
    assert "--no-record" in dm.build_argv(mode="human", record=False, dry_run=True)
    assert "--record" in dm.build_argv(mode="human", record=True, dry_run=True)


def test_a_flight_that_never_saw_the_ring_says_so(tmp_path, capsys):
    keys = ScriptedKeys(n=5, then="land")
    f, drone, d, asked = flight(tmp_path, keys)
    f._save_demo({"outcome": "human_landed"}, aborted=False)      # nothing recorded
    assert asked == []
    assert "ring was never detected" in capsys.readouterr().out


def test_a_human_flight_sends_the_keys_as_rc_and_lands(tmp_path):
    f, drone, d, asked, res = fly_forward(tmp_path, ["n"])
    assert "land" in drone.calls and res["outcome"] == "human_landed"
    forward = [rc[1] for rc in drone.rc_calls()]
    assert max(forward) == round(0.6 * spec.RC_MAX)


def test_a_passed_run_is_saved_as_demo_npz_with_the_blind_ticks(tmp_path):
    from imitation.record_demo import load_demo
    f, drone, d, asked, res = fly_forward(tmp_path, ["y", "y"])
    assert (d / "demo.npz").is_file() and not (d / "demo_failed.npz").exists()
    demo = load_demo(d / "demo.npz")
    assert demo["meta"]["passed"] is True and demo["meta"]["source"] == "human"
    found = demo["obs"][:, spec.OBS_SLICES["found"]].ravel()
    assert found[0] == 1.0, "recording should start at the first lock"
    assert (found == 0.0).any(), "the blind push after the ring leaves view was dropped"


def test_the_pass_count_is_asked_for_and_recorded(tmp_path):
    from imitation.record_demo import load_demo
    f, drone, d, asked, res = fly_forward(tmp_path, ["y", "3"])
    assert "How many times" in asked[1]
    assert res["passes"] == 3
    assert (d / "demo.npz").is_file()
    assert load_demo(d / "demo.npz")["meta"]["passes"] == 3


def test_zero_passes_is_a_miss_and_nonsense_is_asked_again(tmp_path):
    f, drone, d, asked, res = fly_forward(tmp_path, ["y", "lots", "-2", "0"])
    assert len(asked) == 4, "should have re-asked after each unusable answer"
    assert res["passes"] == 0 and (d / "demo_failed.npz").is_file()


def test_a_missed_run_is_kept_apart_from_the_training_glob(tmp_path):
    f, drone, d, asked, res = fly_forward(tmp_path, ["y", "n"])
    assert (d / "demo_failed.npz").is_file() and not (d / "demo.npz").exists()


def test_declining_to_save_writes_nothing(tmp_path):
    f, drone, d, asked, res = fly_forward(tmp_path, ["n"])
    assert not (d / "demo.npz").exists() and not (d / "demo_failed.npz").exists()
    assert len(asked) == 1, "should not ask whether it passed a run that is being dropped"


def test_the_prompt_comes_only_after_the_drone_has_landed(tmp_path):
    f, drone, d, asked = flight(tmp_path, ScriptedKeys({"fb+"}, n=50, then="land"))
    seen = []
    f._ask = lambda prompt: seen.append(list(drone.calls)) or "n"
    f.run(drone)
    assert seen and "land" in seen[0], "asked to save while the drone was still flying"


def test_ctrl_c_at_the_prompt_discards_the_run(tmp_path):
    keys = ScriptedKeys({"fb+"}, n=60, then="land")
    f, drone, d, asked = flight(tmp_path, keys)

    def interrupt(prompt):
        raise KeyboardInterrupt
    f._ask = interrupt
    f.run(drone)
    assert not (d / "demo.npz").exists() and not (d / "demo_failed.npz").exists()


def test_a_motor_cut_is_never_offered_for_saving(tmp_path):
    keys = ScriptedKeys({"fb+"}, n=60, then="kill")
    f, drone, d, asked = flight(tmp_path, keys)
    res = f.run(drone)
    assert res["outcome"] == "killed" and "emergency" in drone.calls
    assert asked == []
    assert not (d / "demo.npz").exists()


def test_landing_before_takeoff_stops_the_flight_on_the_ground(tmp_path):
    keys = ScriptedKeys(n=0, then="land")
    f, drone, d, asked = flight(tmp_path, keys, "--countdown", "1")
    with pytest.raises(FlightError):
        f.run(drone)
    assert "takeoff" not in drone.calls


def test_the_save_prompt_is_human_mode_only(tmp_path):
    from imitation.record_demo import DemoRecorder
    a = parse(["--dry-run", "--countdown", "0", "--settle", "0.2", "--max-airborne", "6"])
    d = tmp_path / "expert"
    d.mkdir()
    f = Flight(a, Settings(), d)
    f.recorder = DemoRecorder(meta={"source": "t"}, obs_dim=spec.OBS_DIM,
                              task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    f._ask = lambda prompt: pytest.fail("an expert flight must not prompt")
    f.run(SimDrone(profile=a.profile, seed=0, detector=f.detector))
    assert (d / "demo.npz").is_file()


def test_the_cli_refuses_a_nonsense_gain():
    with pytest.raises(SystemExit):
        parse(["--mode", "human", "--human-gain", "2"])


# ------------------------------------------------------------- dashboard key source

def _send(port, held):
    import json
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(json.dumps({"held": list(held)}).encode(), ("127.0.0.1", port))
    s.close()


def _drain(keys, tries=200):
    import time
    for _ in range(tries):
        held, _ = keys.poll()
        if held:
            return held
        time.sleep(0.005)
    return set()


def test_udp_keys_deliver_held_keys():
    from onboard.human import UdpKeys
    k = UdpKeys(0)
    try:
        _send(k.port, ["fb+", "yaw-"])
        assert _drain(k) == {"fb+", "yaw-"}
    finally:
        k.close()


def test_udp_keys_drop_unknown_names():
    from onboard.human import UdpKeys
    k = UdpKeys(0)
    try:
        _send(k.port, ["fb+", "takeoff", "kill"])
        assert _drain(k) == {"fb+"}
    finally:
        k.close()


def test_udp_keys_hover_when_the_dashboard_goes_quiet():
    from onboard.human import UdpKeys
    now = [0.0]
    k = UdpKeys(0, stale_s=0.4, clock=lambda: now[0])
    try:
        _send(k.port, ["fb+"])
        assert _drain(k) == {"fb+"}
        now[0] += 0.5
        assert k.poll()[0] == set(), "a stale key must not stay held"
    finally:
        k.close()


def test_a_udp_tap_between_two_polls_still_counts_for_one_tick():
    """The dashboard sends a datagram on key down and another on key up. A tap shorter
    than a tick puts both between two polls; keeping only the last datagram lost it."""
    import time
    from onboard.human import UdpKeys
    k = UdpKeys(0)
    try:
        _send(k.port, ["fb+"])
        _send(k.port, [])
        time.sleep(0.05)                       # both datagrams land before the poll
        assert k.poll()[0] == {"fb+"}, "a quick tap was dropped"
        assert k.poll()[0] == set(), "the release must apply on the next tick"
    finally:
        k.close()


def test_udp_keys_hold_nothing_before_the_first_packet():
    from onboard.human import UdpKeys
    k = UdpKeys(0)
    try:
        assert k.poll() == (set(), [])
    finally:
        k.close()


def test_the_cli_needs_a_port_for_udp_keys():
    with pytest.raises(SystemExit):
        parse(["--mode", "human", "--keys", "udp"])


# ------------------------------------------------------------- dashboard save answer

def dashboard_flight(tmp_path, answer):
    import json
    from imitation.record_demo import DemoRecorder
    a = parse(["--dry-run", "--mode", "human", "--countdown", "0", "--settle", "0.2",
               "--record", "--save-answer", "file"])
    d = tmp_path / "flight"
    d.mkdir()
    if answer is not None:
        (d / "save_answer.json").write_text(json.dumps(answer), encoding="utf-8")
    f = Flight(a, Settings(), d, keys=ScriptedKeys({"fb+"}, n=200, then="land"))
    f.recorder = DemoRecorder(meta={"source": "human"}, obs_dim=spec.OBS_DIM,
                              task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    f.run(SimDrone(profile=a.profile, seed=0, detector=f.detector))
    return d


@pytest.mark.parametrize("answer,saved", [
    ({"save": True, "passed": True}, "demo.npz"),
    ({"save": True, "passed": False}, "demo_failed.npz"),
    ({"save": False, "passed": True}, None),
])
def test_the_dashboard_answer_decides_how_the_run_is_saved(tmp_path, answer, saved):
    d = dashboard_flight(tmp_path, answer)
    files = {p.name for p in d.glob("demo*.npz")}
    assert files == ({saved} if saved else set())


def test_waiting_for_a_save_is_announced_to_the_dashboard(tmp_path):
    import json
    d = dashboard_flight(tmp_path, {"save": False})
    events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "awaiting_save" and e["samples"] > 0 for e in events)
    assert "land_complete" in [e["event"] for e in events][
        :[e["event"] for e in events].index("awaiting_save")], "asked before landing"


def test_no_answer_discards_the_run(tmp_path):
    from imitation.record_demo import DemoRecorder
    a = parse(["--dry-run", "--mode", "human", "--record", "--save-answer", "file"])
    d = tmp_path / "f"
    d.mkdir()
    f = Flight(a, Settings(), d, keys=ScriptedKeys())
    f.recorder = DemoRecorder(meta={}, obs_dim=spec.OBS_DIM, task=spec.TASK_NAME,
                              obs_shape=(spec.OBS_DIM,))
    assert f._wait_for_save_answer(timeout_s=0.5, poll_s=0.01) == {
        "save": False, "passed": False}


# ------------------------------------------------------------- eval_policy.py

def test_eval_policy_writes_its_result_next_to_the_model(tmp_path):
    import json
    from imitation.bc_train import build_model
    from onboard import eval_policy
    m = build_model("ppo", 0, obs_shape=(spec.OBS_DIM,), act_shape=(spec.ACT_DIM,))
    p = tmp_path / "bc_model.zip"
    m.save(str(p))
    res = eval_policy.main(["--model", str(p), "--episodes", "2"])
    saved = list((tmp_path / "evals").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["episodes"] == res["episodes"] == 2
