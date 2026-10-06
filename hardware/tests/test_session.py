"""End-to-end: FlightSession + FakeTello + fake camera. No drone, no pybullet."""
import numpy as np

from common.config import load_config
from common.fake_tello import FakeOverheadCamera, FakeTello, SimClock
from common.flight_session import FlightSession
from common.safety import Level

CFG = load_config()


def make(profile="typical", hoop=None, seed=5, speed_scale=1.0, **prof):
    clock = SimClock()
    fake = FakeTello(CFG, profile, seed=seed, clock=clock, **prof)
    cam = FakeOverheadCamera(fake, seed=seed + 1)
    fake.connect()
    s = FlightSession(CFG, fake, cam, hoop=hoop, speed_scale=speed_scale)
    return s, fake


def settle(s, secs=2.0):
    for _ in range(int(secs * 30)):
        s.tick()


def to_flying(s, max_s=20):
    settle(s, 2.0)
    assert s.request_takeoff() == [], s.problems
    for _ in range(int(max_s * 30)):
        info = s.tick()
        if info.phase == "flying":
            return info
    raise AssertionError(f"never reached flying, phase={s.phase}")


def run(s, n, **kw):
    info = None
    for _ in range(n):
        info = s.tick(**kw)
    return info


def test_takeoff_refused_until_estimator_ready_then_allowed():
    s, fake = make()
    assert s.request_takeoff()                      # nothing seen yet -> refused
    assert fake.log == [] or all(k != "takeoff" for k, _ in fake.log)
    settle(s, 2.0)
    assert s.request_takeoff() == []


def test_full_manual_flight_with_geofence_then_safe_land():
    s, fake = make()
    to_flying(s)
    z0 = s.last_est.pos[2]
    assert z0 > 0.6
    info = run(s, 90, manual_action=np.array([1.0, 0, 0, 1.0]))     # 3 s toward +x
    assert 0.2 < fake.pos[0] < 1.0
    assert info.source == "manual" and np.isfinite(info.rc.as_tuple()).all()
    # keep pushing into the wall: the fence stops us inside the cage and nothing latches
    xmax = 0.0
    for _ in range(600):
        info = s.tick(manual_action=np.array([1.0, 0, 0, 1.0]))
        xmax = max(xmax, fake.pos[0])
    assert xmax < 1.0, xmax                     # coasts to a stop inside the wall
    assert info.decision.level < Level.LAND
    s.request_land()
    for _ in range(600):
        info = s.tick()
        if s.phase == "landed":
            break
    assert s.phase == "landed"
    assert all(k != "emergency" for k, _ in fake.log)
    assert fake.pos[2] < 0.1


def test_kill_cuts_motors_and_latches():
    s, fake = make()
    to_flying(s)
    s.request_kill()
    info = s.tick()
    assert s.phase == "killed" and info.decision.level == Level.KILL
    assert any(k == "emergency" for k, _ in fake.log)
    run(s, 120)
    assert fake.phase == "ground" and not fake.flying
    assert s.request_takeoff()                                       # latched: cannot take off again


def test_hold_toggle_zeroes_motion_and_resumes():
    s, fake = make()
    to_flying(s)
    s.toggle_hold()
    info = run(s, 30, manual_action=np.array([1.0, 0, 0, 1.0]))
    assert info.decision.level == Level.HOLD and info.rc.as_tuple()[:3] == (0, 0, 0)
    s.toggle_hold()
    info = run(s, 3, manual_action=np.array([1.0, 0, 0, 1.0]))
    assert info.decision.level == Level.NONE and info.rc.fb != 0


def test_telemetry_loss_holds_then_lands():
    s, fake = make()
    to_flying(s)
    fake.poll_telemetry = lambda: []                                 # radio link silently dies
    levels = []
    for _ in range(int(8 * 30)):
        levels.append(s.tick().decision.level)
    assert Level.HOLD in levels and Level.LAND in levels
    assert levels.index(Level.HOLD) < levels.index(Level.LAND)
    assert s.phase in ("landing", "landed", "killed")


def test_flipped_axis_is_detected_and_ends_in_landing():
    s, fake = make(axis_flip=True)
    to_flying(s)
    seen, phase = set(), None
    for _ in range(int(14 * 30)):
        info = s.tick(manual_action=np.array([0.0, 1.0, 0, 1.0]))    # command +y, the fake drone goes -y
        seen.update(info.decision.reasons)
        if s.phase in ("landing", "landed"):
            break
    assert "axis_mismatch" in seen or "axis_mismatch_persisted" in seen
    assert s.phase in ("landing", "landed")


def test_policy_reaches_hoop_and_lands_by_itself():
    hoop = np.array([0.5, 0.3, 1.0])
    s, fake = make(hoop=hoop)
    to_flying(s)

    def policy(obs):
        rel = obs[0, 72:75]
        return np.array([*rel, 1.0])

    reached = False
    for _ in range(int(40 * 30)):
        info = s.tick(policy_fn=policy)
        reached = reached or info.reached
        if s.phase == "landed":
            break
    assert reached
    assert s.phase == "landed"
    assert all(k != "emergency" for k, _ in fake.log)


def test_action_buffer_starts_at_zero_and_records_executed_actions():
    hoop = np.array([0.5, 0.3, 1.0])
    s, fake = make(hoop=hoop)
    info = to_flying(s)
    assert np.all(info.obs[0, 12:72] == 0)
    info = run(s, 5, manual_action=np.array([0.0, 1.0, 0, 0.8]))
    buf = info.obs[0, 12:72].reshape(15, 4)
    assert np.allclose(buf[-1], [0, 1, 0, 0.8], atol=1e-6) or buf[-1, 3] > 0   # newest executed action present
    assert np.all(buf[:5] == 0)                                                # oldest entries still zero


def test_policy_speed_ramps_in_from_zero():
    hoop = np.array([0.5, 0.0, 1.0])
    s, fake = make(hoop=hoop)
    to_flying(s)
    first = s.tick(policy_fn=lambda o: np.array([1.0, 0, 0, 1.0]))
    assert first.executed_action[3] < 0.2
    later = run(s, 60, policy_fn=lambda o: np.array([1.0, 0, 0, 1.0]))
    assert later.executed_action[3] > 0.9 or later.decision.level >= Level.HOLD or s.phase != "flying"
