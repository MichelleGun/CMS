"""TelemetryCallback: episode counter, contact flags and the always-written final record."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from telemetry_callback import TelemetryCallback  # noqa: E402


def _step(cb, obs, info, done):
    cb.locals = {"new_obs": np.array([obs]), "infos": [info], "dones": np.array([done])}
    cb.num_timesteps += 1
    return cb._on_step()


def test_episode_end_is_always_written_with_ep_and_success(tmp_path):
    cb = TelemetryCallback(run_dir=tmp_path, hz=0.001)        # throttle: ~nothing but forced records
    cb.num_timesteps = 0
    obs = np.zeros(78, dtype=np.float32)
    obs[-3:] = [0.0, 0.0, 1.0]
    term = obs.copy()
    term[0:3] = [0.5, 0.0, 1.0]
    _step(cb, obs, {"net_contact": False}, False)              # first record (not throttled yet)
    _step(cb, obs, {"net_contact": True}, False)               # throttled, but the hit is remembered
    _step(cb, obs, {"is_success": True, "passes": 1, "terminal_observation": term}, True)
    _step(cb, obs, {}, False)                                  # next episode, throttled
    recs = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    end = recs[-1]
    assert end["done"] and end["success"] and end["ep"] == 0 and end["passes"] == 1
    assert end["net_contact"] is True                          # carried over from the skipped step
    assert end["pos"] == [0.5, 0.0, 1.0]                       # terminal obs, not the reset obs
    assert "hoop_normal" in end                                # 78-D obs -> ring normal included
    assert cb._ep == 1 and cb._ep_step == 1


def test_pace_callback_only_sleeps_while_the_flag_exists(tmp_path):
    from telemetry_callback import PaceCallback
    clock = {"t": 0.0}
    slept = []

    def sleep(dt):
        slept.append(dt)
        clock["t"] += dt
    cb = PaceCallback(run_dir=tmp_path, clock=lambda: clock["t"], sleep=sleep, check_every=0.0)
    for _ in range(5):                       # no flag: full speed
        clock["t"] += 0.001
        cb._on_step()
    assert slept == []
    (tmp_path / "realtime.flag").touch()
    for _ in range(5):                       # flag: each step padded to 1/30 s
        clock["t"] += 0.001
        cb._on_step()
    assert len(slept) == 4 and all(abs(s - (1 / 30 - 0.001)) < 1e-9 for s in slept)
    (tmp_path / "realtime.flag").unlink()
    n = len(slept)
    for _ in range(5):
        clock["t"] += 0.001
        cb._on_step()
    assert len(slept) == n


def test_curriculum_shrinks_over_its_own_horizon():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_hoop_multi import RadiusCurriculumCallback
    cb = RadiusCurriculumCallback(horizon=1000, start=1.5)
    for t, want in [(0, 1.5), (500, 1.25), (1000, 1.0), (10 ** 9, 1.0)]:
        cb.num_timesteps = t
        assert cb._scale() == want
