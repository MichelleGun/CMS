import copy

import numpy as np

from common.config import load_config
from state_estimation.time_sync import TelemetryGate


def pkt(**kw):
    d = dict(pitch=0, roll=0, yaw=0, vgx=0, vgy=0, vgz=0, h=50, bat=80, tof=50, temph=60, time=1)
    d.update(kw)
    return d


def test_duplicates_dropped():
    g = TelemetryGate()
    a = g.process_batch([pkt()], 10.00)
    b = g.process_batch([pkt()], 10.02)          # identical payload 20 ms later
    assert len(a) == 1 and len(b) == 0
    assert g.stats.rejected["duplicate"] == 1


def test_burst_spread_monotonic_and_not_in_future():
    g = TelemetryGate(delay_s=0.0)
    batch = [pkt(yaw=i) for i in range(4)]       # four packets arrive together
    out = g.process_batch(batch, 10.0)
    ts = [s.t for s in out]
    assert len(out) == 4
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert max(ts) <= 10.0 + 1e-9
    assert abs((ts[-1] - ts[0]) - 0.3) < 0.02    # spread over the nominal 100 ms grid


def test_time_never_goes_backwards_across_batches():
    g = TelemetryGate(delay_s=0.0)
    g.process_batch([pkt(yaw=1), pkt(yaw=2), pkt(yaw=3)], 10.0)
    out = g.process_batch([pkt(yaw=4)], 10.01)   # late arrival right behind a burst
    last = g.last.t
    assert all(s.t > 9.7 for s in out)
    assert last >= 9.9 or not out


def test_garbage_rejected_with_reason():
    g = TelemetryGate()
    bad = [
        {"pitch": "x"},                          # unparsable / missing fields
        pkt(bat=250),
        pkt(pitch=170),
        pkt(vgx=100),                            # 10 m/s at 0.1 m/s per unit
        pkt(h=99999),
    ]
    for i, b in enumerate(bad):
        assert g.process_batch([b], 20.0 + i) == []
    assert g.stats.accepted == 0
    assert set(g.stats.rejected) >= {"parse", "battery_range", "attitude_range", "speed_range", "height_range"}


def test_accel_jump_rejected():
    g = TelemetryGate(delay_s=0.0)
    assert g.process_batch([pkt(vgx=0)], 1.0)
    assert g.process_batch([pkt(vgx=25, h=51)], 1.1) == []     # 2.5 m/s in 0.1 s = 25 m/s^2
    assert g.stats.rejected["accel_jump"] == 1


def test_stale_age():
    g = TelemetryGate()
    assert g.age(5.0) == float("inf")
    g.process_batch([pkt()], 5.0)
    assert abs(g.age(5.7) - 0.7) < 1e-9


def test_timestamps_never_closer_than_min_spacing_regression():
    """Float-epsilon spacing once made d(yaw)/dt explode to ~1e10 and saturate the yaw loop."""
    from state_estimation.time_sync import MIN_SPACING
    g = TelemetryGate(delay_s=0.0)
    ts = []
    for k in range(50):
        out = g.process_batch([pkt(yaw=k % 7, h=50 + k % 3)], 10.0 + k * 0.1)
        out += g.process_batch([pkt(yaw=(k + 3) % 7, h=51 + k % 3)], 10.0 + k * 0.1 + 1e-9)
        ts += [s.t for s in out]
    gaps = np.diff(ts)
    assert (gaps >= MIN_SPACING - 1e-12).all()
