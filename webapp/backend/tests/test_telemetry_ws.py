"""
The run WebSocket (telemetry_ws.stream_run) must keep the far view fed for hours: busy files,
rotation and half-written lines may not end it; only the client leaving does.
"""
import asyncio
import builtins
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import telemetry_ws  # noqa: E402


class FakeWS:
    def __init__(self, fail_after=None):
        self.sent, self.fail_after = [], fail_after

    async def accept(self):
        pass

    async def send_json(self, msg):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise RuntimeError("client gone")
        self.sent.append(msg)

    def telemetry_ts(self):
        return [m["t"] for m in self.sent if m["type"] == "telemetry"]


def rec(t):
    return json.dumps({"t": t, "pos": [0, 0, 0], "rpy": [0, 0, 0], "target": [0, 0, 1]}) + "\n"


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry_ws, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(telemetry_ws, "POLL_INTERVAL", 0.01)
    d = tmp_path / "r1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"final_status": None}))
    return d


async def _run_for(ws, seconds, actions=()):
    task = asyncio.create_task(telemetry_ws.stream_run(ws, "r1"))
    for delay, act in actions:
        await asyncio.sleep(delay)
        act()
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return task


def test_busy_files_do_not_end_the_stream(run, monkeypatch):
    tel = run / "telemetry.jsonl"
    tel.write_text(rec(1))
    real_open, fails = builtins.open, {"n": 0}

    def flaky_open(path, *a, **k):
        if str(path).endswith("telemetry.jsonl") and fails["n"] < 6:
            fails["n"] += 1
            raise PermissionError("being replaced")
        return real_open(path, *a, **k)
    monkeypatch.setattr(telemetry_ws, "open", flaky_open, raising=False)

    def append(t):
        return lambda: tel.open("a").write(rec(t))
    ws = FakeWS()
    task = asyncio.run(_run_for(ws, 0.3, [(0.05, append(2)), (0.1, append(3))]))
    assert fails["n"] == 6                       # the errors really happened
    assert 2 in ws.telemetry_ts() and 3 in ws.telemetry_ts()
    assert task.cancelled()                       # still running until we stopped it


def test_rotation_is_not_replayed(run):
    tel = run / "telemetry.jsonl"
    tel.write_text("".join(rec(t) for t in range(10)))

    def rotate():
        tel.write_text("".join(rec(t) for t in range(8, 12)))   # shrinks: keeps the newest
    ws = FakeWS()
    asyncio.run(_run_for(ws, 0.2, [(0.05, lambda: tel.open("a").write(rec(10))),
                                   (0.05, rotate),
                                   (0.05, lambda: tel.open("a").write(rec(12)))]))
    ts = ws.telemetry_ts()
    assert ts[0] == 9                             # newest record on connect
    assert ts == sorted(ts) and ts.count(8) == 0  # nothing from the rotated file re-sent
    assert ts[-1] == 12


def test_half_written_line_waits_for_its_end(run):
    tel = run / "telemetry.jsonl"
    tel.write_text(rec(1))
    line = rec(2)
    ws = FakeWS()
    asyncio.run(_run_for(ws, 0.15, [(0.03, lambda: tel.open("a").write(line[:10])),
                                    (0.05, lambda: tel.open("a").write(line[10:]))]))
    assert ws.telemetry_ts() == [1, 2]


def test_client_leaving_ends_the_stream(run):
    (run / "telemetry.jsonl").write_text(rec(1))
    ws = FakeWS(fail_after=0)
    task = asyncio.run(_run_for(ws, 0.1))
    assert task.done() and not task.cancelled()   # returned on its own
