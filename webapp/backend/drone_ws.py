"""
Tails a flight's ticks.jsonl, status.json and events.jsonl and pushes new content over a
WebSocket.

Same shape as telemetry_ws.py: byte-offset tailing, flat messages discriminated by a
"type" field, and polling the files rather than any inter-process signalling, so
hardware/onboard/fly_hoop.py never has to know the backend exists. That matters more here
than on the training side: the flight script has to be runnable from a bare terminal with
no dashboard at all, because that is how every real flight is started.

The poll rate matches the flight's own 15 Hz control tick. Rows are forwarded almost
verbatim, since fly_hoop.py already writes them with a "type" discriminator.
"""
import asyncio
import json
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from config import FLIGHTS_DIR

POLL_INTERVAL = 0.066          # ~15 Hz, matching spec.CTRL_HZ
MAX_TICKS_PER_POLL = 40        # a backlog is drained steadily, not in one burst


async def _tail(path: Path, pos: int, send, kind: str, limit: int = 0) -> int:
    """Forward new whole lines from `path`, returning the new byte offset."""
    if not path.is_file():
        return pos
    size = path.stat().st_size
    if size < pos:
        return 0               # truncated or replaced
    if size == pos:
        return pos
    with open(path, "r", errors="replace") as f:
        f.seek(pos)
        text = f.read()
        pos = f.tell()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if limit and len(lines) > limit:
        lines = lines[-limit:]          # fell behind: the newest state is what matters
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue                     # a partially flushed line; it will be re-read
        await send({"type": kind, **record} if "type" not in record
                   else {**record, "type": kind})
    return pos


async def stream_flight(ws: WebSocket, flight_id: str):
    await ws.accept()
    d = FLIGHTS_DIR / flight_id
    ticks_path = d / "ticks.jsonl"
    events_path = d / "events.jsonl"
    status_path = d / "status.json"
    meta_path = d / "meta.json"

    tick_pos = 0
    event_pos = 0
    last_status_mtime = 0.0
    last_final_status = None

    try:
        while True:
            tick_pos = await _tail(ticks_path, tick_pos, ws.send_json, "drone_tick",
                                   limit=MAX_TICKS_PER_POLL)
            event_pos = await _tail(events_path, event_pos, ws.send_json, "drone_event")

            if status_path.is_file():
                mtime = status_path.stat().st_mtime
                if mtime != last_status_mtime:
                    last_status_mtime = mtime
                    try:
                        status = json.loads(status_path.read_text(encoding="utf-8"))
                        await ws.send_json({"type": "drone_status", **status})
                    except (OSError, json.JSONDecodeError):
                        pass          # caught mid atomic-replace; the next poll gets it

            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    meta = {}
                final = meta.get("final_status")
                if final and final != last_final_status:
                    last_final_status = final
                    await ws.send_json({"type": "drone_ended", "status": final})

            await asyncio.sleep(POLL_INTERVAL)
    except WebSocketDisconnect:
        pass
