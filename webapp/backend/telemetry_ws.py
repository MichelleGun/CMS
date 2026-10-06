"""
Tails a run's telemetry.jsonl and live_status.json and pushes new content
over a WebSocket, roughly matching the write rates in
scripts/telemetry_callback.py (about 12 Hz for telemetry, once per eval for
status). Polling the files rather than any inter-process signalling, which
keeps this decoupled from the training subprocess entirely, it never has to
know the backend exists.
"""
import asyncio
import json
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from config import RUNS_DIR

POLL_INTERVAL = 0.08  # seconds, ~12 Hz, matches TelemetryCallback's own rate


async def stream_run(ws: WebSocket, run_id: str):
    await ws.accept()
    run_dir = RUNS_DIR / run_id
    telemetry_path = run_dir / "telemetry.jsonl"
    status_path = run_dir / "live_status.json"
    meta_path = run_dir / "meta.json"
    log_path = run_dir / "train.log"

    last_telemetry_pos = 0
    last_status_mtime = 0.0
    last_final_status = None
    # Attach "live", like tailing a real terminal: start from current EOF,
    # not the beginning of the file, once it exists.
    last_log_pos = log_path.stat().st_size if log_path.is_file() else 0

    try:
        while True:
            if telemetry_path.is_file():
                size = telemetry_path.stat().st_size
                if size < last_telemetry_pos:
                    # file was rotated/truncated (see TelemetryCallback._rotate)
                    last_telemetry_pos = 0
                if size > last_telemetry_pos:
                    with open(telemetry_path, "r") as f:
                        f.seek(last_telemetry_pos)
                        new_lines = f.read()
                        last_telemetry_pos = f.tell()
                    for line in new_lines.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        await ws.send_json({"type": "telemetry", **record})

            if status_path.is_file():
                mtime = status_path.stat().st_mtime
                if mtime != last_status_mtime:
                    last_status_mtime = mtime
                    try:
                        status = json.loads(status_path.read_text())
                        await ws.send_json({"type": "status", **status})
                    except json.JSONDecodeError:
                        pass

            if log_path.is_file():
                size = log_path.stat().st_size
                if size < last_log_pos:
                    last_log_pos = 0
                if size > last_log_pos:
                    with open(log_path, "r", errors="replace") as f:
                        f.seek(last_log_pos)
                        new_text = f.read()
                        last_log_pos = f.tell()
                    for line in new_text.splitlines():
                        if line.strip():
                            await ws.send_json({"type": "log", "line": line})

            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text())
                except json.JSONDecodeError:
                    meta = {}
                final_status = meta.get("final_status")
                if final_status and final_status != last_final_status:
                    last_final_status = final_status
                    await ws.send_json({"type": "run_ended", "status": final_status})

            await asyncio.sleep(POLL_INTERVAL)
    except WebSocketDisconnect:
        pass
