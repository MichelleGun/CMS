"""
Tails a run's telemetry.jsonl, showcase.jsonl, live_status.json, train.log and meta.json and
pushes new content over a WebSocket, roughly matching the write rates in
scripts/telemetry_callback.py (about 15 Hz for telemetry, once per eval for status). Polling the
files rather than any inter-process signalling keeps this decoupled from the training subprocess
entirely: it never has to know the backend exists.

Robustness matters here because this one socket feeds the dashboard's far view for hours:
- On Windows a file can briefly refuse to open or read while the trainer replaces it (atomic
  os.replace writes, telemetry rotation) or while OneDrive syncs it (runs live under OneDrive).
  Any such error just skips that file for one poll; it never ends the stream.
- Only complete lines are consumed, so a half-written record is read on the next poll.
- When telemetry.jsonl is rotated (it shrinks), tailing continues from the new end instead of
  replaying the ~1800 records it kept, which were already sent.
"""
import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from config import RUNS_DIR

POLL_INTERVAL = 0.08  # seconds, ~12 Hz
log = logging.getLogger("telemetry_ws")


class _ClientGone(Exception):
    pass


class _Tail:
    """Byte-offset tail of a JSON-lines (or text) file that never raises on file access."""

    def __init__(self, path, start_at_end=True, on_shrink="end"):
        self.path = path
        self.on_shrink = on_shrink          # "end": skip rotated content, "start": re-read
        self.pos = self._size() if start_at_end else 0

    def _size(self):
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def new_lines(self):
        """Complete new lines since the last call ([] if none, or if the file is busy)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.pos:                 # rotated / truncated
            self.pos = size if self.on_shrink == "end" else 0
            if self.on_shrink == "end":
                return []
        if size == self.pos:
            return []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                chunk = f.read()
        except OSError:
            return []
        done = chunk.rfind(b"\n") + 1       # a half-written last line waits for the next poll
        self.pos += done
        return chunk[:done].decode(errors="replace").splitlines()


def _last_record(path):
    """The newest complete record of a JSON-lines file, so a fresh connection shows where the
    drone is right away instead of waiting for the next write."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


async def stream_run(ws: WebSocket, run_id: str):
    await ws.accept()
    run_dir = RUNS_DIR / run_id
    telemetry = _Tail(run_dir / "telemetry.jsonl", start_at_end=True, on_shrink="end")
    # a new showcase truncates its file first (run_manager.start_showcase): read it from the top
    showcase = _Tail(run_dir / "showcase.jsonl", start_at_end=True, on_shrink="start")
    train_log = _Tail(run_dir / "train.log", start_at_end=True, on_shrink="start")
    status_path = run_dir / "live_status.json"
    meta_path = run_dir / "meta.json"
    last_status_mtime = 0.0
    last_final_status = None

    async def send(msg):
        try:
            await ws.send_json(msg)
        except Exception as e:              # the client went away (closed tab, network)
            raise _ClientGone() from e

    def records(lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    try:
        first = _last_record(run_dir / "telemetry.jsonl")
        if first is not None:
            await send({"type": "telemetry", **first})
        while True:
            try:
                for rec in records(telemetry.new_lines()):
                    await send({"type": "telemetry", **rec})

                for rec in records(showcase.new_lines()):
                    await send({**rec, "type": "showcase"})

                try:
                    mtime = status_path.stat().st_mtime
                except OSError:
                    mtime = None
                if mtime is not None and mtime != last_status_mtime:
                    status = _read_json(status_path)
                    if status is not None:      # busy or half-written: retry next poll
                        last_status_mtime = mtime
                        await send({"type": "status", **status})

                for line in train_log.new_lines():
                    if line.strip():
                        await send({"type": "log", "line": line})

                meta = _read_json(meta_path) or {}
                final_status = meta.get("final_status")
                if final_status and final_status != last_final_status:
                    last_final_status = final_status
                    await send({"type": "run_ended", "status": final_status})
            except _ClientGone:
                raise
            except Exception:                   # never let one bad poll end the live view
                log.exception("telemetry stream for %s: poll failed, continuing", run_id)
            await asyncio.sleep(POLL_INTERVAL)
    except (WebSocketDisconnect, _ClientGone):
        pass
