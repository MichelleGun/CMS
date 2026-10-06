"""
Real Tello backend (djitellopy), exposing the same small API as FakeTello:
connect / close / now / send_rc / takeoff / land / emergency / poll_telemetry / advance.

djitellopy keeps its own receiver on UDP 8890 and exposes the latest state through
get_current_state(); it gives no per-packet arrival times. A poller thread therefore samples
that dict at ~50 Hz and queues a packet whenever the payload changed, stamping it with
time.monotonic(). The telemetry gate (state_estimation/time_sync.py) then removes
duplicates / bursts and rejects garbage.

takeoff / land block inside djitellopy while they wait for the drone's reply (up to 7 s),
so they run on worker threads: the 30 Hz safety loop must never block. `emergency` is sent
fire-and-forget (and repeated), because a drone with its motors cut may not answer.
"""
import queue
import threading
import time
from typing import List


class TelloLink:
    is_real = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tello = None
        self._q = queue.SimpleQueue()
        self._stop = threading.Event()
        self._poller = None
        self._last_payload = None
        self.connected = False
        self.log = []

    # -- lifecycle ---------------------------------------------------------------
    def connect(self):
        try:
            from djitellopy import Tello
        except ImportError as e:
            raise RuntimeError("djitellopy is not installed: pip install djitellopy "
                               "(see hardware/requirements-hardware.txt)") from e
        self.tello = Tello()
        self.tello.connect()                    # raises if no reply; PC must be on the Tello's Wi-Fi
        self.connected = True
        self._stop.clear()
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._poller.start()

    def close(self):
        self._stop.set()
        if self._poller is not None:
            self._poller.join(timeout=1.0)
        if self.tello is not None:
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                self.tello.end()
            except Exception:
                pass

    def now(self):
        return time.monotonic()

    def advance(self, dt):
        pass                                     # real time passes by itself

    # -- telemetry --------------------------------------------------------------------
    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                state = dict(self.tello.get_current_state())
            except Exception:
                state = None
            if state:
                payload = tuple(sorted((k, v) for k, v in state.items() if not isinstance(v, (list, dict))))
                if payload != self._last_payload:
                    self._last_payload = payload
                    self._q.put((time.monotonic(), state))
            time.sleep(0.02)

    def poll_telemetry(self) -> List[dict]:
        out = []
        while True:
            try:
                out.append(self._q.get_nowait()[1])
            except queue.Empty:
                break
        return out

    # -- commands -----------------------------------------------------------------------
    def send_rc(self, lr, fb, ud, yaw):
        self.log.append(("rc", (lr, fb, ud, yaw)))
        self.tello.send_rc_control(int(lr), int(fb), int(ud), int(yaw))

    def _bg(self, name, fn):
        self.log.append((name, None))

        def run():
            try:
                fn()
            except Exception as e:                # djitellopy raises on timeout / error replies
                print(f"[tello_link] {name} failed: {e}")
        threading.Thread(target=run, daemon=True).start()

    def takeoff(self):
        self._bg("takeoff", self.tello.takeoff)

    def land(self):
        self._bg("land", self.tello.land)

    def emergency(self):
        self.log.append(("emergency", None))
        for _ in range(3):                         # fire and forget; repeated against packet loss
            try:
                self.tello.send_command_without_return("emergency")
            except Exception as e:
                print(f"[tello_link] emergency send failed: {e}")
            time.sleep(0.02)
