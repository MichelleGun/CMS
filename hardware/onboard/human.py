"""
A human at the keyboard as a drop-in controller, so their flying can be recorded and cloned.

It has the same call signature as the scripted Expert -- (ring, state) -> (action, info) --
so the flight loop, the safety layers and the demonstration recorder treat it identically,
and the recorded observations are the ones a trained policy will see.

Keys (the small window must have focus; Ctrl+C in the terminal still lands):
    W / S              forward / back
    A / D              STRAFE left / right (slide sideways, nose keeps pointing ahead)
    Up / Down arrow    up / down
    Left / Right arrow ROTATE left / right (turn in place)
    L or Esc           land
    X or Backspace     emergency motor cut

The pilot watches the drone, not the screen, so the window only carries a status readout.
"""
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from . import spec
from .expert import DONE

HUMAN = "human"
# Logical names a key source reports: "<channel>+" / "<channel>-" while held.
CHANNELS = (("lr", spec.ACT_LR), ("fb", spec.ACT_FB), ("ud", spec.ACT_UD),
            ("yaw", spec.ACT_YAW))


@dataclass
class HumanInfo:
    phase: str = HUMAN
    reason: str = ""


class PygameKeys:
    """Held-key state from a small pygame window. pygame is imported lazily."""

    HELD = {"K_w": "fb+", "K_s": "fb-", "K_d": "lr+", "K_a": "lr-",
            "K_UP": "ud+", "K_DOWN": "ud-", "K_RIGHT": "yaw+", "K_LEFT": "yaw-"}
    ONCE = {"K_l": "land", "K_ESCAPE": "land", "K_x": "kill", "K_BACKSPACE": "kill"}

    def __init__(self, title: str = "Tello human control", size=(520, 250)):
        try:
            import pygame
        except ImportError as e:
            raise RuntimeError(
                "--mode human needs pygame. Install it once with:\n"
                "    & $PY -m pip install pygame") from e
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        self._held = {getattr(pygame, k): v for k, v in self.HELD.items()}
        self._once = {getattr(pygame, k): v for k, v in self.ONCE.items()}

    def poll(self) -> Tuple[Set[str], List[str]]:
        pg = self.pg
        held: Set[str] = set()
        pressed: List[str] = []
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pressed.append("land")
            elif e.type == pg.KEYDOWN:
                # A tap shorter than one 15 Hz tick is gone by the time get_pressed() is
                # read, so a key-down event counts as held for this tick.
                if e.key in self._held:
                    held.add(self._held[e.key])
                if e.key in self._once:
                    pressed.append(self._once[e.key])
        keys = pg.key.get_pressed()
        for code, logical in self._held.items():
            if keys[code]:
                held.add(logical)
        return held, pressed

    def show(self, lines: List[str], alert: bool = False):
        pg = self.pg
        self.screen.fill((70, 10, 10) if alert else (22, 22, 33))
        y = 8
        for i, line in enumerate(lines):
            font = self.big if i == 0 else self.font
            colour = (255, 120, 90) if alert and i == 0 else (230, 230, 230)
            self.screen.blit(font.render(line, True, colour), (10, y))
            y += 30 if i == 0 else 20
        pg.display.flip()

    def close(self):
        self.pg.quit()


class UdpKeys:
    """Held keys sent by the dashboard as UDP datagrams {"held": [...]} on 127.0.0.1.

    A deadman: if no datagram arrives for `stale_s`, nothing is held and the drone hovers.
    The browser sends a heartbeat while a flight is live, so a closed tab, a crashed
    backend or a lost socket all end in a hover rather than a key stuck down. Land and
    kill come through the flag files, like every other flight, not through this socket.
    """

    def __init__(self, port: int, stale_s: float = 0.4, clock=None):
        import socket
        import time
        self._clock = clock or time.monotonic
        self.stale_s = float(stale_s)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", int(port)))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self._held: Set[str] = set()
        self._last = None

    def poll(self) -> Tuple[Set[str], List[str]]:
        import json
        while True:
            try:
                data = self.sock.recv(4096)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:            # Windows: ICMP port-unreachable surfaces here
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                names = msg.get("held", [])
            except (ValueError, AttributeError, UnicodeDecodeError):
                continue
            valid = set(PygameKeys.HELD.values())
            self._held = {n for n in names if n in valid}
            self._last = self._clock()
        if self._last is None or self._clock() - self._last > self.stale_s:
            return set(), []
        return set(self._held), []

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class HumanController:
    """Maps held keys to the 4-channel action. `keys.poll()` -> (held, pressed)."""

    def __init__(self, keys, gain: float = 0.6):
        if not 0.0 < gain <= 1.0:
            raise ValueError("gain must be in (0, 1]")
        self.keys = keys
        self.gain = float(gain)
        self.phase = HUMAN
        self.reason = ""
        self.kill_requested = False
        self.status_extra = ""
        self._land = False

    @property
    def finished(self) -> bool:
        return self._land

    def reset(self):
        self.phase = HUMAN
        self.reason = ""
        self.kill_requested = False
        self._land = False

    def _read(self) -> Set[str]:
        held, pressed = self.keys.poll()
        if "kill" in pressed:
            self.kill_requested = True
            self.reason = "emergency cut"
        if "land" in pressed and not self._land:
            self._land = True
            self.phase = DONE
            self.reason = "landing requested"
        return held

    def pump(self):
        """Keep the window alive (and latch land/kill) before the controller is engaged."""
        self._read()
        self._show(None, np.zeros(spec.ACT_DIM), waiting=True)

    def __call__(self, ring, state: Optional[dict] = None):
        held = self._read()
        a = np.zeros(spec.ACT_DIM)
        if not (self._land or self.kill_requested):
            for name, idx in CHANNELS:
                a[idx] = self.gain * ((f"{name}+" in held) - (f"{name}-" in held))
        self._show(ring, a, waiting=False)
        return a, HumanInfo(phase=self.phase, reason=self.reason)

    def _show(self, ring, action, waiting: bool):
        if not hasattr(self.keys, "show"):
            return
        if waiting:
            seen = "waiting for takeoff"
        elif ring is not None and ring.found:
            rng = f"{ring.range_m:.2f} m" if np.isfinite(ring.range_m) else "--"
            seen = (f"RING  range {rng}  cx {ring.cx:+.2f}  cy {ring.cy:+.2f}  "
                    f"axis {ring.axis_ratio:.2f}")
        else:
            seen = "ring not in view"
        rc = spec.action_to_rc(action)
        self.keys.show([
            "LANDING" if self._land else "HUMAN FLIGHT",
            seen,
            f"rc lr {rc[0]:+d}  fb {rc[1]:+d}  ud {rc[2]:+d}  yaw {rc[3]:+d}"
            + (f"   {self.status_extra}" if self.status_extra else ""),
            "W/S fwd/back   A/D strafe   Up/Down height   Left/Right rotate",
            "L / Esc: land      X / Backspace: EMERGENCY motor cut",
        ], alert=self.kill_requested)

    def close(self):
        if hasattr(self.keys, "close"):
            self.keys.close()
