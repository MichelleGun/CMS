"""
Input sources for the control loop. The loop only sees logical names, so tests can drive it with
ScriptedInput and the real scripts use PygameInput (keyboard window).

Logical names
  held (movement, world/cage frame; yaw is held by the controller):
      x+ x- y+ y- z+ z-
  pressed (one-shot):
      takeoff land kill hold quit speed1 speed2 speed3 record

Key map (PygameInput; the window must have focus, the global hotkeys in common/safety.py do not need it):
  T takeoff        L safe land        X / BACKSPACE  EMERGENCY motor cut        SPACE hold/hover (toggle)
  ESC land + quit  1/2/3 speed 40/70/100 %          R toggle demo recording
  Arrow keys: LEFT/RIGHT = x-/x+,  UP/DOWN = y+/y-      W/S = up/down (z+/z-)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set

MOVES = ("x+", "x-", "y+", "y-", "z+", "z-")


@dataclass
class InputState:
    held: Set[str] = field(default_factory=set)
    pressed: List[str] = field(default_factory=list)


class ScriptedInput:
    """events: {tick_index: {"press": [...], "hold": [...]}}; `hold` lasts until changed."""

    def __init__(self, events: Dict[int, dict]):
        self.events = events
        self.n = 0
        self._held: Set[str] = set()

    def poll(self) -> InputState:
        ev = self.events.get(self.n, {})
        if "hold" in ev:
            self._held = set(ev["hold"])
        self.n += 1
        return InputState(set(self._held), list(ev.get("press", [])))

    def close(self):
        pass


class NullInput:
    def poll(self) -> InputState:
        return InputState()

    def close(self):
        pass


class PygameInput:
    """Keyboard window with a status panel. pygame is imported lazily."""

    HELD = {"LEFT": "x-", "RIGHT": "x+", "UP": "y+", "DOWN": "y-", "w": "z+", "s": "z-"}
    ONCE = {"t": "takeoff", "l": "land", "x": "kill", "BACKSPACE": "kill", "SPACE": "hold",
            "ESCAPE": "quit", "1": "speed1", "2": "speed2", "3": "speed3", "r": "record"}

    def __init__(self, title="Tello control", size=(560, 330)):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        self._lines: List[str] = []
        self._color = (230, 230, 230)

    def poll(self) -> InputState:
        pg = self.pg
        st = InputState()
        for e in pg.event.get():
            if e.type == pg.QUIT:
                st.pressed.append("quit")
            elif e.type == pg.KEYDOWN:
                name = pg.key.name(e.key)
                key = name.upper() if name in ("escape", "backspace", "space") else name
                key = {"ESCAPE": "ESCAPE", "BACKSPACE": "BACKSPACE", "SPACE": "SPACE"}.get(key, key)
                if key in self.ONCE:
                    st.pressed.append(self.ONCE[key])
        keys = pg.key.get_pressed()
        for name, logical in self.HELD.items():
            code = getattr(pg, f"K_{name}", None) or getattr(pg, f"K_{name.lower()}")
            if keys[code]:
                st.held.add(logical)
        return st

    def show(self, lines: List[str], alert=False):
        pg = self.pg
        self.screen.fill((70, 10, 10) if alert else (22, 22, 33))
        y = 8
        for i, ln in enumerate(lines):
            f = self.big if i == 0 else self.font
            self.screen.blit(f.render(ln, True, (255, 120, 90) if alert and i == 0 else self._color), (10, y))
            y += 30 if i == 0 else 20
        pg.display.flip()

    def close(self):
        self.pg.quit()
