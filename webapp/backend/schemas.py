"""Pydantic request/response models for the dashboard API."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float
    y: float
    z: float

    def as_list(self):
        return [self.x, self.y, self.z]


class StartRunRequest(BaseModel):
    algo: str
    seed: Optional[int] = None
    start_pos: Optional[Position] = None
    target_pos: Optional[Position] = None
    timesteps: int = 1_000_000
    n_envs: int = 4
    eval_freq: int = 2000
    n_eval_episodes: int = 10


class StartRunResponse(BaseModel):
    run_id: str
    status: str


class ReplayRequest(BaseModel):
    variant: Literal["current", "best"] = "current"
    episodes: int = 3
    pov: bool = False


class ReplayResponse(BaseModel):
    replay_id: str
    status: str


class CameraToggleRequest(BaseModel):
    enabled: bool


# --- real drone (hardware/onboard/fly_hoop.py) ------------------------------

class StartFlightRequest(BaseModel):
    """dry_run defaults to True and the manager refuses anything else.

    It is accepted as a field at all only so that an attempt to set it False gets a clear
    explanation of where real flights are started from, rather than being silently ignored.
    """
    mode: Literal["expert", "policy", "hover"] = "expert"
    profile: Literal["ideal", "typical", "harsh"] = "typical"
    seed: int = 0
    attempts: int = 1
    no_forward: bool = False
    no_commit: bool = False
    max_travel: Optional[float] = None
    max_airborne: float = 40.0
    record: bool = False
    model: Optional[str] = None
    dry_run: bool = True


class DroneCameraToggleRequest(BaseModel):
    enabled: bool
