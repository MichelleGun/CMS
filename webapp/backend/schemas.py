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
