"""Pydantic request/response models for the dashboard API."""
from typing import List, Literal, Optional

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
    until_stopped: bool = False     # ignore timesteps: train until Stop (overnight)
    n_envs: int = 4
    eval_freq: int = 5000           # per env: with 4 envs, one evaluation every 20k timesteps
    n_eval_episodes: int = 10
    env: Literal["net", "legacy"] = "net"     # net = 2x2x2 net + ring (hoop_net_env.py)
    ring_jitter: float = Field(1.0, ge=0.0, le=2.0)
    radius_start: float = Field(1.5, ge=1.0, le=3.0)
    sway_deg: float = Field(20.0, ge=0.0, le=45.0)   # net env: largest ring swing about its rope knot


class StartRunResponse(BaseModel):
    run_id: str
    status: str


class ReplayRequest(BaseModel):
    variant: Literal["current", "best"] = "current"
    episodes: int = 3
    pov: bool = False


class ShowcaseRequest(BaseModel):
    variant: Literal["latest", "best"] = "latest"
    seed: Optional[int] = None
    realtime: bool = False          # False = play as fast as possible


class RealtimeToggleRequest(BaseModel):
    enabled: bool


class ReplayResponse(BaseModel):
    replay_id: str
    status: str


class CameraToggleRequest(BaseModel):
    enabled: bool


# --- real drone (hardware/onboard/fly_hoop.py) ------------------------------

class StartFlightRequest(BaseModel):
    """dry_run defaults to True. A real flight (dry_run=False) also needs confirm=True,
    and is accepted only from this computer (see main.py)."""
    mode: Literal["expert", "policy", "hover", "human"] = "expert"
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
    confirm: bool = False
    human_gain: float = 0.6


class SaveAnswerRequest(BaseModel):
    save: bool
    passed: bool = False


# --- imitation learning -------------------------------------------------------

class GenerateDemosRequest(BaseModel):
    name: str
    episodes: int = 60
    profile: Literal["ideal", "typical", "harsh"] = "typical"
    seed: int = 0
    render: bool = False


class LabelDemoRequest(BaseModel):
    demo_id: str
    passed: bool


class TrainCloneRequest(BaseModel):
    name: str
    demo_ids: List[str]
    epochs: int = 200
    lr: float = 3e-3
    use: Literal["intent", "executed"] = "intent"
    eval_episodes: int = Field(20, ge=0, le=200)
    seed: int = 0


class EvaluateRequest(BaseModel):
    episodes: int = 12
    render: bool = False


class DroneCameraToggleRequest(BaseModel):
    enabled: bool
