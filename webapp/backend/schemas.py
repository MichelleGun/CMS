"""Pydantic request/response models for the dashboard API."""
from typing import Dict, List, Literal, Optional

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
    n_eval_episodes: int = 18       # every start zone twice (zone starts)
    # net = 2x2x2 net + ring (hoop_net_env.py); onboard = the Tello camera task
    # (hardware/onboard/train_onboard.py: PPO, optionally from a clone and guided by demos)
    env: Literal["net", "legacy", "onboard"] = "net"
    vec: Literal["auto", "subproc", "dummy"] = "auto"   # auto: env copies in their own processes
    init_run: Optional[str] = None      # continue from this run's latest -> best -> final model
    init_clone: Optional[str] = None    # onboard: start from this behaviour clone (hardware/flights/bc)
    demo_ids: List[str] = []            # onboard: demos for the behaviour-cloning term
    human_weight: float = Field(3.0, ge=0.0, le=20.0)  # onboard: weight of human vs simulated demos
    bc_coef: float = Field(1.0, ge=0.0, le=10.0)       # onboard: strength of the demo term at the start
    profile: Literal["ideal", "typical", "harsh", "measured"] = "typical"   # onboard: simulated drone behaviour
    # onboard robustness (hardware/onboard/sim.py): randomisation strength, the ring's swing,
    # the share of hard starts (behind / beside / below / anywhere), the ramp-up, demo augmentation
    onboard_dr: float = Field(1.0, ge=0.0, le=1.0)
    onboard_sway_deg: float = Field(5.0, ge=0.0, le=45.0)    # the real noodle hardly swings
    onboard_wash_deg: float = Field(15.0, ge=0.0, le=45.0)   # ... but prop wash pushes it away
    hard_start_p: float = Field(0.5, ge=0.0, le=1.0)
    dr_steps: int = Field(500_000, ge=0, le=100_000_000)
    bc_augment: bool = True
    onboard_laps: int = Field(0, ge=0, le=20)        # onboard: passes per episode (0 = endless)
    ring_jitter: float = Field(1.0, ge=0.0, le=2.0)
    radius_start: float = Field(1.5, ge=1.0, le=3.0)
    sway_deg: float = Field(5.0, ge=0.0, le=45.0)    # net env: largest ring swing about its rope knot
    wash_deg: float = Field(15.0, ge=0.0, le=45.0)   # net env: prop-wash tilt of the ring (0 = off)
    start_z: float = Field(0.8, ge=0.0, le=1.6)      # net env: hover start height (0 = the floor)
    same_side: bool = True          # laps: every pass the same way, round the outside (else back-and-forth)
    start_zones: bool = True        # starts cycle through 9 zones of the cage (else random / front runway)
    streak: float = Field(0.5, ge=0.0, le=5.0)       # pass k pays x min(1 + streak x (k-1), 3)
    carry_start: float = Field(0.7, ge=0.0, le=1.0)  # net env: start where the last episode ended
    reverse_prob: float = Field(0.5, ge=0.0, le=1.0) # net env: fly through back to front
    lean_flip_prob: float = Field(0.3, ge=0.0, le=1.0)  # net env: ring leans the other way
    wind_steps: int = Field(1_500_000, ge=0, le=100_000_000)  # net env: swing ramps up over this
    reward_mode: Literal["shaped", "budget"] = "shaped"     # net env: MDP v2 / v3
    budget: float = Field(3.0, gt=0.0, le=100.0)            # net env, budget mode
    laps: int = Field(1, ge=0, le=20)                       # net env, shaped mode (0 = endless)
    reward_weights: Optional[Dict[str, float]] = None       # net / onboard env: overrides of the defaults
    # --- onboard: PPO and the rest of the run configuration (run_config.FIELDS describes them for the
    # Config card). None = the trainer's own default, so the old Training panel's payload is unchanged.
    learning_rate: Optional[float] = Field(None, ge=1e-6, le=1e-2)
    gamma: Optional[float] = Field(None, ge=0.8, le=0.9999)
    gae_lambda: Optional[float] = Field(None, ge=0.8, le=1.0)
    ent_coef: Optional[float] = Field(None, ge=0.0, le=0.1)
    target_kl: Optional[float] = Field(None, ge=0.001, le=0.5)
    batch_size: Optional[int] = Field(None, ge=16, le=65536)
    n_epochs: Optional[int] = Field(None, ge=1, le=50)
    critic_warmup_steps: Optional[int] = Field(None, ge=0, le=10_000_000)
    log_std_reset: Optional[float] = Field(None, ge=-5.0, le=1.0)
    sde: bool = False
    ring_tracker: bool = False
    streak_cap: Optional[float] = Field(None, ge=1.0, le=20.0)
    bc_coef_min: Optional[float] = Field(None, ge=0.0, le=10.0)
    bc_decay_steps: Optional[int] = Field(None, ge=1, le=100_000_000)
    # onboard: training speed (TRAINING_SPEEDUPS.md). None = the trainer's default.
    n_eval_envs: Optional[int] = Field(None, ge=1, le=16)
    checkpoint_min: Optional[float] = Field(None, ge=0.0, le=1440.0)
    zone_focus: Optional[float] = Field(None, ge=0.0, le=10.0)
    stall_sec: Optional[float] = Field(None, ge=0.0, le=600.0)
    norm_reward: bool = False


class RunLabelRequest(BaseModel):
    label: str = Field("", max_length=60)


class RunStarRequest(BaseModel):
    starred: bool


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
    max_airborne: float = 150.0
    record: bool = False
    model: Optional[str] = None
    dry_run: bool = True
    confirm: bool = False
    human_gain: float = 0.6
    # 2026-10-14: the height window and start height reach fly_hoop (before, the dashboard
    # always flew its CLI defaults), the human keys come from the popup window by default,
    # and the ClearML upload can be skipped
    max_height: int = 190
    min_height: int = 10
    start_height: int = 80
    keys: Literal["popup", "browser"] = "popup"
    clearml: bool = True


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
    dr: float = Field(0.0, ge=0.0, le=1.0)              # randomised (robust) task
    sway_deg: float = Field(5.0, ge=0.0, le=45.0)
    hard_start_p: float = Field(0.0, ge=0.0, le=1.0)


class TeacherDemosRequest(BaseModel):
    run_id: str                         # a net-env (PyBullet) training run
    name: str
    episodes: int = 150
    profile: Literal["ideal", "typical", "harsh"] = "typical"
    seed: int = 0


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
