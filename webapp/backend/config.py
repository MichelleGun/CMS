"""
Paths and constants shared across the backend. Room/margin/height numbers are
imported directly from scripts/hoop_env.py, not duplicated, so a change there
propagates here automatically.
"""
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
CMS_DIR = WEBAPP_DIR.parent
FRONTEND_DIR = WEBAPP_DIR / "frontend"
SCRIPTS_DIR = CMS_DIR / "scripts"
RESULTS_HOOP_DIR = CMS_DIR / "results" / "hoop"
RUNS_DIR = RESULTS_HOOP_DIR / "runs"

# The real-drone side. hardware/onboard/fly_hoop.py writes one directory per flight
# under FLIGHTS_DIR, in the same spirit as RUNS_DIR: the folder IS the database, so
# restarting the backend just re-reads it.
HARDWARE_DIR = CMS_DIR / "hardware"
FLIGHTS_DIR = HARDWARE_DIR / "flights"
FLY_HOOP_SCRIPT = HARDWARE_DIR / "onboard" / "fly_hoop.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import hoop_env  # noqa: E402  (local module, path set above)

ROOM = hoop_env.ROOM.tolist()
XY_MARGIN = hoop_env.XY_MARGIN
Z_RANGE = list(hoop_env.Z_RANGE)
START_Z = hoop_env.START_Z
REACH_RADIUS = hoop_env.REACH_RADIUS

# (id, label, env action variant). "continuous_vel" feeds HoopFindAviary's own
# VEL action directly; "discrete_vel" wraps it with HoopFindDiscreteVel (see
# scripts/discrete_action_wrapper.py) so SB3's DQN (Discrete-only) can use it.
ALGORITHMS = [
    {"id": "ppo", "label": "PPO", "env_variant": "continuous_vel"},
    {"id": "a2c", "label": "A2C", "env_variant": "continuous_vel"},
    {"id": "sac", "label": "SAC", "env_variant": "continuous_vel"},
    {"id": "td3", "label": "TD3", "env_variant": "continuous_vel"},
    {"id": "dqn", "label": "DQN (7 discrete moves)", "env_variant": "discrete_vel"},
]
ALGORITHM_IDS = {a["id"] for a in ALGORITHMS}

RUNS_DIR.mkdir(parents=True, exist_ok=True)
FLIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def bounds_ok(pos):
    """True if a (x, y, z) position is inside the room, respecting the same
    margins hoop_env.py itself samples within."""
    if pos is None:
        return True
    x, y, z = pos
    half_xy = (ROOM[0] / 2 - XY_MARGIN, ROOM[1] / 2 - XY_MARGIN)
    return (-half_xy[0] <= x <= half_xy[0]
            and -half_xy[1] <= y <= half_xy[1]
            and 0.0 <= z <= ROOM[2])
