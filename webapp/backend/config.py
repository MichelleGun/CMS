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
ASSETS_DIR = CMS_DIR / "assets"
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
import hoop_net_env as hoop_env  # noqa: E402  (local module, path set above)

ROOM = hoop_env.ROOM.tolist()
XY_MARGIN = hoop_env.XY_MARGIN
START_MARGIN = getattr(hoop_env, "START_MARGIN", XY_MARGIN)   # drone starts: almost the whole floor
Z_RANGE = list(hoop_env.Z_RANGE)
START_Z = hoop_env.START_Z
REACH_RADIUS = hoop_env.REACH_RADIUS

# Net + ring geometry for the dashboard's 3D views (the same HoopSpec the training env uses).
_SPEC = hoop_env.DEFAULT_SPEC
HOOP_SPEC = {
    "centre": list(_SPEC.centre),
    "inner_radius": _SPEC.inner_radius,
    "tube_radius": _SPEC.tube_radius,
    "normal_yaw_deg": _SPEC.normal_yaw_deg,
    "normal_tilt_deg": _SPEC.normal_tilt_deg,
    "centre_jitter": list(_SPEC.centre_jitter),
    "yaw_jitter_deg": _SPEC.yaw_jitter_deg,
    "tilt_jitter_deg": _SPEC.tilt_jitter_deg,
    "radius_jitter_frac": _SPEC.radius_jitter_frac,
    "drone_radius": _SPEC.drone_radius,
    "approach_dist": _SPEC.approach_dist,
    "sway_tilt_deg": _SPEC.sway_tilt_deg,
    "sway_top_amp": _SPEC.sway_top_amp,
    "sway_period_s": list(_SPEC.sway_period_s),
}

# Observation size of today's net env: KIN state (12) + a 15-step action buffer (4 x ctrl_freq/2)
# + the ring terms. Runs saved with another size come from an older env version: their scores
# are not comparable, so they are never picked as the best run (they can still be starred).
NET_OBS_DIM = 72 + hoop_env.OBS_EXTRA
# + the lap "armed" flag (armed_obs, runs since 2026-10-11 night): the same task, so comparable
NET_OBS_DIMS = (NET_OBS_DIM, NET_OBS_DIM + 1, NET_OBS_DIM + 2)   # + the pass count (v5)

# The net env's reward (MDP), for the Training panel's editable "Reward" block and the run
# overlay's read-only card: the default weights, and which mode uses which with a one-liner each.
REWARD_INFO = {
    "modes": list(hoop_env.REWARD_MODES),
    # budget_shape: the env's own default stays 0 (old runs replay unchanged), but new runs from
    # the dashboard start from MDP v4 (train_hoop_multi.py's --budget-shape default)
    "defaults": dict(hoop_env.DEFAULT_REWARD_WEIGHTS, budget_shape=2.0, budget_streak=0.5,
                     budget_wrong=0.5),
    "terms": {
        "shaped": [
            ["pass_bonus", "+ per pass through the ring"],
            ["centre_bonus", "+ up to this extra for a dead-centre pass"],
            ["shaping_gain", "x metres of progress towards / through the ring (potential-based)"],
            ["axis_pull", "potential weight of being off the ring's axis just in front of it"],
            ["time_penalty", "- per control step (30 per second)"],
            ["rim_penalty", "- per rim clip"],
            ["net_penalty", "- touching the net (ends the episode)"],
            ["net_bounce_penalty", "- touching the net when it bounces instead"],
            ["wrong_way_penalty", "- passing through the wrong way"],
        ],
        "budget": [
            ["budget_pass", "+ per pass (endless laps)"],
            ["budget_time", "- per control step"],
            ["budget_rim", "- per rim clip"],
            ["budget_net", "- at least this for the net, which also ends the episode at -budget"],
            ["budget_shape", "x metres of progress towards / through the ring (MDP v4, potential-based; 0 = v3)"],
            ["budget_streak", "streak: pass k pays budget_pass x (1 + this x (k-1)), capped (MDP v5; 0 = flat)"],
            ["budget_streak_cap", "largest streak multiplier"],
            ["budget_wrong", "- going back through the opening the wrong way"],
        ],
    },
}

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


def bounds_ok(pos, margin=None):
    """True if a (x, y, z) position is inside the room, respecting the same
    margins hoop_env.py itself samples within."""
    if pos is None:
        return True
    x, y, z = pos
    m = XY_MARGIN if margin is None else margin
    half_xy = (ROOM[0] / 2 - m, ROOM[1] / 2 - m)
    return (-half_xy[0] <= x <= half_xy[0]
            and -half_xy[1] <= y <= half_xy[1]
            and 0.0 <= z <= ROOM[2])
