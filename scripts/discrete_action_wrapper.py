"""
A discrete-action wrapper around HoopFindAviary's velocity control, so that
Stable-Baselines3's DQN (which only trains on a Discrete action space) can be
used on this task at all.

Why a wrapper around VEL rather than a separate PID-based discrete env: the
project already tried PID here and found it destabilised training in this
small room (see PROGRESS.md, "Found and fixed why the first HoopFindAviary
training run wouldn't converge"). Reusing the already-stable VEL controller
and only discretising its direction/speed input avoids reintroducing that
problem, at the cost of the drone only ever being able to move along 6 fixed
axis directions plus hover, one per action, instead of any continuous
direction. That is worth knowing when comparing DQN's reward curve to the
continuous algorithms (PPO, A2C, SAC, TD3): DQN starts from a strictly
smaller set of moves and will likely need more timesteps, and may cap out
lower, since it can never move diagonally in a single action.
"""
import numpy as np
from gymnasium import ActionWrapper, spaces

# index -> (dx, dy, dz, speed_fraction), fed straight into HoopFindAviary's
# VEL action (BaseRLAviary.ActionType.VEL: direction vector + speed fraction
# of SPEED_LIMIT).
DISCRETE_MOVES = np.array([
    [1, 0, 0, 1.0],   # +x
    [-1, 0, 0, 1.0],  # -x
    [0, 1, 0, 1.0],   # +y
    [0, -1, 0, 1.0],  # -y
    [0, 0, 1, 1.0],   # +z (up)
    [0, 0, -1, 1.0],  # -z (down)
    [0, 0, 0, 0.0],   # hover
], dtype=np.float32)

MOVE_LABELS = ["+x", "-x", "+y", "-y", "+z", "-z", "hover"]


class HoopFindDiscreteVel(ActionWrapper):
    """Wraps a HoopFindAviary(act=ActionType.VEL) env with 7 fixed moves."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_MOVES))

    def action(self, action):
        return DISCRETE_MOVES[int(action)].reshape(1, 4)
