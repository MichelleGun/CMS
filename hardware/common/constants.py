"""
Constants mirrored from the simulator (scripts/hoop_env.py, BaseRLAviary.py).

Kept as plain numbers on purpose: importing hoop_env pulls in pybullet, which
the hardware runtime must not need. sim_check/obs_parity.py asserts these
still match the simulator, so drift is caught by a test rather than by a
crash in the air.
"""
import numpy as np

# --- cage -------------------------------------------------------------------
ROOM = np.array([2.0, 2.0, 2.0])      # x, y, z (m); x/y centred on 0, floor at z = 0
XY_MARGIN = 0.25
HOOP_XY_MAX = ROOM[:2] / 2 - XY_MARGIN   # hoop sampled uniformly in +-0.75 m
HOOP_Z_RANGE = (0.3, 1.6)
START_Z = 0.1
REACH_RADIUS = 0.15

# --- control / observation --------------------------------------------------
CTRL_HZ = 30
CTRL_DT = 1.0 / CTRL_HZ
ACT_BUFFER = CTRL_HZ // 2            # 15 past actions in the observation
ACT_DIM = 4                          # [dx, dy, dz, speed_fraction], VEL action type
OBS_DIM = 12 + ACT_BUFFER * ACT_DIM + 3   # 75
NET_OBS_DIM = OBS_DIM + 3                 # 78: HoopNetAviary also carries the ring normal
EPISODE_SEC = 15.0
TILT_LIMIT = 0.5                     # rad, sim truncation threshold on roll/pitch

# BaseRLAviary: SPEED_LIMIT = 0.03 * MAX_SPEED_KMH * (1000 / 3600), CF2X max 30 km/h
SPEED_LIMIT = 0.03 * 30.0 * (1000.0 / 3600.0)   # 0.25 m/s

GRAVITY = 9.81
