"""Shared data types for state estimation."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

OK, DEGRADED, LOST = "OK", "DEGRADED", "LOST"


@dataclass
class CameraDetection:
    """Drone position seen by the overhead camera, in the cage frame."""
    t: float               # capture timestamp (monotonic clock, stamped at grab(), before latency correction)
    x: float
    y: float
    yaw: Optional[float]   # rad, cage frame, CCW positive (None if unavailable)
    quality: float = 1.0   # (0, 1]: marker size / reprojection quality; scales the noise


@dataclass
class StateEstimate:
    t: float
    pos: np.ndarray            # (3,) cage frame, metres
    vel: np.ndarray            # (3,) cage frame, m/s
    roll: float
    pitch: float
    yaw: float                 # cage-frame yaw, rad, CCW positive
    yaw_rate: float            # rad/s
    pos_std: float             # metres (largest horizontal std)
    pos_health: str
    yaw_health: str
    tele_age: float            # s since last accepted telemetry packet
    cam_age: float             # s since last camera detection
    ready: bool                # filter initialised and telemetry fresh
    z_smooth: Optional[float] = None
    fields: dict = field(default_factory=dict)   # latest raw Tello fields (battery, temp, ...)

    def vector(self) -> np.ndarray:
        """Unified state vector [x, y, z, vx, vy, vz, yaw, pitch, roll]."""
        return np.array([*self.pos, *self.vel, self.yaw, self.pitch, self.roll])

    @property
    def rpy(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])
