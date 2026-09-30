"""
Grab the drone's own onboard camera frame. Shared by the live camera callback
(telemetry_callback.py) and replay POV video generation (replay_hoop.py), so
there is exactly one place that knows how to do this.

Gotcha, confirmed by reading gym_pybullet_drones/envs/BaseAviary.py: env.IMG_RES
is only ever set when the env was built with a vision observation type (RGB,
DEP, ALL). HoopFindAviary uses the plain kinematic (KIN) observation, so
IMG_RES is never set on it, and BaseAviary._getDroneImages() checks
`if self.IMG_RES is None` and calls exit() in its own error path if it isn't.
For a KIN env the attribute does not exist at all, so that check itself would
raise AttributeError first. Either way, calling _getDroneImages() unmodified
on this env would kill the calling process. get_onboard_frame() below sets
IMG_RES on the instance before the first call, defensively, every time.
"""
import numpy as np


def get_onboard_frame(env, nth_drone=0, res=(160, 120)):
    """RGB uint8 (h, w, 3) frame from the drone's own forward-facing camera."""
    if getattr(env, "IMG_RES", None) is None:
        env.IMG_RES = np.array(res)
    rgb, _dep, _seg = env._getDroneImages(nth_drone, segmentation=False)
    # p.getCameraImage's reshape (in BaseAviary) does not fix a dtype, so this
    # can come back as int64; PIL's Image.fromarray needs uint8 for an RGB
    # array, cast explicitly rather than relying on the source dtype.
    return rgb[:, :, :3].astype(np.uint8)
