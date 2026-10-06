"""
Guard for sim-to-real transferability: drive the real HoopFindAviary and our ObsBuilder with the
same ground-truth state and the same actions, and assert the 75-dim observations are identical.
Also asserts hardware/common/constants.py still matches the simulator.

    C:/Users/miche/miniconda3/envs/drones/python.exe hardware/sim_check/obs_parity.py

Needs pybullet + gym_pybullet_drones (the `drones` env). The hardware runtime itself does not.
"""
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
for p in (HW, HW.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import constants as C                    # noqa: E402
from common.obs_builder import ObsBuilder            # noqa: E402


def check_constants(env):
    raw = env.unwrapped
    assert np.allclose(raw.ROOM, C.ROOM), "ROOM differs from hoop_env"
    assert abs(raw.SPEED_LIMIT - C.SPEED_LIMIT) < 1e-9, f"SPEED_LIMIT {raw.SPEED_LIMIT} vs {C.SPEED_LIMIT}"
    assert raw.CTRL_FREQ == C.CTRL_HZ
    assert raw.ACTION_BUFFER_SIZE == C.ACT_BUFFER
    assert env.observation_space.shape == (1, C.OBS_DIM), env.observation_space.shape
    assert env.action_space.shape == (1, C.ACT_DIM), env.action_space.shape
    import hoop_env
    assert abs(hoop_env.REACH_RADIUS - C.REACH_RADIUS) < 1e-12
    assert tuple(hoop_env.Z_RANGE) == tuple(C.HOOP_Z_RANGE)
    assert abs(hoop_env.START_Z - C.START_Z) < 1e-12


def check_parity(steps=60, seed=0, verbose=True):
    from hoop_env import HoopFindAviary
    env = HoopFindAviary(gui=False, start_pos=np.array([0.2, -0.3, 0.1]),
                         target_pos=np.array([-0.4, 0.5, 1.1]))
    try:
        check_constants(env)
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        raw = env.unwrapped
        ob = ObsBuilder(raw.TARGET_POS)

        def truth():
            s = raw._getDroneStateVector(0)
            return s[0:3], s[7:10], s[10:13], s[13:16]

        pos, rpy, vel, ang = truth()
        mine = ob.build(pos, rpy, vel, 0.0, angvel=ang)
        worst = float(np.max(np.abs(mine - obs)))
        assert mine.shape == obs.shape == (1, C.OBS_DIM)
        for k in range(steps):
            a = rng.uniform(-1, 1, size=(1, 4)).astype(np.float32)
            if k % 7 == 0:
                a[:] = 0
            obs, _r, _te, _tr, _ = env.step(a)
            ob.push_action(a[0])
            pos, rpy, vel, ang = truth()
            mine = ob.build(pos, rpy, vel, (k + 1) * C.CTRL_DT, angvel=ang)
            worst = max(worst, float(np.max(np.abs(mine - obs))))
            if _te or _tr:
                break
        if verbose:
            print(f"[obs_parity] {k + 1} steps, obs shape {obs.shape}, max |sim - ours| = {worst:.3e}")
        assert worst < 1e-5, f"observation layouts differ (max abs diff {worst})"

        # informational: how well does d(rpy)/dt approximate the simulator's angular velocity?
        env.reset(seed=seed + 1)
        ob2 = ObsBuilder(raw.TARGET_POS)
        errs = []
        for k in range(60):
            a = rng.uniform(-1, 1, size=(1, 4)).astype(np.float32)
            env.step(a)
            pos, rpy, vel, ang = truth()
            est = ob2.angular_velocity(rpy, (k + 1) * C.CTRL_DT)
            errs.append(np.linalg.norm(est - ang))
        if verbose:
            print(f"[obs_parity] angular-velocity proxy (low-passed d rpy/dt): median err {np.median(errs):.3f} rad/s "
                  f"(sim values are ~{np.median([np.linalg.norm(ang)]):.3f} rad/s; informational)")
        return worst
    finally:
        env.close()


if __name__ == "__main__":
    check_parity()
    print("[obs_parity] OK")
