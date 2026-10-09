import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("gym_pybullet_drones")

from common.constants import ACT_BUFFER, NET_OBS_DIM, OBS_DIM
from common.obs_builder import ObsBuilder
from sim_check.obs_parity import check_parity


def test_obs_layout_matches_simulator():
    assert check_parity(steps=40, verbose=False) < 1e-5


def test_builder_shape_buffer_order_and_zero_init():
    ob = ObsBuilder([0.5, 0.5, 1.0])
    o = ob.build([0, 0, 1], [0, 0, 0], [0, 0, 0], 0.0)
    assert o.shape == (1, OBS_DIM) and o.dtype == np.float32
    assert np.all(o[0, 12:12 + ACT_BUFFER * 4] == 0)             # hardware starts with a zero buffer
    ob.push_action([1, 0, 0, 1.0])
    ob.push_action([0, 1, 0, 0.5])
    o = ob.build([0, 0, 1], [0, 0, 0], [0, 0, 0], 0.03)
    buf = o[0, 12:72].reshape(ACT_BUFFER, 4)
    assert np.allclose(buf[-1], [0, 1, 0, 0.5]) and np.allclose(buf[-2], [1, 0, 0, 1.0])   # newest last
    assert np.allclose(o[0, 72:75], [0.5, 0.5, 0.0])                                        # hoop - pos
    ob.reset()
    assert np.all(ob.build([0, 0, 1], [0, 0, 0], [0, 0, 0], 0.0)[0, 12:72] == 0)


def test_net_env_layout_matches_builder_with_normal():
    """HoopNetAviary's 78-D observation == ObsBuilder(hoop_normal=...) on the same state/actions."""
    from hoop_net_env import HoopNetAviary
    env = HoopNetAviary(seed=0, wind=0.0)     # ObsBuilder holds one fixed hoop pose: no live sway here
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1, NET_OBS_DIM)
        # the observed normal points the way this episode must be flown (flipped when reversed);
        # on hardware that is the direction the operator gives ObsBuilder
        ob = ObsBuilder(env.ring_c, hoop_normal=env.ring_n * env._dir)
        rng = np.random.default_rng(0)
        for k in range(30):
            a = rng.uniform(-1, 1, size=(1, 4)).astype(np.float32)
            obs, _r, te, tr, _ = env.step(a)
            ob.push_action(a[0])
            s = env._getDroneStateVector(0)
            mine = ob.build(s[0:3], s[7:10], s[10:13], (k + 1) / 30, angvel=s[13:16])
            assert np.max(np.abs(mine - obs)) < 1e-5
            if te or tr:
                break
    finally:
        env.close()
