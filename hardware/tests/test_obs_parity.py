import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("gym_pybullet_drones")

from common.constants import ACT_BUFFER, OBS_DIM
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
