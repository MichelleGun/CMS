"""The race's stop rule (w10_race.judge) and the stop.flag hook in train_onboard."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import w10_race  # noqa: E402
from w10_race import CHECK_EVERY, judge  # noqa: E402


def test_never_judged_before_the_first_checkpoint():
    ref = [0.9] * 100
    at, _, checks = judge([0.0] * (CHECK_EVERY - 1), ref)
    assert at is None and checks == []


def test_a_clear_loser_stops_at_the_first_checkpoint():
    at, why, _ = judge([0.4] * 60, [0.6] * 60)
    assert at == CHECK_EVERY and "below C" in why


def test_noise_sized_gaps_do_not_stop_and_two_warnings_in_a_row_do():
    ref = [0.70] * 100
    assert judge([0.66] * 100, ref)[0] is None                    # 0.04 below: noise
    at, why, _ = judge([0.63] * 100, ref)                         # 0.07 below, twice
    assert at == 2 * CHECK_EVERY and "two checkpoints" in why
    wobble = [0.63] * CHECK_EVERY + [0.70] * CHECK_EVERY + [0.63] * CHECK_EVERY
    assert judge(wobble, ref)[0] is None                          # not in a row


def test_a_run_never_stops_against_itself():
    c = w10_race.curve(w10_race.REF)
    if not c:
        pytest.skip("no w10_front_C_kl curve on this machine")
    assert judge(c, c)[0] is None


@pytest.mark.parametrize("key", ["A", "B", "D", "F"])
def test_the_rule_replayed_on_round_one_stops_every_run_c_beat(key):
    ref = w10_race.curve(w10_race.REF)
    run = w10_race.curve(w10_race.run_dir("front", key))
    if not ref or not run:
        pytest.skip("no A-F curves on this machine")
    assert judge(run, ref)[0] is not None


def test_stop_flag_ends_training(tmp_path):
    import train_onboard as t
    cb = t.StopRequestCallback(tmp_path, every=1)
    try:
        cb.n_calls = 1
        assert cb._on_step() is True
        (tmp_path / "stop.flag").write_text("behind C")
        assert cb._on_step() is False
    finally:
        t._stop_requested["flag"] = False
