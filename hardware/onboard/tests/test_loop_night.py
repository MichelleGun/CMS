"""loop_night.py: the stop rule for laps curves, and the commands it builds (nothing is started)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loop_night as ln  # noqa: E402


def test_nothing_is_judged_before_800k_steps():
    ref = [0.4] * 39
    run = [0.0] * 39                     # far behind, but too early to tell
    assert ln.judge_laps(run, ref)[0] is None


def test_a_clear_loser_is_stopped_at_the_first_judged_checkpoint():
    ref, run = [0.4] * 60, [0.2] * 60
    at, why, _ = ln.judge_laps(run, ref)
    assert at == 40 and "below T" in why


def test_a_run_level_with_the_control_is_never_stopped():
    ref = list(0.3 + 0.03 * ((-1) ** i) for i in range(150))
    assert ln.judge_laps(list(ref), ref)[0] is None


def test_commands_are_one_process_with_evaluations_every_20k_steps():
    demos = ["hardware/flights/tello_x/demo.npz"]
    x = ln.run_cmd("X", ln.PARENT, 2_000_000, 3, ln.out_dir("X"), True, demos)
    assert x[x.index("--vec") + 1] == "dummy" and x[x.index("--eval-freq") + 1] == "6666"
    assert "--ring-tracker" in x and x[x.index("--bc-coef") + 1] == "0.1"
    # the variant's flags come after the defaults, so they win (argparse keeps the last)
    assert len(x) - 1 - x[::-1].index("--bc-coef") > x.index("--bc-coef")
    assert x[x.index("--run-id") + 1] == "w10_laps3_X_trk_combo"
    y = ln.run_cmd("Y", ln.PARENT, 2_000_000, 3, ln.out_dir("Y"), True, demos)
    assert y[-2:] == ["--demos", demos[0]]                   # the looping flight replaces the demo set
    assert "--demos" not in x[x.index("--vec"):]
