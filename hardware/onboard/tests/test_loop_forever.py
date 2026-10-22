"""loop_forever.py: a seed per round, only clear wins replace the parent, one copy at a time."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import loop_forever as lf  # noqa: E402


def test_only_a_clear_win_replaces_the_parent():
    assert lf.choose(2.60, [dict(score=2.65), dict(score=2.55)], margin=0.10) is None
    win = lf.choose(2.60, [dict(score=2.75, run="a"), dict(score=2.71, run="b")], margin=0.10)
    assert win["run"] == "a"
    assert lf.choose(None, [dict(score=1.0, run="c")])["run"] == "c"
    assert lf.choose(2.0, [dict(score=None)]) is None


def test_corners_count_fully_in_the_score():
    s = dict(lpe=2.33, oblique_entry=0.57)
    assert abs(lf.total(s, 1.0) - 2.90) < 1e-9 and abs(lf.total(s, 0.5) - 2.615) < 1e-9


def test_each_round_trains_with_its_own_seed(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(lf, "LOG", tmp_path / "log.txt")
    monkeypatch.setattr(lf, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(lf.subprocess, "run", lambda cmd, cwd=None: seen.append(cmd) or type("R", (), {"returncode": 0})())
    monkeypatch.setattr(lf, "best_of_run", lambda run, a: (None, None, None))
    a = type("A", (), dict(pool=["CZ", "CE"], timesteps=1000, n_envs=1, parallel=1, n_eval_envs=1,
                           relabel_episodes=1, rescore_episodes=1, oblique_episodes=1, oblique_weight=1.0,
                           margin=0.1))()
    st = dict(round=8, parent="results/hoop/runs/w10_mg6_C9_loop_practice", parent_key="C9",
              parent_score=2.6, tried=[], history=[])
    lf.run_round(st, a)
    lf.run_round(st, a)
    seeds = [c[c.index("--extra-args") + 1] for c in seen]
    assert seeds == ["--seed 9", "--seed 10"]
    assert seen[0][seen[0].index("--keys") + 1:seen[0].index("--parent")] == ["C9", "CZ"]
    assert seen[1][seen[1].index("--keys") + 1:seen[1].index("--parent")] == ["C9", "CE"]
    assert st["parent_key"] == "C9" and st["round"] == 10


def test_a_second_copy_does_not_start(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(lf, "LOCK", tmp_path / "lock")
    assert lf.take_lock()                                   # this process takes it
    (tmp_path / "lock").write_text(str(os.getppid()))       # another live process holds it
    assert not lf.take_lock()
    (tmp_path / "lock").write_text("999999999")             # a stale lock from a dead process
    assert lf.take_lock()
