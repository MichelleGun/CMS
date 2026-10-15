"""rescore_checkpoints.py and sweep_onboard.py: the parts that need no training."""
import json
import sys
from pathlib import Path

import pytest

CMS = Path(__file__).resolve().parents[3]
for p in (CMS / "hardware" / "onboard", CMS / "hardware", CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rescore_checkpoints as R  # noqa: E402
import sweep_onboard as SW  # noqa: E402


def test_candidates_drop_duplicates_and_include_checkpoints(tmp_path):
    (tmp_path / "best_model.zip").write_bytes(b"a")
    (tmp_path / "latest_model.zip").write_bytes(b"b")
    (tmp_path / "final_model.zip").write_bytes(b"b")              # same as latest
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "step_000000001000.zip").write_bytes(b"c")
    labels = [lab for lab, _ in R.candidates(tmp_path)]
    assert labels == ["best", "latest", "ckpt_000000001000"]


def test_a_checkpoint_gets_its_runs_meta_for_the_flight_loader(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"obs_history": [0, 1]}))
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    m = ck / "step_000000001000.zip"
    m.write_bytes(b"c")
    side = R._meta_for(m, tmp_path)
    assert json.loads(side.read_text()) == {"obs_history": [0, 1]}


def test_interval_narrows_with_more_episodes():
    lo1, hi1 = R.interval(1.0, 1.0, 36)
    lo2, hi2 = R.interval(1.0, 1.0, 144)
    assert hi2 - lo2 == pytest.approx((hi1 - lo1) / 2, abs=0.01)


def test_sweep_score_is_the_mean_of_the_last_three_evaluations(tmp_path):
    assert SW.score_of(tmp_path) == (None, 0)
    (tmp_path / "passes.json").write_text(json.dumps([{"mean": m} for m in (0.1, 0.5, 1.0, 1.5)]))
    s, n = SW.score_of(tmp_path)
    assert n == 4 and s == pytest.approx(1.0)


def test_sweep_flags_come_last_so_they_override_the_experiment():
    optuna = pytest.importorskip("optuna")
    flags = SW.sample_flags(optuna.trial.FixedTrial({
        "lr": 5e-5, "ent_coef": 1e-3, "target_kl": 0.02, "gae_lambda": 0.95, "gamma": 0.99,
        "n_epochs": 5, "batch_size": 512}))
    cmd = SW.trial_command("O", "results/hoop/runs/w10_n9_O_oblique", 1000, 4, "subproc",
                           SW.trial_dir("x", 3), flags)
    assert cmd[-len(flags):] == flags
    assert cmd[cmd.index("--run-id") + 1] == "sweep_x_t003"
    assert cmd.count("--lr") == 2 and cmd[len(cmd) - cmd[::-1].index("--lr")] == "5e-05"
