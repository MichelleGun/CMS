"""
The learning loop, end to end with no drone: expert -> demonstrations -> behaviour
cloning -> a policy that flies the task.

This is what "trainable by any method" has to mean concretely. It also guards the thing
most likely to break silently when two tasks share one trainer: that the simulator's task
and this one each get the observation shape and the loss that belong to them.
"""
import numpy as np
import pytest

from onboard import spec


@pytest.fixture(scope="module")
def demos(tmp_path_factory):
    """A small demonstration set from the scripted expert."""
    from onboard.make_demos import generate
    rec, tally = generate(episodes=14, seed=0, verbose=False)
    assert tally["passed"] >= 8, "the expert should mostly succeed; see test_expert_sim"
    assert len(rec) > 200
    path = rec.save(tmp_path_factory.mktemp("demos") / "onboard.npz")
    return path, tally


def test_demos_carry_the_task_identity_and_shape(demos):
    from imitation.record_demo import load_demo
    path, _tally = demos
    d = load_demo(path)
    assert d["obs"].shape[1] == spec.OBS_DIM
    assert d["intent"].shape[1] == spec.ACT_DIM
    assert d["meta"]["task"] == spec.TASK_NAME
    assert d["meta"]["obs_dim"] == spec.OBS_DIM
    assert d["meta"]["obs_shape"] == [spec.OBS_DIM]
    assert np.all(np.isfinite(d["obs"]))
    assert np.all(np.abs(d["intent"]) <= 1.0 + 1e-6)


def test_only_successful_episodes_are_recorded_by_default(demos):
    """A clone copies its demonstrator, so failed approaches must not be taught."""
    from onboard.make_demos import generate
    rec, tally = generate(episodes=6, seed=50, verbose=False)
    assert tally["kept_episodes"] == tally["passed"]

    rec_all, tally_all = generate(episodes=6, seed=50, keep_failures=True, verbose=False)
    assert tally_all["kept_episodes"] == 6
    assert len(rec_all) >= len(rec)


def test_the_trainer_picks_the_onboard_loss_and_shape_from_the_demo_meta(demos, tmp_path):
    """The simulator's cosine-direction loss is wrong for four independent rc channels,
    and its (1, 75) observation shape is wrong for this task. Both must follow the data,
    not a flag the caller has to remember."""
    from imitation.bc_train import main as bc_main
    path, _ = demos
    meta = bc_main(["--demos", str(path), "--out", str(tmp_path / "clone"),
                    "--epochs", "30", "--seed", "0"])
    assert meta["task"] == spec.TASK_NAME
    assert meta["loss"] == "mse"
    assert meta["obs_shape"] == [spec.OBS_DIM]


@pytest.mark.slow
def test_a_clone_of_the_expert_flies_the_task(demos, tmp_path):
    """The headline: a policy learned purely from demonstrations gets through the ring."""
    from imitation.bc_train import main as bc_main
    from onboard.gym_env import rollout
    from onboard.policy import load_policy

    path, _ = demos
    out = tmp_path / "clone"
    meta = bc_main(["--demos", str(path), "--out", str(out),
                    "--epochs", "250", "--lr", "3e-3", "--seed", "0"])
    assert meta["val_dir_loss"] < 0.05, "cloning should fit a scripted demonstrator well"

    fn = load_policy(out / "bc_model.zip")
    res = rollout(fn, episodes=12, seed=300)
    assert res["pass_rate"] >= 0.4, (
        f"the clone did not learn to traverse: {res}. Compare against the expert's own "
        "rate in test_expert_sim.py; a large gap points at the loss or the observation, "
        "a small one at needing more demonstrations.")


def test_a_policy_for_the_wrong_task_is_refused(tmp_path):
    """A simulator-task model would fly on four numbers that mean something else
    entirely -- a direction and a speed fraction, not rc channels. Nothing downstream
    would raise, so it has to be caught at load time."""
    from imitation.bc_train import build_model
    from onboard.policy import load_policy

    sim_model = build_model("ppo", 0, obs_shape=(1, 75), act_shape=(1, 4))
    p = tmp_path / "sim_model.zip"
    sim_model.save(str(p))
    with pytest.raises(ValueError, match="HoopFindAviary|shape"):
        load_policy(p)


def test_a_model_whose_meta_names_another_task_is_refused(tmp_path):
    import json
    from imitation.bc_train import build_model
    from onboard.policy import load_policy

    m = build_model("ppo", 0, obs_shape=(spec.OBS_DIM,), act_shape=(spec.ACT_DIM,))
    p = tmp_path / "bc_model.zip"
    m.save(str(p))
    (tmp_path / "meta.json").write_text(json.dumps({"algo": "ppo", "task": "sim_hoop"}),
                                        encoding="utf-8")
    with pytest.raises(ValueError, match="trained for task"):
        load_policy(p)


def test_a_correctly_shaped_policy_loads_and_returns_bounded_actions(tmp_path):
    from imitation.bc_train import build_model
    from onboard.policy import load_policy

    m = build_model("ppo", 0, obs_shape=(spec.OBS_DIM,), act_shape=(spec.ACT_DIM,))
    p = tmp_path / "bc_model.zip"
    m.save(str(p))
    fn = load_policy(p, algo="ppo")
    a = fn(np.zeros(spec.OBS_DIM, dtype=np.float32))
    assert a.shape == (spec.ACT_DIM,)
    assert np.all(np.abs(a) <= 1.0)


def test_warm_critic_is_refused_for_this_task(demos, tmp_path):
    """mc_returns reads obs[:, 72:75] as hoop-minus-position, which does not exist here.
    Silently computing returns from unrelated columns would be worse than refusing."""
    from imitation.bc_train import main as bc_main
    path, _ = demos
    with pytest.raises(SystemExit, match="warm-critic"):
        bc_main(["--demos", str(path), "--out", str(tmp_path / "c"),
                 "--epochs", "2", "--warm-critic"])


def test_mc_returns_refuses_a_non_simulator_observation():
    from imitation.record_demo import mc_returns
    obs = np.zeros((10, spec.OBS_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="specific to the simulator"):
        mc_returns(obs, np.zeros(10, dtype=np.int64))
