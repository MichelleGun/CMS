"""
Vectorised-env construction and policy loading shared by the trainers
(train_hoop_multi.py and hardware/onboard/train_onboard.py).

Parallel envs: stable_baselines3's make_vec_env defaults to DummyVecEnv, which steps every
env copy one after another in the training process, so a 4-env PyBullet run uses about one
core. SubprocVecEnv puts each copy in its own process. It changes only wall-clock time: PPO
still sees exactly the same amount of data per update.

Stopping: the dashboard stops a run with CTRL_BREAK_EVENT sent to the run's process group
(webapp/backend/jobs.py), and env worker processes are in that group too. Left alone, the
break would kill each worker mid-step and the trainer would crash with a broken pipe instead
of saving. So each worker ignores SIGINT/SIGBREAK, and the trainer, which handles the signal,
closes its workers itself.
"""
import os
import signal
from functools import partial
from pathlib import Path

VEC_CHOICES = ("auto", "subproc", "dummy")


def _quiet_env(env_cls, **kwargs):
    """Runs inside the env's process: ignore the break/interrupt signals meant for the trainer."""
    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), signal.SIG_IGN)
            except ValueError:            # not the main thread (DummyVecEnv in a test runner)
                pass
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    return env_cls(**kwargs)


def resolve_vec(vec: str, n_envs: int) -> str:
    """auto -> subproc whenever there is more than one env copy.

    Measured with scripts/bench_vec.py (16-core PC, random actions, env steps/s):
        PyBullet net env, 4 envs:  dummy  806   subproc 1702   (8 envs: 840 vs 2258)
        onboard sim,      4 envs:  dummy 5245   subproc 6397   (8 envs: 5794 vs 8434)
        3 net runs at once, 4 subproc envs each: ~944 each, still above one dummy run.
    Even the cheap numpy onboard sim gains, so there is no env for which auto picks dummy."""
    if vec not in VEC_CHOICES:
        raise ValueError(f"vec must be one of {VEC_CHOICES}")
    if vec == "auto":
        return "subproc" if n_envs > 1 else "dummy"
    return vec


RANK_SEED_STRIDE = 1_000_003          # a prime, so per-env episode seed streams never overlap


def make_train_vec_env(env_cls, env_kwargs: dict, n_envs: int, seed: int,
                       wrapper_class=None, vec: str = "dummy", rank_seed_kwarg=None):
    """make_vec_env with the chosen VecEnv class. `vec` must already be resolved.

    rank_seed_kwarg: name of a constructor argument that seeds the env's OWN per-episode
    sequence (OnboardHoopEnv's `seed`: each reset() without a seed uses the next one). Every
    copy then gets env_kwargs[name] + rank * RANK_SEED_STRIDE. Without this, copies built with
    the same kwargs fly identical layouts from their second episode on."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if rank_seed_kwarg is not None:
        base = int(env_kwargs.get(rank_seed_kwarg, 0))

        def thunk(rank):
            kw = dict(env_kwargs, **{rank_seed_kwarg: base + rank * RANK_SEED_STRIDE})

            def _init():
                env = Monitor(_quiet_env(env_cls, **kw) if vec == "subproc" else env_cls(**kw))
                if wrapper_class is not None:
                    env = wrapper_class(env)
                env.action_space.seed(seed + rank)
                return env
            return _init
        fns = [thunk(i) for i in range(n_envs)]
        # No venv.seed(): that would force the first reset onto seed + rank instead of the
        # env's own stream; the streams above are already distinct and reproducible.
        return SubprocVecEnv(fns, start_method="spawn") if vec == "subproc" else DummyVecEnv(fns)

    if vec == "subproc":
        return make_vec_env(partial(_quiet_env, env_cls), env_kwargs=env_kwargs, n_envs=n_envs,
                            seed=seed, wrapper_class=wrapper_class,
                            vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method="spawn"))
    return make_vec_env(env_cls, env_kwargs=env_kwargs, n_envs=n_envs, seed=seed,
                        wrapper_class=wrapper_class, vec_env_cls=DummyVecEnv)


MODEL_NAMES = ("latest_model.zip", "best_model.zip", "final_model.zip", "bc_model.zip")


def resolve_model_path(path) -> Path:
    """A .zip, or a run / clone directory (latest -> best -> final -> bc_model, i.e. the policy
    as of the moment it stopped, which is what "continue training" means)."""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        for n in MODEL_NAMES:
            if (p / n).is_file():
                return p / n
        raise FileNotFoundError(f"no model zip in {p} (looked for {', '.join(MODEL_NAMES)})")
    raise FileNotFoundError(f"no such model or directory: {p}")


def _obs_width(space) -> int:
    return int(space.shape[-1])


def widen_policy_input(zp, new_space, insert_at, out_path):
    """Write a copy of the saved model `zp` whose policy takes `new_space`'s wider observation.

    The extra inputs are inserted at index `insert_at` of the old observation (negative counts
    from its end) and get all-zero weights in every first layer that reads the observation, so
    the widened policy acts and values exactly like the old one until training uses them. The
    optimizer's running moments belong to the old shapes, so they are dropped (Adam restarts them).
    Returns out_path."""
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

    data, params, variables = load_from_zip_file(str(zp), device="cpu")
    old_w, new_w = _obs_width(data["observation_space"]), _obs_width(new_space)
    k = new_w - old_w
    if k <= 0:
        raise ValueError(f"cannot widen a {old_w}-wide policy to {new_w}")
    at = insert_at if insert_at >= 0 else old_w + insert_at
    widened = 0
    for name, t in list(params["policy"].items()):
        if t.ndim == 2 and t.shape[1] == old_w and name.endswith(".0.weight"):
            zeros = torch.zeros(t.shape[0], k, dtype=t.dtype)
            params["policy"][name] = torch.cat([t[:, :at], zeros, t[:, at:]], dim=1)
            widened += 1
    if not widened:
        raise ValueError(f"no first layer of width {old_w} found in {zp}")
    if "policy.optimizer" in params:
        params["policy.optimizer"] = {"state": {},
                                      "param_groups": params["policy.optimizer"]["param_groups"]}
    data["observation_space"] = new_space
    save_to_zip_file(str(out_path), data=data, params=params, pytorch_variables=variables)
    return out_path


def load_for_resume(algo_cls, path, env, widen_at=None, **overrides):
    """Load a saved model onto a new env, refusing one whose spaces do not match.
    `overrides` (e.g. learning_rate) replace the saved hyper-parameters.

    widen_at: if the env's observation is wider than the model's, widen the model first
    (widen_policy_input, new inputs inserted at this index of the old observation) instead of
    refusing. Used when an env gains an observation field (net env: the lap "armed" flag;
    onboard: the "pushed" odometer)."""
    zp = resolve_model_path(path)
    if widen_at is not None:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(str(zp), device="cpu", load_data=True)
        old_w, new_w = _obs_width(data["observation_space"]), _obs_width(env.observation_space)
        if new_w > old_w:
            import tempfile
            tmp = Path(tempfile.mkdtemp(prefix="widen_")) / zp.name
            widen_policy_input(zp, env.observation_space, widen_at, tmp)
            print(f"[INFO] widened {zp.name} from {old_w} to {new_w} observation inputs "
                  f"(new inputs at index {widen_at}, zero weights: same actions to start)")
            zp_load = tmp
        else:
            zp_load = zp
    else:
        zp_load = zp
    # Passing env to load() (rather than set_env afterwards) rebuilds the rollout/replay
    # buffers for THIS env's number of copies; SB3 also checks the spaces match there.
    try:
        model = algo_cls.load(str(zp_load), env=env, device="cpu", custom_objects=overrides or None)
    except ValueError as e:
        raise ValueError(f"cannot continue from {zp}: {e} (a model from a different task, env "
                         f"or algorithm?)") from e
    return model, zp


def cpu_count() -> int:
    return os.cpu_count() or 1
