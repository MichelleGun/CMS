"""
Measure env throughput (env steps per second) for DummyVecEnv vs SubprocVecEnv, so the
trainers' `--vec auto` choice is backed by numbers on this machine rather than a guess.

    python scripts/bench_vec.py                        # both envs, n_envs 1/4/8
    python scripts/bench_vec.py --env net --n-envs 4 --seconds 20
    python scripts/bench_vec.py --concurrent 3         # 3 copies at once, like 3 dashboard runs

Random actions, no learning: this isolates the simulator + VecEnv cost, which is what the
choice affects. (PPO's gradient step is the same either way.)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CMS = HERE.parent
for p in (HERE, CMS / "hardware"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vec_util import make_train_vec_env  # noqa: E402


def _env_spec(name):
    if name == "net":
        from gym_pybullet_drones.utils.enums import ActionType, ObservationType
        from hoop_net_env import HoopNetAviary, ROOM
        return HoopNetAviary, dict(obs=ObservationType.KIN, act=ActionType.VEL, room=ROOM)
    if name == "onboard":
        from onboard.gym_env import OnboardHoopEnv
        return OnboardHoopEnv, dict(profile="typical")
    raise ValueError(name)


def bench(env_name, n_envs, vec, seconds):
    cls, kw = _env_spec(env_name)
    venv = make_train_vec_env(cls, kw, n_envs, seed=0, vec=vec)
    try:
        venv.reset()
        rng = np.random.default_rng(0)
        shape = (n_envs,) + venv.action_space.shape
        for _ in range(20):                                   # warm-up
            venv.step(rng.uniform(-1, 1, shape).astype(np.float32))
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            venv.step(rng.uniform(-1, 1, shape).astype(np.float32))
            steps += n_envs
        return steps / (time.perf_counter() - t0)
    finally:
        venv.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", choices=["net", "onboard", "both"], default="both")
    ap.add_argument("--n-envs", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--vec", choices=["dummy", "subproc", "both"], default="both")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--concurrent", type=int, default=1,
                    help="run this many benchmark processes at once and report each one's rate")
    ap.add_argument("--json", action="store_true", help="print one JSON line per result")
    a = ap.parse_args()

    if a.concurrent > 1:
        argv = [sys.executable, __file__, "--env", a.env, "--vec", a.vec, "--seconds", str(a.seconds),
                "--json", "--n-envs", *map(str, a.n_envs)]
        procs = [subprocess.Popen(argv, stdout=subprocess.PIPE, text=True) for _ in range(a.concurrent)]
        outs = [p.communicate()[0] for p in procs]
        rows = {}
        for out in outs:
            for line in out.splitlines():
                if line.startswith("{"):
                    r = json.loads(line)
                    rows.setdefault((r["env"], r["vec"], r["n_envs"]), []).append(r["steps_per_s"])
        print(f"{a.concurrent} concurrent processes (steps/s each, then total):")
        for (env, vec, n), v in sorted(rows.items()):
            print(f"  {env:8s} {vec:8s} n_envs={n:<2d}  each {np.mean(v):8.0f}   total {np.sum(v):8.0f}")
        return

    envs = ["net", "onboard"] if a.env == "both" else [a.env]
    vecs = ["dummy", "subproc"] if a.vec == "both" else [a.vec]
    for env in envs:
        for n in a.n_envs:
            for vec in vecs:
                if vec == "subproc" and n == 1:
                    continue
                rate = bench(env, n, vec, a.seconds)
                if a.json:
                    print(json.dumps(dict(env=env, vec=vec, n_envs=n, steps_per_s=rate)), flush=True)
                else:
                    print(f"{env:8s} {vec:8s} n_envs={n:<2d}  {rate:8.0f} steps/s", flush=True)


if __name__ == "__main__":
    main()
