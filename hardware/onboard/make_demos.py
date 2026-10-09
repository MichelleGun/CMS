"""
Generate demonstrations for the onboard-camera task in the simulator, with the scripted
expert as the demonstrator.

    python hardware/onboard/make_demos.py --episodes 60 \
        --out hardware/flights/demos/onboard_sim.npz
    python hardware/imitation/bc_train.py --demos hardware/flights/demos/onboard_sim.npz \
        --out hardware/flights/bc/onboard_clone --eval-episodes 20

The counterpart of hardware/imitation/make_sim_demos.py, which does the same thing for
the simulator's point-reaching task. The reason to want it: it closes the learning loop
with no drone and no flying, so behaviour cloning and RL can be developed, tested and
compared before anything with propellers is involved. Demonstrations recorded on the real
drone (fly_hoop.py --record) land in the same format and can be mixed in or used alone.

Only successful episodes are kept by default. A clone copies its demonstrator's mistakes,
so feeding it failed approaches teaches it to fail.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HW_ROOT = Path(__file__).resolve().parents[1]
if str(HW_ROOT) not in sys.path:
    sys.path.insert(0, str(HW_ROOT))

# flake8: noqa: E402
from imitation.record_demo import DEMO_DIR, DemoRecorder
from onboard import spec
from onboard.expert import ExpertConfig, ExpertPolicy
from onboard.gym_env import OnboardHoopEnv


def generate(episodes=60, profile="typical", seed=0, keep_failures=False,
             render_frames=False, cfg: ExpertConfig = None, verbose=True):
    """-> (DemoRecorder, tally dict)."""
    env = OnboardHoopEnv(profile=profile, seed=seed, render_frames=render_frames)
    policy = ExpertPolicy(cfg)
    rec = DemoRecorder(
        meta={"source": "onboard_scripted_expert", "profile": profile,
              "episodes_requested": episodes, "sim": True,
              "expert": (cfg or ExpertConfig()).to_json()},
        obs_dim=spec.OBS_DIM, task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    tally = {"passed": 0, "struck": 0, "missed": 0, "out_of_cage": 0, "timeout": 0}
    kept = 0

    try:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            policy.reset()
            # Buffer the episode: whether to keep it is only known at the end.
            episode, t = [], 0.0
            phase = "timeout"
            while True:
                action = policy(obs)
                episode.append((obs.copy(), np.asarray(action, dtype=np.float32), t))
                obs, _r, term, trunc, info = env.step(action)
                t += spec.CTRL_DT
                if term or trunc:
                    phase = info["phase"] if term else "timeout"
                    break
            tally[phase] = tally.get(phase, 0) + 1
            if phase == "passed" or keep_failures:
                for o, a, ts in episode:
                    rec.add(o, a, a, ts)
                rec.next_episode()
                kept += 1
            if verbose and (e + 1) % max(1, episodes // 10) == 0:
                print(f"[make_demos] {e + 1}/{episodes} episodes, {kept} kept, "
                      f"{len(rec)} samples")
    finally:
        env.close()
    tally["kept_episodes"] = kept
    return rec, tally


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh"], default="typical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default: {DEMO_DIR}/onboard_sim_<n>.npz")
    ap.add_argument("--keep-failures", action="store_true",
                    help="also record episodes that did not pass through (a clone copies "
                         "its demonstrator, including its mistakes)")
    ap.add_argument("--render", action="store_true",
                    help="RECOMMENDED. Put the real OpenCV detector in the loop instead "
                         "of the fast analytic model of it. Several times slower, and it "
                         "matters: a clone trained without it scored 5/12 against the "
                         "real detector and 10/12 with it (see the README)")
    a = ap.parse_args(argv)

    rec, tally = generate(a.episodes, a.profile, a.seed, a.keep_failures, a.render)
    if not len(rec):
        sys.exit("[make_demos] no samples recorded: the expert passed through nothing. "
                 "Check the expert's config, or pass --keep-failures to inspect it.")
    out = a.out or (DEMO_DIR / f"onboard_sim_{a.episodes}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec.meta.update(tally)
    path = rec.save(out)
    print(f"[make_demos] {len(rec)} samples from {tally['kept_episodes']} episodes, "
          f"{tally['passed']}/{a.episodes} passed -> {path}")
    return path, tally


if __name__ == "__main__":
    main()
