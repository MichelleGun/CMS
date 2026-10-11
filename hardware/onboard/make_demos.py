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


def _keep_upto(episode_infos, failed) -> int:
    """Lap mode: how many leading samples of an episode to keep. A failure at the end (a strike,
    the net) is cut off: keep up to the last pass and the turn-round after it, until the ring was
    back in view. An episode that just ran out of time is kept whole."""
    n = len(episode_infos)
    passes = [i for i, inf in enumerate(episode_infos) if inf.get("phase") == "passed"]
    if not passes:
        return 0
    if not failed:
        return n
    last = passes[-1]
    for j in range(last + 1, n):
        if episode_infos[j].get("ring_found"):
            return j + 1
    return last + 1


def generate(episodes=60, profile="typical", seed=0, keep_failures=False,
             render_frames=False, cfg: ExpertConfig = None, verbose=True,
             dr=0.0, sway_deg=15.0, hard_start_p=0.0, laps=1, wash_deg=0.0):
    """-> (DemoRecorder, tally dict). dr / sway_deg / hard_start_p / wash_deg: the randomised
    (robust) task, see sim.py; only the episodes the expert still passes are kept. laps=0: the
    expert flies laps (ExpertConfig.laps) and episodes run on after each pass; see _keep_upto."""
    env = OnboardHoopEnv(profile=profile, seed=seed, render_frames=render_frames,
                         dr=dr, sway_deg=sway_deg, hard_start_p=hard_start_p, laps=laps,
                         wash_deg=wash_deg)
    if laps != 1:
        from dataclasses import replace as _replace
        cfg = _replace(cfg or ExpertConfig(), laps=True)
    policy = ExpertPolicy(cfg)
    rec = DemoRecorder(
        meta={"source": "onboard_scripted_expert", "profile": profile,
              "robust": {"dr": dr, "sway_deg": sway_deg, "hard_start_p": hard_start_p,
                         "wash_deg": wash_deg},
              "laps": laps,
              "episodes_requested": episodes, "sim": True,
              "expert": (cfg or ExpertConfig()).to_json()},
        obs_dim=spec.OBS_DIM, task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    tally = {"passed": 0, "struck": 0, "missed": 0, "out_of_cage": 0, "timeout": 0}
    kept = 0
    lap_passes = []

    try:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            policy.reset()
            # Buffer the episode: whether to keep it is only known at the end.
            episode, infos, t = [], [], 0.0
            phase = "timeout"
            while True:
                action = policy(obs)
                episode.append((obs.copy(), np.asarray(action, dtype=np.float32), t))
                obs, _r, term, trunc, info = env.step(action)
                infos.append(info)
                t += spec.CTRL_DT
                if term or trunc:
                    phase = info["phase"] if term else "timeout"
                    break
            if laps != 1:
                lap_passes.append(int(info["passes"]))
                upto = len(episode) if keep_failures else _keep_upto(infos, failed=bool(term))
                tally["passed" if info["passes"] else phase] += 1
            else:
                tally[phase] = tally.get(phase, 0) + 1
                upto = len(episode) if (phase == "passed" or keep_failures) else 0
            if upto:
                for o, a, ts in episode[:upto]:
                    rec.add(o, a, a, ts)
                rec.next_episode()
                kept += 1
            if verbose and (e + 1) % max(1, episodes // 10) == 0:
                print(f"[make_demos] {e + 1}/{episodes} episodes, {kept} kept, "
                      f"{len(rec)} samples")
    finally:
        env.close()
    tally["kept_episodes"] = kept
    if lap_passes:
        tally["mean_passes"] = float(np.mean(lap_passes))
        tally["max_passes"] = int(max(lap_passes))
    return rec, tally


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh"], default="typical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", type=float, default=0.0,
                    help="domain randomisation 0..1: varied drone, a swinging ring, detector noise")
    ap.add_argument("--sway-deg", type=float, default=15.0)
    ap.add_argument("--hard-start-p", type=float, default=0.0,
                    help="share of episodes starting behind / beside / below the ring or anywhere")
    ap.add_argument("--laps", type=int, default=1,
                    help="0 = the expert flies laps (through, clear, turn round, find, again) and "
                         "each episode runs on after a pass; 1 = one pass per episode")
    ap.add_argument("--wash-deg", type=float, default=0.0,
                    help="prop-wash push of the ring (deg), see sim.py")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default: {DEMO_DIR}/onboard_sim_<n>.npz")
    ap.add_argument("--keep-failures", action="store_true",
                    help="also record episodes that did not pass through (a clone copies "
                         "its demonstrator, including its mistakes)")
    ap.add_argument("--render", action="store_true",
                    help="Put the real OpenCV detector in the loop instead of the fast "
                         "analytic model of it. Several times slower. It helped at the "
                         "50 cm ring (5/12 -> 10/12) but not at the measured 30 cm ring, "
                         "where the clone trained with it scored 0/12 (see the README)")
    a = ap.parse_args(argv)

    rec, tally = generate(a.episodes, a.profile, a.seed, a.keep_failures, a.render,
                          dr=a.dr, sway_deg=a.sway_deg, hard_start_p=a.hard_start_p,
                          laps=a.laps, wash_deg=a.wash_deg)
    if not len(rec):
        sys.exit("[make_demos] no samples recorded: the expert passed through nothing. "
                 "Check the expert's config, or pass --keep-failures to inspect it.")
    out = a.out or (DEMO_DIR / f"onboard_sim_{a.episodes}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec.meta.update(tally)
    path = rec.save(out)
    print(f"[make_demos] {len(rec)} samples from {tally['kept_episodes']} episodes, "
          f"{tally['passed']}/{a.episodes} passed"
          + (f", {tally['mean_passes']:.2f} passes/episode (max {tally['max_passes']})"
             if "mean_passes" in tally else "") + f" -> {path}")
    return path, tally


if __name__ == "__main__":
    main()
