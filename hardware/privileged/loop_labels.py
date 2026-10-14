"""
Teacher labels for the loop leg, on the states OUR policy gets into (one DAgger-style relabelling
pass). Used as a small, decaying imitation term inside PPO (train_onboard.py --demos), so the RL
reward stays in charge and the teacher only shows the way round and back to the ring.

    python hardware/privileged/loop_labels.py results/hoop/runs/w10_laps4_P_pure_more/latest_model.zip
    python hardware/privileged/loop_labels.py <model.zip> --episodes 4 --out /tmp/x.npz     # smoke test

Why: after its first pass the RL policy survives 20-80 s but sees the ring on only 0-3 % of the
ticks: it never turns back to find it (2026-10-14 diagnosis). RL alone rarely stumbles on "go round,
turn, find the ring again", because the reward for it (the next pass) is far away.

How: the student (a saved policy, loaded the way fly_hoop.py loads it) flies the laps task, starting
just in front of the ring (near_pass_p = 1) so every episode reaches the loop quickly. Every tick
AFTER the first pass, the privileged teacher's action is recorded as the label for what the student
saw. Close to the noodle or the net the teacher takes the sticks for 1 s (dagger.danger_margins), so
episodes carry on. Pre-pass ticks are not recorded: the policy already passes well, and the
teacher's precise final push is what earlier pure-imitation students failed to copy.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent
if str(HW) not in sys.path:
    sys.path.insert(0, str(HW))

from imitation.record_demo import DemoRecorder  # noqa: E402
from onboard import spec  # noqa: E402
from onboard.gym_env import OnboardHoopEnv  # noqa: E402
from onboard.policy import load_policy  # noqa: E402
from onboard.sim import CAGE_XY  # noqa: E402
from privileged.dagger import DEMO_DIR, TAKEOVER_TICKS, danger_margins, make_labeller  # noqa: E402
from privileged.gates import LAPS, MEASURED  # noqa: E402


def collect(model, episodes=60, seed=70_000, danger_round=3, out=None, log=print):
    fn = load_policy(model)
    if fn.obs_width != spec.TRACKER_OBS_DIM:
        raise SystemExit(f"{model}: expected a ring-tracker policy ({spec.TRACKER_OBS_DIM} inputs)")
    label = make_labeller(search_first=False)
    env = OnboardHoopEnv(seed=seed, profile=MEASURED, near_pass_p=1.0, **dict(LAPS, ring_tracker=True))
    rim_m, net_m = danger_margins(danger_round)
    meta = dict(task=spec.TASK_NAME, source="teacher_loop", profile="measured", dry_run=None,
                yaw_convention=spec.YAW_CONVENTION, student=str(model), near_pass_p=1.0, obs_live=True,
                note="teacher labels on the student's own post-pass states (loop_labels.py)")
    rec = DemoRecorder(meta=meta, obs_dim=spec.TRACKER_OBS_DIM, task=spec.TASK_NAME,
                       obs_shape=(spec.TRACKER_OBS_DIM,))
    ticks = taken = passed = second = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        hold, n_pass = 0, 0
        while True:
            if n_pass >= 1:
                lab = np.asarray(label(env), dtype=np.float32)
                p = env.sim.pos
                if (env.sim.rim_distance() < rim_m
                        or CAGE_XY - max(abs(p[0]), abs(p[1])) < net_m):
                    hold = TAKEOVER_TICKS
                if hold > 0:
                    act, hold = lab, hold - 1
                    taken += 1
                else:
                    act = fn(obs)
                rec.add(obs, lab, act, env.sim.t)
                ticks += 1
            else:
                act = fn(obs)
            obs, _, term, trunc, info = env.step(act)
            if info["phase"] == "passed":
                n_pass += 1
                passed += n_pass == 1
                second += n_pass == 2
            if term or trunc:
                break
        rec.next_episode()
    if not len(rec):
        raise SystemExit("the student never passed, so there is no loop leg to label")
    out = Path(out) if out else DEMO_DIR / f"teacher_loop_{Path(model).parent.name}_{time.strftime('%Y%m%d-%H%M')}.npz"
    rec.save(out)
    log(f"[LOOP-LABELS] {episodes} episodes: student passed in {passed}, passed again in {second}; "
        f"{ticks} post-pass ticks labelled (teacher flew {taken / max(ticks, 1):.0%}) -> {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=70_000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    print(collect(a.model, a.episodes, a.seed, out=a.out))


if __name__ == "__main__":
    main()
