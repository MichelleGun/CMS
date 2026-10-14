"""
DAgger with the privileged teacher (Ross et al. 2011; as used by Loquercio et al. 2021): teach the
camera-only student to enter the hoop, loop round, and re-enter, as many times as possible.

    python hardware/privileged/dagger.py                         # 6 rounds (~1-2 h, one core)
    python hardware/privileged/dagger.py --rounds 2 --episodes 3 --eval-episodes 3 --tag smoke

Round 0: the teacher flies; every tick, (what the student would see, what the teacher does) is
recorded. A student (the usual 37-input clone, imitation/bc_train.py) is fitted to it.
Round k: the STUDENT flies; the teacher still labels every tick (that is the point of DAgger: the
labels come from the states the student itself gets into, its own mistakes included). If the
student gets into danger (too close to the noodle or the net), the teacher takes the sticks for a
moment, so an episode is not wasted on a crash; the danger margins shrink every round (the handover
distance of Loquercio et al.), so later rounds let the student fly closer to the edge.
All rounds' data are pooled and the student is refitted from scratch each round.

After each round the student is scored on the goal (same-side laps from all 9 zones, 90 s):
laps per 90 s, clean episodes, best streak. G4 = student >= 80 % of the teacher's laps / 90 s.
Writes hardware/privileged/reports/dagger[_tag].md and the clones hardware/flights/bc/dagger[_tag]_rN/
(normal clone folders: train_onboard.py --init and fly_hoop.py take them as they are).
"""
import argparse
import json
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
from onboard.sim import CAGE_XY  # noqa: E402
from privileged.gates import FRONT, LAPS, MEASURED, run  # noqa: E402
from privileged.teacher import PrivilegedTeacher  # noqa: E402

DEMO_DIR = HW / "flights" / "demos"
BC_DIR = HW / "flights" / "bc"
REPORTS = HERE / "reports"
TAKEOVER_TICKS = 15            # once the teacher takes over, it keeps the sticks this long (1 s)
# --search-first: until the student's camera has seen the ring once this episode, the label is a
# fixed search the student can reproduce from its own inputs (hold position, turn clockwise) instead
# of the teacher's privileged "turn towards where I know the ring is". Without it the clone averages
# left and right turns at the blind start and flies into the net (measured 2026-10-13: 4 of 6 episodes
# ended within 6 s without ever seeing the ring).
SEARCH_ACTION = np.array([0.0, 0.0, 0.0, 0.8], np.float32)     # [lr, fb, ud, yaw]: 0.45 -> 1.9 laps/90 s, 0.8 -> 2.6


SEARCH_HEIGHT_M = 1.2            # training rings hang at 1.1-1.3 m


def make_labeller(search_first):
    teacher = PrivilegedTeacher()

    def label(env):
        if search_first and not env.builder.tracker.valid:
            # turn, and climb to the usual ring height: from a low start close to the ring, the ring
            # sits ~45 deg up, above the camera's ~27 deg half-height, and turning alone never finds it
            a = SEARCH_ACTION.copy()
            h = float(env.sim.state().get("h", 0)) / 100.0          # the height the student also sees
            a[spec.ACT_UD] = float(np.clip((SEARCH_HEIGHT_M - h) * 2.0, -0.6, 0.6))
            return a
        return teacher(env)
    return label


def danger_margins(rnd: int):
    """(rim m, net m): the teacher takes over when the drone gets this close to the noodle / the
    net. Shrinks every round, so the student is trusted closer to the edge as it improves."""
    return max(0.0, 0.06 - 0.012 * rnd), max(0.05, 0.20 - 0.03 * rnd)


def collect(student, rnd, env_kw, episodes, seed, recorder, search_first=False):
    """Fly `episodes`, recording (obs, teacher action) every tick. -> (ticks, share flown by teacher)."""
    teacher = make_labeller(search_first)
    env = OnboardHoopEnv(seed=seed, profile=MEASURED, **env_kw)
    rim_m, net_m = danger_margins(rnd)
    ticks = teacher_ticks = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if student is not None and hasattr(student, "reset"):
            student.reset()
        hold = 0
        while True:
            label = teacher(env)
            if student is None:
                act = label
            else:
                p = env.sim.pos
                danger = (env.sim.rim_distance() < rim_m
                          or CAGE_XY - max(abs(p[0]), abs(p[1])) < net_m)
                if danger:
                    hold = TAKEOVER_TICKS
                if hold > 0:
                    act, hold = label, hold - 1
                else:
                    act = np.asarray(student(obs), dtype=np.float32)
            teacher_ticks += int(act is label)
            recorder.add(obs, label, act, env.sim.t)
            obs, _, term, trunc, _ = env.step(act)
            ticks += 1
            if term or trunc:
                break
        recorder.next_episode()
    return ticks, teacher_ticks / max(ticks, 1)


def fit(demo_files, out_dir, epochs, seed, ring_tracker=False):
    from imitation.bc_train import main as bc_main
    return bc_main(["--demos", *map(str, demo_files), "--out", str(out_dir), "--epochs", str(epochs),
                    "--seed", str(seed), "--no-clearml", *(["--ring-tracker"] if ring_tracker else [])])


class Student:
    """A clone folder as an obs -> action callable (onboard/policy.py, deterministic)."""

    def __init__(self, clone_dir):
        from onboard.policy import load_policy
        self.fn = load_policy(Path(clone_dir) / "bc_model.zip", algo="ppo")

    def __call__(self, obs, *_):
        return self.fn(obs)


def score_laps(controller, episodes, seed=7000, laps_kw=LAPS):
    r = run(controller, laps_kw, MEASURED, episodes, seed=seed)
    good = [p >= 2 and o not in ("struck", "out_of_cage") for p, o in zip(r["passes"], r["outcomes"])]
    return dict(laps=float(np.mean(r["passes"])), clean=float(np.mean(good)), best=int(max(r["passes"])),
                outcomes={k: r["outcomes"].count(k) for k in sorted(set(r["outcomes"]))})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=6, help="rounds after round 0")
    ap.add_argument("--episodes", type=int, default=18, help="laps episodes collected per round (90 s each)")
    ap.add_argument("--front-episodes", type=int, default=20, help="front-runway episodes in round 0")
    ap.add_argument("--eval-episodes", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--search-first", action="store_true",
                    help="label a fixed search (hold, turn clockwise) until the student has seen the ring")
    ap.add_argument("--ring-tracker", action="store_true",
                    help="the student also gets the 4 ring-tracker inputs (onboard.obs.RingTracker)")
    a = ap.parse_args(argv)
    name = "dagger" + (f"_{a.tag}" if a.tag else "")
    laps_kw, front_kw = dict(LAPS), dict(FRONT)
    width = spec.OBS_DIM
    if a.ring_tracker:
        laps_kw["ring_tracker"] = front_kw["ring_tracker"] = True
        width = spec.TRACKER_OBS_DIM
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"{name}.md"
    t0 = time.time()

    teacher_score = score_laps(lambda env, obs: PrivilegedTeacher()(env), a.eval_episodes)
    print(f"[DAGGER] teacher: {teacher_score}", flush=True)
    rows, demo_files, student = [], [], None
    for rnd in range(a.rounds + 1):
        meta = dict(task=spec.TASK_NAME, source="privileged_teacher", profile="measured", dry_run=None,
                    yaw_convention=spec.YAW_CONVENTION, dagger_round=rnd)
        rec = DemoRecorder(meta=meta, obs_dim=width, task=spec.TASK_NAME, obs_shape=(width,))
        ticks, share = collect(student, rnd, laps_kw, a.episodes, 50_000 + 1000 * rnd, rec, a.search_first)
        if rnd == 0 and a.front_episodes:
            t2, _ = collect(None, rnd, front_kw, a.front_episodes, 60_000, rec, a.search_first)
            ticks += t2
        demo = DEMO_DIR / f"{name}_r{rnd}.npz"
        rec.save(demo)
        demo_files.append(demo)
        out = BC_DIR / f"{name}_r{rnd}"
        fit(demo_files, out, a.epochs, a.seed, a.ring_tracker)
        student = Student(out)
        s = score_laps(lambda env, obs, st=student: st(obs), a.eval_episodes, laps_kw=laps_kw)
        rows.append(dict(round=rnd, ticks=ticks, teacher_share=share, rim_m=danger_margins(rnd)[0],
                         clone=str(out.relative_to(HW.parent)), **s))
        print(f"[DAGGER] round {rnd}: {ticks} ticks (teacher flew {share:.0%}), student laps/90s {s['laps']:.2f} "
              f"clean {s['clean']:.0%} best {s['best']}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        report.write_text(render(rows, teacher_score, a), encoding="utf-8")
    (REPORTS / f"{name}.json").write_text(json.dumps(dict(teacher=teacher_score, rounds=rows), indent=2),
                                          encoding="utf-8")
    print(f"[DONE] {report}")
    return rows


def render(rows, teacher, a):
    g4 = 0.8 * teacher["laps"]
    L = ["# DAgger: teaching the camera-only student to loop the hoop", "",
         f"Goal: enter, loop round the outside, re-enter, as many times as possible (same-side laps, all 9 start "
         f"zones, 90 s, measured profile, dr 1). Each round: {a.episodes} laps episodes; scored on "
         f"{a.eval_episodes}. Generated by `hardware/privileged/dagger.py`.", "",
         f"**Teacher** (sees the true state): {teacher['laps']:.2f} laps / 90 s, clean {teacher['clean']:.0%}, "
         f"best streak {teacher['best']}. **G4** = student ≥ {g4:.2f} laps / 90 s (80 % of the teacher).", "",
         "| Round | Ticks collected | Teacher flew | Handover at (noodle) | Student laps / 90 s | Clean | Best streak | G4 | Clone |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['round']} | {r['ticks']} | {r['teacher_share']:.0%} | {r['rim_m'] * 100:.0f} cm | "
                 f"**{r['laps']:.2f}** | {r['clean']:.0%} | {r['best']} | {'✅' if r['laps'] >= g4 else '–'} | "
                 f"`{r['clone']}` |")
    L += ["", "Round 0 = the teacher flew every tick (plain cloning). Later rounds: the student flies and the "
          "teacher only labels, taking the sticks for 1 s when the student gets within the handover margin "
          "of the noodle or the net.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
