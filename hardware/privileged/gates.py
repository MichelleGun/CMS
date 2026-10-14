"""
Trust gates for a teacher: may it teach a student? Scores the privileged teacher, and the scripted
expert as a baseline, on the same episodes; writes hardware/privileged/reports/teacher_gates.md.

    python hardware/privileged/gates.py                 # full (a few minutes, one CPU core)
    python hardware/privileged/gates.py --episodes 5    # quick

THE GOAL: enter the hoop, loop round the outside, re-enter, as many times as possible (same-side
laps). The laps numbers come first; the single front-runway pass is a stepping stone.

G1 correctness   measured profile, dr 1, the training ring (5 deg sway, 15 deg prop wash):
                 same-side laps from all 9 zones: >= 90 % of 90 s episodes with at least 2 passes
                 and no strike / out of cage, AND >= 4 laps per 90 s on average;
                 front runway >= 95 % passes
G2 robustness    front runway >= 90 % passes with the dynamics deliberately wrong: forward and
                 sideways gain x0.6 and x1.5 (the number the logs could not measure), lag x2,
                 delay +0.1 s, yaw gain x0.7
G3 commands      every action in [-1, 1]; per-tick change <= 0.25; no yaw command inside the yaw
                 dead zone, no other stick inside the shared one (a command the drone ignores)

Every gate is scored twice: on onboard/sim.py as it is, and with the crossing fix of
privileged/sim_patches.py (as it is, centred passes can go unrecorded; see that file).
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent
if str(HW) not in sys.path:
    sys.path.insert(0, str(HW))

from onboard import sim as onboard_sim  # noqa: E402
from onboard import spec  # noqa: E402
from onboard.expert import ExpertConfig, ExpertPolicy  # noqa: E402
from onboard.gym_env import OnboardHoopEnv  # noqa: E402
from privileged.sim_patches import fixed_crossing  # noqa: E402
from privileged.teacher import PrivilegedTeacher  # noqa: E402

RING = dict(sway_deg=5.0, wash_deg=15.0, dr=1.0)        # what train_onboard uses
FRONT = dict(laps=1, start_mode="ring", **RING)
LAPS = dict(laps=0, lap_dir="same", start_mode="zones", **RING)
MEASURED = onboard_sim.PROFILES["measured"]
STRESS = {
    "forward/side x0.6": replace(MEASURED, name="stress", gain_fb=0.6, gain_lr=0.6),
    "forward/side x1.5": replace(MEASURED, name="stress", gain_fb=1.5, gain_lr=1.5),
    # keep=: survive the dr-1 re-draw (sim.sample_profile); before 2026-10-13 these two rows
    # silently tested ordinary randomised lag and delay
    "lag x2": replace(MEASURED, name="stress", tau_accel=2 * MEASURED.tau_accel,
                      tau_decel=2 * MEASURED.tau_decel, keep=("tau_accel", "tau_decel")),
    "delay +0.1 s": replace(MEASURED, name="stress", delay_s=MEASURED.delay_s + 0.1, keep=("delay_s",)),
    "yaw gain x0.7": replace(MEASURED, name="stress", gain_yaw=0.7 * MEASURED.gain_yaw),
}
OUT = HERE / "reports"


class ExpertAdapter:
    """The obs-only scripted expert behind the same (env, obs) interface."""

    def __init__(self, laps=False):
        self.pol = ExpertPolicy(ExpertConfig(laps=True, lap_same=True) if laps else None)

    def reset(self):
        self.pol.reset()

    def __call__(self, env, obs):
        return self.pol(obs)


def run(controller, env_kw, profile, episodes, seed=7000):
    """-> dict(outcomes, passes per episode, actions) over `episodes` fixed layouts."""
    env = OnboardHoopEnv(seed=seed, profile=profile, **env_kw)
    outcomes, passes, acts = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(controller, "reset"):
            controller.reset()
        ep_acts = []
        while True:
            a = np.asarray(controller(env, obs), dtype=float)
            ep_acts.append(a)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                outcomes.append(info["phase"] if term else "timeout")
                passes.append(int(info.get("passes", 0)))
                break
        acts.append(np.array(ep_acts))
    return dict(outcomes=outcomes, passes=passes, acts=acts)


def command_checks(acts, profile):
    a = np.concatenate(acts) if acts else np.zeros((0, 4))
    d = np.concatenate([np.abs(np.diff(x, axis=0)) for x in acts if len(x) > 1]) if acts else np.zeros((0, 4))
    yaw_dead = profile.yaw_deadband if profile.yaw_deadband is not None else profile.deadband
    yaw = a[:, spec.ACT_YAW]
    others = a[:, [spec.ACT_LR, spec.ACT_FB, spec.ACT_UD]]
    return dict(in_range=bool(np.all(np.abs(a) <= 1.0 + 1e-6)),
                max_step=float(d.max()) if len(d) else 0.0,
                yaw_in_dead=float(np.mean((yaw != 0) & (np.abs(yaw) < yaw_dead - 1e-6))) if len(yaw) else 0.0,
                other_in_dead=float(np.mean((others != 0) & (np.abs(others) < profile.deadband - 1e-6)))
                if len(others) else 0.0)


def score(controller_factory, episodes, fixed):
    res = {}
    with fixed_crossing(fixed):
        f = run(controller_factory(False), FRONT, MEASURED, episodes)
        res["front_pass"] = float(np.mean([o == "passed" for o in f["outcomes"]]))
        res["front_outcomes"] = {k: f["outcomes"].count(k) for k in sorted(set(f["outcomes"]))}
        lap_eps = max(9, episodes // 2 // 9 * 9)                     # every zone equally often
        lp = run(controller_factory(True), LAPS, MEASURED, lap_eps)
        good = [p >= 2 and o not in ("struck", "out_of_cage") for p, o in zip(lp["passes"], lp["outcomes"])]
        res["laps_good"] = float(np.mean(good))
        res["laps_mean_passes"] = float(np.mean(lp["passes"]))
        # an episode's passes all come before its first failure (a strike / the net ends it)
        res["laps_best_streak"] = int(max(lp["passes"]))
        res["laps_median_streak"] = float(np.median(lp["passes"]))
        res["laps_outcomes"] = {k: lp["outcomes"].count(k) for k in sorted(set(lp["outcomes"]))}
        res["stress"] = {}
        for name, prof in STRESS.items():
            s = run(controller_factory(False), FRONT, prof, episodes)
            res["stress"][name] = float(np.mean([o == "passed" for o in s["outcomes"]]))
        res["commands"] = command_checks(f["acts"] + lp["acts"], MEASURED)
    c = res["commands"]
    res["G1"] = res["laps_good"] >= 0.90 and res["laps_mean_passes"] >= 4.0 and res["front_pass"] >= 0.95
    res["G2"] = min(res["stress"].values()) >= 0.90
    res["G3"] = c["in_range"] and c["max_step"] <= 0.25 + 1e-6 and c["yaw_in_dead"] == 0 and c["other_in_dead"] == 0
    return res


def render(results, episodes):
    yes = lambda b: "**pass**" if b else "fail"
    L = ["# Teacher trust gates", "",
         f"{episodes} front-runway episodes per row (same layouts for every controller), measured profile, dr 1, "
         "5° sway, 15° prop wash. Generated by `hardware/privileged/gates.py`.", "",
         "**The goal: enter, loop round, re-enter, as many times as possible.** Laps columns first.", "",
         "| Controller | Sim | Laps per 90 s | Best / median streak | Clean lap episodes (≥2, no crash) | Front-runway pass | G2 worst stress case | G3 commands | Gates |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (name, simname), r in results.items():
        worst = min(r["stress"], key=r["stress"].get)
        c = r["commands"]
        g3 = (f"max step {c['max_step']:.2f}, yaw in dead zone {c['yaw_in_dead']:.0%}, "
              f"others in dead zone {c['other_in_dead']:.0%}")
        L.append(f"| {name} | {simname} | **{r['laps_mean_passes']:.1f}** | {r['laps_best_streak']} / {r['laps_median_streak']:.0f} | "
                 f"{r['laps_good']:.0%} {r['laps_outcomes']} | {r['front_pass']:.0%} | {r['stress'][worst]:.0%} ({worst}) | {g3} | "
                 f"G1 {yes(r['G1'])}, G2 {yes(r['G2'])}, G3 {yes(r['G3'])} |")
    L += ["", "## G2 detail (front-runway passes per stress case)", "",
          "| Controller | Sim | " + " | ".join(STRESS) + " |", "|---|---|" + "---|" * len(STRESS)]
    for (name, simname), r in results.items():
        L.append(f"| {name} | {simname} | " + " | ".join(f"{r['stress'][k]:.0%}" for k in STRESS) + " |")
    L += ["", "\"fixed crossing\" = onboard/sim.py with the crossing check of `privileged/sim_patches.py`: as the sim is, "
          "a centred pass can go unrecorded when the prop-washed ring swings back at the moment of crossing.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    controllers = {"privileged teacher": lambda laps: PrivilegedTeacher(),
                   "scripted expert": lambda laps: ExpertAdapter(laps)}
    results = {}
    for name, factory in controllers.items():
        for simname, fixed in (("as is", False), ("fixed crossing", True)):
            r = score(factory, a.episodes, fixed)
            results[(name, simname)] = r
            print(f"[GATES] {name:20s} {simname:15s} laps/90s {r['laps_mean_passes']:.1f} clean {r['laps_good']:.0%} "
                  f"best streak {r['laps_best_streak']}  front {r['front_pass']:.0%} "
                  f"worst stress {min(r['stress'].values()):.0%}  G1 {r['G1']} G2 {r['G2']} G3 {r['G3']}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "teacher_gates.json").write_text(json.dumps({f"{k[0]} | {k[1]}": v for k, v in results.items()},
                                                         indent=2), encoding="utf-8")
    (a.out / "teacher_gates.md").write_text(render(results, a.episodes), encoding="utf-8")
    print(f"[DONE] {a.out / 'teacher_gates.md'}")
    return results


if __name__ == "__main__":
    main()
