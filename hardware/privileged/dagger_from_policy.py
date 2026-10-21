"""
One DAgger relabelling pass for a trained MEMORY policy (history + asymmetric critic, like n8 / O):
the policy flies, the privileged teacher labels every tick, and the labels are saved as a demo file
that joins the BC term of the next DAPG run (train_onboard.py --demos ...).

    python hardware/privileged/dagger_from_policy.py results/hoop/runs/w10_n9_O_oblique --tag mgO
    python hardware/privileged/dagger_from_policy.py <run> --episodes 2 --tag smoke      # smoke test

Why: n8 (the first policy that loops) came from a DAgger clone with memory, and its BC term is that
clone's 442k teacher labels, all gathered on the CLONE's states. The policy has since drifted (O:
slanted starts, 6M+ more steps), so the states it now visits, above all on the way round after a
pass, are not in those labels. Relabelling its own states is DAgger's core step (Ross et al. 2011).
With memory, the labels no longer average out (that is what sank the memoryless loop-labels run L).

Same collection as dagger_hist.py (privileged.dagger.collect: the student flies, the teacher takes
the sticks near the noodle or the net, margins of round --round) on the same env (dagger_hist.
env_kwargs: history 0,1,2,4,8,16, ring tracker, privileged critic inputs), so the observations are
full width and train_onboard.py takes them as they are. Writes
hardware/flights/demos/dagger_<tag>_r<k>.npz (k = the next free index).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent
CMS = HW.parent
for p in (HW, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from imitation.record_demo import DemoRecorder  # noqa: E402
from onboard import spec  # noqa: E402
from privileged.dagger import collect  # noqa: E402
from privileged.dagger_hist import DEMO_DIR, env_kwargs  # noqa: E402


def load_student(model_zip: Path):
    """The run's model as obs(full width) -> action, exactly as PPO trained it."""
    from onboard.aac_policy import MaskedActorCriticPolicy
    from onboard.train_onboard import BCRegularizedPPO
    model = BCRegularizedPPO.load(str(model_zip), device="cpu",
                                  custom_objects={"policy_class": MaskedActorCriticPolicy})

    def student(obs, m=model):
        act, _ = m.predict(np.asarray(obs, np.float32), deterministic=True)
        return np.clip(act, -1, 1)
    return student, int(model.observation_space.shape[0])


def relabel(run: Path, tag: str, episodes: int = 36, rnd: int = 6, seed: int = 80_000, log=print) -> Path:
    run = Path(run)
    model_zip = next((run / n for n in ("latest_model.zip", "best_model.zip", "final_model.zip")
                      if (run / n).is_file()), None)
    if model_zip is None:
        raise SystemExit(f"no model in {run}")
    student, width = load_student(model_zip)
    env_kw = env_kwargs()
    meta = dict(task=spec.TASK_NAME, source="privileged_teacher", profile="measured", dry_run=None,
                yaw_convention=spec.YAW_CONVENTION, dagger_round=rnd, obs_history=list(env_kw["history_lags"]),
                asym_critic=True, student=str(model_zip.relative_to(CMS)),
                note="teacher labels on a trained memory policy's own states (dagger_from_policy.py)")
    rec = DemoRecorder(meta=meta, obs_dim=width, task=spec.TASK_NAME, obs_shape=(width,))
    ticks, share = collect(student, rnd, env_kw, episodes, seed, rec, search_first=True)
    k = 0
    while (DEMO_DIR / f"dagger_{tag}_r{k}.npz").exists():
        k += 1
    out = DEMO_DIR / f"dagger_{tag}_r{k}.npz"
    rec.save(out)
    log(f"[RELABEL] {run.name}: {episodes} episodes, {ticks} ticks labelled "
        f"(teacher flew {share:.0%}) -> {out.relative_to(CMS)}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", type=int, default=36)
    ap.add_argument("--round", type=int, default=6, help="danger margins of this DAgger round (smaller = trust more)")
    ap.add_argument("--seed", type=int, default=80_000)
    a = ap.parse_args(argv)
    relabel(a.run if a.run.is_absolute() else CMS / a.run, a.tag, a.episodes, a.round, a.seed)


if __name__ == "__main__":
    main()
