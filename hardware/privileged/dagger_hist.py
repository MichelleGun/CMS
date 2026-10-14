"""
DAgger v2 (2026-10-14 overnight): the privileged teacher teaches a student WITH MEMORY.

    python hardware/privileged/dagger_hist.py --tag n1                 # ~1-2 h, one core
    python hardware/privileged/dagger_hist.py --rounds 1 --episodes 2 --eval-episodes 9 --tag smoke

Method: privileged expert -> sensorimotor student by DAgger (Ross et al. 2011, "A Reduction of
Imitation Learning and Structured Prediction to No-Regret Online Learning"), the recipe of
Loquercio et al. 2021 ("Learning High-Speed Flight in the Wild", arXiv:2110.05113) and Chen et al.
2019 ("Learning by Cheating", arXiv:1912.12294). Same loop as dagger.py (teacher labels every tick,
the student flies from round 1, the teacher takes the sticks near the noodle / net with margins that
shrink each round). What changed, and why dagger.py's students stalled at 0-0.5 laps / 90 s:

  * the student sees a HISTORY of observations (onboard/history.py, lags 0,1,2,4,8,16 ticks) on top
    of the ring tracker: on the way round the outside the ring is out of view and the velocity input
    is blank, so a memoryless student cannot tell where on the loop it is;
  * a 256x256 network (the teacher's behaviour has several phases; 64x64 underfit it);
  * the student is a MaskedActorCriticPolicy PPO model from the start (the critic also sees the true
    state, aac_policy.py), so train_onboard.py --init <clone> can fine-tune it with PPO directly,
    and fly_hoop.py --mode policy flies it as is (policy.py stacks the history on the drone).

Each round's clone: hardware/flights/bc/dagger_<tag>_rN/ (bc_model.zip + meta.json, a normal clone
folder). Demos: hardware/flights/demos/dagger_<tag>_rN.npz (full-width observations, usable as
train_onboard --demos for the same env). Report: hardware/privileged/reports/dagger_<tag>.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent
CMS = HW.parent
for p in (HW, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from imitation.record_demo import DemoRecorder, load_demos  # noqa: E402
from onboard import spec  # noqa: E402
from onboard.gym_env import OnboardHoopEnv  # noqa: E402
from onboard.history import DEFAULT_LAGS, PRIV_DIM  # noqa: E402
from privileged.dagger import collect, danger_margins  # noqa: E402
from privileged.gates import MEASURED  # noqa: E402
from privileged.teacher import PrivilegedTeacher  # noqa: E402

DEMO_DIR = HW / "flights" / "demos"
BC_DIR = HW / "flights" / "bc"
REPORTS = HERE / "reports"


def env_kwargs(lags=DEFAULT_LAGS, sway_deg=2.0, lap_dir="same", laps=0, start_mode="zones"):
    return dict(laps=laps, lap_dir=lap_dir, start_mode=start_mode, sway_deg=sway_deg, wash_deg=15.0,
                dr=1.0, ring_tracker=True, history_lags=tuple(lags), privileged=True)


def new_student(env_kw, net_arch=(256, 256), seed=0, log_std=-1.2):
    """A BCRegularizedPPO with the masked asymmetric policy, on a throwaway env of the right shape."""
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from onboard.aac_policy import MaskedActorCriticPolicy
    from onboard.train_onboard import BCRegularizedPPO
    env = OnboardHoopEnv(profile=MEASURED, seed=seed, **env_kw)
    actor_dim = env.actor_dim
    vec = DummyVecEnv([lambda: env])
    model = BCRegularizedPPO(MaskedActorCriticPolicy, vec, device="cpu", seed=seed, verbose=0,
                             policy_kwargs=dict(net_arch=list(net_arch), actor_dim=actor_dim))
    with torch.no_grad():
        model.policy.log_std.fill_(log_std)
    return model


def fit(model, obs, act, epochs, batch=512, lr=1e-3, seed=0):
    """Supervised: the policy's mean action -> the teacher's label (MSE), actor parameters only."""
    import torch
    pol = model.policy
    params = (list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    X = torch.as_tensor(np.asarray(obs, np.float32))
    Y = torch.as_tensor(np.clip(np.asarray(act, np.float32), -1, 1))
    rng = np.random.default_rng(seed)
    n = len(X)
    pol.set_training_mode(True)
    last = 0.0
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = torch.as_tensor(idx[i:i + batch])
            feats = pol.pi_features_extractor(X[b])
            mean = pol.action_net(pol.mlp_extractor.forward_actor(feats))
            loss = ((mean - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        last = tot / n
        # cosine-ish decay: big steps first, fine at the end
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (ep + 1) / epochs)))
    pol.set_training_mode(False)
    return last


def save_clone(model, out_dir: Path, env_kw, extra: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "bc_model.zip"))
    lags = list(env_kw["history_lags"])
    frame = spec.TRACKER_OBS_DIM if env_kw.get("ring_tracker") else spec.OBS_DIM
    meta = {"source": "bc", "algo": "ppo", "task": spec.TASK_NAME, "env": "onboard",
            "obs_dim": int(model.observation_space.shape[0]), "frame_dim": frame,
            "ring_tracker": bool(env_kw.get("ring_tracker")), "obs_history": lags,
            "asym_critic": True, "priv_dim": PRIV_DIM, "actor_dim": frame * len(lags),
            "yaw_convention": spec.YAW_CONVENTION, "lap_dir": env_kw["lap_dir"], "laps": env_kw["laps"],
            "method": "DAgger (privileged teacher -> history student)", **extra}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def score(controller, env_kw, episodes, seed=7000):
    """Same-side laps from every zone (episodes rounded up to a multiple of 9): per-episode passes,
    outcome and start zone. `controller(env, obs) -> action`."""
    episodes = int(np.ceil(episodes / 9) * 9)
    env = OnboardHoopEnv(seed=seed, profile=MEASURED, **env_kw)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(controller, "reset"):
            controller.reset()
        zone = env.sim.start_type
        while True:
            obs, _, term, trunc, info = env.step(np.asarray(controller(env, obs), dtype=float))
            if term or trunc:
                rows.append(dict(zone=zone, passes=int(info.get("passes", 0)),
                                 outcome=info["phase"] if term else "timeout"))
                break
    return summarise(rows)


def summarise(rows):
    passes = [r["passes"] for r in rows]
    clean = [r["passes"] >= 2 and r["outcome"] not in ("struck", "out_of_cage") for r in rows]
    zones = {}
    for r in rows:
        zones.setdefault(r["zone"], []).append(r["passes"])
    return dict(laps=float(np.mean(passes)), clean=float(np.mean(clean)), best=int(max(passes)),
                min_zone=float(min(np.mean(v) for v in zones.values())),
                zones={k: float(np.mean(v)) for k, v in sorted(zones.items())},
                outcomes={k: sum(r["outcome"] == k for r in rows) for k in sorted({r["outcome"] for r in rows})})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=10, help="rounds after round 0")
    ap.add_argument("--episodes", type=int, default=27, help="laps episodes collected per round (90 s each)")
    ap.add_argument("--eval-episodes", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="hist")
    ap.add_argument("--lap-dir", choices=["same", "alternate"], default="same")
    ap.add_argument("--lags", default=",".join(map(str, DEFAULT_LAGS)))
    ap.add_argument("--net-arch", default="256,256")
    ap.add_argument("--start-round", type=int, default=0,
                    help="continue an earlier run of this --tag: its demos r0..r(K-1) are pooled and its "
                         "clone r(K-1) is the student (the fit then starts from it, so fewer --epochs do)")
    ap.add_argument("--no-search-first", action="store_true",
                    help="label the teacher's own search too (default: a fixed search the student can copy)")
    a = ap.parse_args(argv)
    lags = tuple(int(x) for x in a.lags.split(","))
    env_kw = env_kwargs(lags, lap_dir=a.lap_dir)
    name = f"dagger_{a.tag}"
    REPORTS.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"{name}.md"
    t0 = time.time()

    teacher = score(lambda env, obs: PrivilegedTeacher()(env), env_kw, a.eval_episodes)
    print(f"[DAGGER] teacher: {teacher}", flush=True)
    model = new_student(env_kw, tuple(int(x) for x in a.net_arch.split(",")), a.seed)
    width = int(model.observation_space.shape[0])
    rows, demo_files, student = [], [], None
    best = None
    k0 = a.start_round
    if k0 > 0:
        import torch
        prev = BC_DIR / f"{name}_r{k0 - 1}" / "bc_model.zip"
        from onboard.train_onboard import BCRegularizedPPO
        from onboard.aac_policy import MaskedActorCriticPolicy
        old = BCRegularizedPPO.load(str(prev), device="cpu",
                                    custom_objects={"policy_class": MaskedActorCriticPolicy})
        model.policy.load_state_dict(old.policy.state_dict())
        demo_files = [DEMO_DIR / f"{name}_r{k}.npz" for k in range(k0)]
        try:
            rows = json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))["rounds"][:k0]
            best = max(rows, key=lambda r: r["laps"]) if rows else None
        except (OSError, ValueError, KeyError):
            rows = []

        def student(obs, m=model):
            act, _ = m.predict(np.asarray(obs, np.float32), deterministic=True)
            return np.clip(act, -1, 1)
        print(f"[DAGGER] resuming at round {k0} from {prev}", flush=True)
    for rnd in range(k0, k0 + a.rounds + 1 if k0 else a.rounds + 1):
        meta = dict(task=spec.TASK_NAME, source="privileged_teacher", profile="measured", dry_run=None,
                    yaw_convention=spec.YAW_CONVENTION, dagger_round=rnd, obs_history=list(lags),
                    asym_critic=True)
        rec = DemoRecorder(meta=meta, obs_dim=width, task=spec.TASK_NAME, obs_shape=(width,))
        ticks, share = collect(student, rnd, env_kw, a.episodes, 50_000 + 1000 * rnd, rec,
                               search_first=not a.no_search_first)
        demo = DEMO_DIR / f"{name}_r{rnd}.npz"
        rec.save(demo)
        demo_files.append(demo)
        d = load_demos([str(f) for f in demo_files])
        mse = fit(model, d["obs"], d["intent"], a.epochs, seed=a.seed + rnd)

        def student(obs, m=model):
            act, _ = m.predict(np.asarray(obs, np.float32), deterministic=True)
            return np.clip(act, -1, 1)
        s = score(lambda env, obs: student(obs), env_kw, a.eval_episodes)
        out = save_clone(model, BC_DIR / f"{name}_r{rnd}", env_kw,
                         {"dagger_round": rnd, "samples": int(len(d["obs"])), "fit_mse": mse,
                          "sim_eval": {"laps_per_90s": s["laps"], "clean": s["clean"], "zones": s["zones"],
                                       "episodes": int(np.ceil(a.eval_episodes / 9) * 9)}})
        rows.append(dict(round=rnd, ticks=ticks, samples=int(len(d["obs"])), teacher_share=share,
                         rim_m=danger_margins(rnd)[0], mse=mse, clone=str(out.relative_to(CMS)), **s))
        if best is None or s["laps"] > best["laps"]:
            best = rows[-1]
        print(f"[DAGGER] round {rnd}: {ticks} ticks (teacher flew {share:.0%}), fit mse {mse:.4f}, "
              f"student laps/90s {s['laps']:.2f} clean {s['clean']:.0%} worst zone {s['min_zone']:.2f} "
              f"best {s['best']}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
        report.write_text(render(rows, teacher, a, best), encoding="utf-8")
        (REPORTS / f"{name}.json").write_text(json.dumps(dict(teacher=teacher, rounds=rows, best=best), indent=2),
                                              encoding="utf-8")
    print(f"[DONE] {report}; best round {best['round']} ({best['laps']:.2f} laps/90s) -> {best['clone']}")
    return rows


def render(rows, teacher, a, best):
    L = [f"# DAgger v2 ({a.tag}): history student taught by the privileged teacher", "",
         f"Goal: {a.lap_dir}-side laps from all 9 start zones, 90 s, measured profile, dr 1, 2° sway. "
         f"{a.episodes} episodes collected per round, scored on {int(np.ceil(a.eval_episodes / 9) * 9)}. "
         "Generated by `hardware/privileged/dagger_hist.py`.", "",
         f"**Teacher** (true state): {teacher['laps']:.2f} laps / 90 s, clean {teacher['clean']:.0%}, "
         f"worst zone {teacher['min_zone']:.2f}.", "",
         "| Round | Samples | Teacher flew | Handover (noodle) | Fit MSE | Student laps / 90 s | Clean | Worst zone | Best | Clone |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['round']} | {r['samples']} | {r['teacher_share']:.0%} | {r['rim_m'] * 100:.0f} cm | "
                 f"{r['mse']:.4f} | **{r['laps']:.2f}** | {r['clean']:.0%} | {r['min_zone']:.2f} | {r['best']} | "
                 f"`{r['clone']}` |")
    L += ["", f"Best so far: round {best['round']}, per zone {best['zones']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
