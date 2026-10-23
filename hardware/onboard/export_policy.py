"""
Score trained models exactly the way the Tello will run them, and copy them to hardware/policies/
under names that say what they are.

    python hardware/onboard/export_policy.py results/hoop/runs/w10_laps3_X_trk_combo [more runs ...]
    python hardware/onboard/export_policy.py --all-laps3        # every w10_laps3_* run (and _ext)

Training output = Tello input, checked for every file:
  * it is loaded with policy.load_policy, the same loader fly_hoop.py --mode policy uses (yaw
    convention conversion, legacy / ring-tracker widths, action clip to [-1, 1]);
  * it is flown in the sim through that loader's policy_fn on observations built by obs.ObsBuilder
    at fn.obs_width (the class fly_hoop builds its observations with), on the laps task: endless
    same-side laps from all 9 start zones, the run's own profile / randomisation;
  * the exported copy gets <name>.meta.json beside it (policy.load_policy reads that), and is
    re-loaded from hardware/policies/ and must give the same actions as the original.

Each run gives 2 candidates, best_model.zip (best evaluation reward) and final_model.zip (or
latest_model.zip). Files: hardware/policies/LOOP_<date>_<run>_<best|final>_<laps>lpe.zip, ranked in
hardware/policies/README.md with the command to fly each one.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CMS = HERE.parents[1]
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

RUNS = CMS / "results" / "hoop" / "runs"
POLICIES = CMS / "hardware" / "policies"
SEED = 950_000_000


def _short(run: Path) -> str:
    n = run.name
    for p in ("w10_laps3_", "w10_laps_", "w10_front_", "onboard_"):
        if n.startswith(p):
            n = n[len(p):]
            break
    return n.replace("_", "-")


def candidates(run: Path):
    out = []
    if (run / "best_model.zip").is_file():
        out.append(("best", run / "best_model.zip"))
    for name in ("final_model.zip", "latest_model.zip"):
        if (run / name).is_file():
            out.append(("final", run / name))
            break
    return out


def env_kwargs(meta: dict, tracker: bool) -> dict:
    rb = meta.get("robust") or {}
    return dict(profile=meta.get("profile") or "measured", laps=0, lap_dir="same", start_mode="zones",
                dr=float(rb.get("dr", 1.0)), sway_deg=float(rb.get("sway_deg", 5.0)),
                wash_deg=float(rb.get("wash_deg", 15.0)), ring_tracker=tracker)


def score(model_path: Path, episodes=36, seed=SEED, meta_override=None, oblique_episodes=30):
    """Fly a saved model through the flight loader on the laps task. -> dict of scores."""
    from onboard import spec
    from onboard.gym_env import OnboardHoopEnv
    from onboard.policy import load_policy
    fn = load_policy(model_path)
    tracker = fn.obs_width == spec.TRACKER_OBS_DIM
    meta = meta_override if meta_override is not None else fn.meta
    env = OnboardHoopEnv(seed=seed, **env_kwargs(meta, tracker))
    if env.observation_space.shape[0] != fn.obs_width:
        raise RuntimeError(f"{model_path}: the policy reads {fn.obs_width} inputs but the flight "
                           f"builder makes {env.observation_space.shape[0]}")
    env.set_episode_seed(seed)
    passes, acts, zones = [], [], {}
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            if hasattr(fn, "reset"):        # a memory policy (onboard/history.py) clears its frame stack
                fn.reset()
            first = not passes
            done, info = False, {}
            while not done:
                a = fn(obs)
                if not (a.shape == (spec.ACT_DIM,) and np.all(np.abs(a) <= 1.0)):
                    raise RuntimeError(f"{model_path}: action {a} is not 4 values in [-1, 1]")
                if first and len(acts) < 200:     # one episode, in order: a memory policy replays it
                    acts.append((obs.copy(), a.copy()))
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
            passes.append(int(info.get("passes", 0)))
            zones.setdefault(str(info.get("start_type", "?")).replace("zone_", ""), []).append(passes[-1])
    finally:
        env.close()
    p = np.array(passes)
    oblique = oblique_entry(fn, tracker, meta, episodes=oblique_episodes, seed=seed + 1)
    return dict(zones={k: round(float(np.mean(v)), 2) for k, v in sorted(zones.items())},
                oblique_entry=oblique, lpe=float(p.mean()), best_streak=int(p.max()), two_plus=float((p >= 2).mean()),
                one_plus=float((p >= 1).mean()), episodes=episodes, obs_width=int(fn.obs_width),
                per_episode=[int(v) for v in p],
                yaw_converted=bool(fn.yaw_converted), algo=(fn.meta or {}).get("algo", "ppo"),
                _probe=acts)


def oblique_entry(fn, tracker, meta, episodes=30, seed=SEED + 1):
    """Share of slanted starts (sim.place_oblique: 0.5-1.1 m from the ring, 35-80 deg off its axis,
    ring in view) that end in a pass, not a strike: the corner approach that clipped the hoop on
    the real drone (2026-10-14). Each episode ends at the first pass or failure."""
    from onboard.gym_env import OnboardHoopEnv
    env = OnboardHoopEnv(seed=seed, oblique_p=1.0, **dict(env_kwargs(meta, tracker), laps=1))
    env.set_episode_seed(seed)
    ok = 0
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            if hasattr(fn, "reset"):
                fn.reset()
            done, info = False, {}
            while not done:
                obs, _, term, trunc, info = env.step(fn(obs))
                done = term or trunc
            ok += info.get("phase") == "passed"
    finally:
        env.close()
    return round(ok / episodes, 3)


def _same_actions(path: Path, probe) -> bool:
    from onboard.policy import load_policy
    fn = load_policy(path)
    if hasattr(fn, "reset"):
        fn.reset()
    return all(np.allclose(fn(o), a, atol=1e-5) for o, a in probe)


def export(runs, episodes=36, log=print, date=None):
    """Score every candidate of `runs`, copy each to hardware/policies/, write the README ranking.
    Returns the ranking as Markdown."""
    date = date or datetime.now().strftime("%Y%m%d")
    POLICIES.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        run = Path(run)
        for kind, src in candidates(run):
            try:
                s = score(src, episodes)
            except Exception as e:
                log(f"[EXPORT] {run.name}/{src.name}: not exportable: {e!r}")
                continue
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
            name = f"LOOP_{date}_{_short(run)}_{kind}_{s['lpe']:.2f}lpe"
            dst = POLICIES / f"{name}.zip"
            shutil.copy2(src, dst)
            info = dict(meta, exported_from=str(src.relative_to(CMS)), exported_at=datetime.now().isoformat(
                timespec="seconds"), export_scores={k: v for k, v in s.items() if not k.startswith("_")})
            (POLICIES / f"{name}.meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            if not _same_actions(dst, s["_probe"]):
                log(f"[EXPORT] {dst.name}: the copy does NOT act like the original; removed")
                dst.unlink()
                (POLICIES / f"{name}.meta.json").unlink()
                continue
            steps = meta.get("final_timesteps") or meta.get("timesteps_requested")
            rows.append(dict(name=name, run=run.name, kind=kind, steps=steps, **s))
            log(f"[EXPORT] {dst.name}: {s['lpe']:.2f} laps/episode, best {s['best_streak']}, "
                f"2+ passes in {s['two_plus']:.0%} of episodes")
    md = write_ranking()
    return md


def all_exported():
    """Every policy in hardware/policies/ with its export scores (from its .meta.json), so the
    ranking covers all exports, not only the runs of the latest call."""
    rows = []
    for m in sorted(POLICIES.glob("LOOP_*.meta.json")):
        try:
            meta = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s = meta.get("export_scores")
        name = m.name[:-len(".meta.json")]
        if not s or not (POLICIES / f"{name}.zip").exists():
            continue
        src = meta.get("exported_from", "")
        rows.append(dict(name=name, run=Path(src).parent.name, kind="best" if "best" in Path(src).name else "final",
                         steps=meta.get("final_timesteps") or meta.get("timesteps_requested"),
                         memory=bool(meta.get("obs_history") or meta.get("asym_critic")), **s))
    rows.sort(key=lambda r: (r["episodes"] >= 36, r["lpe"], r["two_plus"]), reverse=True)
    return rows


def write_ranking():
    rows = all_exported()
    md = ranking_md(rows)
    (POLICIES / "README.md").write_text(
        "# Exported policies (Tello-ready)\n\nWritten by `hardware/onboard/export_policy.py` (every export, "
        "ranked together; re-run it to refresh).\n\n" + md + "\n", encoding="utf-8")
    if rows:
        (POLICIES / "FLY_THIS_FIRST.txt").write_text(
            f"{rows[0]['name']}.zip\n\npython hardware/onboard/fly_hoop.py --mode policy --model "
            f"hardware/policies/{rows[0]['name']}.zip\n", encoding="utf-8")
    return md


def ranking_md(rows):
    L = ["## Ranking (laps task in the sim: same-side laps from all 9 zones, 90 s, measured profile, "
         "full randomisation, fixed layouts)", "",
         "lpe = laps (passes) per 90 s episode. Fly the top row first; the file name is also written into "
         "each flight's meta.json (`model`), so a flight log always says which policy flew. Scores from fewer "
         "than 36 episodes are listed last. Memory = the policy keeps a history of past frames "
         "(onboard/history.py); it flies as it is, `policy.load_policy` keeps the same history on the drone. "
         "Oblique entry = share of 30 slanted starts near the ring (35-80 deg off its axis, as from a corner "
         "of the real cage) that end in a pass rather than a strike; per-zone scores are in each "
         "file's .meta.json (export_scores.zones).", "",
         "| # | File | From | Steps | Laps/ep | Best in 1 ep | 2+ passes | 1+ pass | Oblique entry | Memory | Episodes |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | `{r['name']}.zip` | {r['run']} ({r['kind']}) | {r['steps'] or '?'} | {r['lpe']:.2f} | "
                 f"{r['best_streak']} | {r['two_plus']:.0%} | {r['one_plus']:.0%} | "
                 f"{(format(r['oblique_entry'], '.0%') if r.get('oblique_entry') is not None else '-')} | "
                 f"{'yes' if r['memory'] else 'no'} | {r['episodes']} |")
    if rows:
        L += ["", "Fly the top one:", "", "```bash",
              f"python hardware/onboard/fly_hoop.py --mode policy --model hardware/policies/{rows[0]['name']}.zip",
              "```"]
    else:
        L += ["", "Nothing exported yet."]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--all-laps3", action="store_true")
    ap.add_argument("--episodes", type=int, default=36)
    a = ap.parse_args(argv)
    runs = list(a.runs) + (sorted(RUNS.glob("w10_laps3_*")) if a.all_laps3 else [])
    if not runs:
        ap.error("give run folders or --all-laps3")
    print(export([r if r.is_absolute() else CMS / r for r in runs], a.episodes))


if __name__ == "__main__":
    main()
