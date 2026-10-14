"""
Score policies the way the REAL drone runs them: loaded by onboard/policy.load_policy (which stacks
the history and zero-pads the critic-only inputs itself) and fed one plain frame per tick from the
same ObsBuilder fly_hoop uses. So a mistake in the deploy path shows up here, not in the cage.

    python hardware/privileged/eval_deploy.py results/hoop/runs/n2_aac_ppo_W hardware/flights/bc/dagger_n1_r8
    python hardware/privileged/eval_deploy.py --glob "results/hoop/runs/n*_*" --episodes 27 --write-labels

Per policy: same-side (or the policy's own lap_dir) laps per 90 s from all 9 start zones, clean
episodes, the worst zone, outcomes. Also a front-runway single-pass check and the "harsh" profile.
--write-labels writes label.json (the dashboard's name for the run) with the score, and stars the
best one as RECOMMENDED. Report: hardware/privileged/reports/deploy_eval.md (+ .json).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent
CMS = HW.parent
for p in (HW, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from onboard import sim as onboard_sim  # noqa: E402
from onboard import spec  # noqa: E402
from onboard.gym_env import OnboardHoopEnv  # noqa: E402
from onboard.policy import load_policy  # noqa: E402

REPORTS = HERE / "reports"
MODEL_ORDER = ("best_model.zip", "bc_model.zip", "latest_model.zip", "final_model.zip")


def model_file(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for n in MODEL_ORDER:
        if (path / n).is_file():
            return path / n
    raise FileNotFoundError(f"no model in {path}")


def run(fn, env_kw, profile, episodes, seed):
    env = OnboardHoopEnv(seed=seed, profile=profile, **env_kw)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(fn, "reset"):
            fn.reset()
        zone = env.sim.start_type
        while True:
            obs, _, term, trunc, info = env.step(fn(obs))
            if term or trunc:
                rows.append(dict(zone=zone, passes=int(info.get("passes", 0)),
                                 outcome=info["phase"] if term else "timeout"))
                break
    return rows


def summarise(rows):
    passes = [r["passes"] for r in rows]
    zones = {}
    for r in rows:
        zones.setdefault(r["zone"], []).append(r["passes"])
    clean = [r["passes"] >= 2 and r["outcome"] not in ("struck", "out_of_cage") for r in rows]
    return dict(laps=float(np.mean(passes)), clean=float(np.mean(clean)), best=int(max(passes)), episodes=len(rows),
                min_zone=float(min(np.mean(v) for v in zones.values())),
                zones={k: round(float(np.mean(v)), 2) for k, v in sorted(zones.items())},
                outcomes={k: sum(r["outcome"] == k for r in rows) for k in sorted({r["outcome"] for r in rows})})


def evaluate(path, episodes=27, seed=7000, front=18, harsh=True):
    mf = model_file(path)
    fn = load_policy(mf)
    meta = getattr(fn, "meta", {}) or {}
    tracker = getattr(fn, "obs_width", spec.OBS_DIM) == spec.TRACKER_OBS_DIM
    lap_dir = meta.get("lap_dir") or "same"
    base = dict(sway_deg=2.0, wash_deg=15.0, dr=1.0, ring_tracker=tracker)
    episodes = int(np.ceil(episodes / 9) * 9)
    laps_kw = dict(base, laps=0, lap_dir=lap_dir, start_mode="zones")
    res = {"path": str(Path(path)), "model": str(mf), "lap_dir": lap_dir,
           "history": meta.get("obs_history"), "asym": bool(meta.get("asym_critic")),
           "laps": summarise(run(fn, laps_kw, onboard_sim.PROFILES["measured"], episodes, seed))}
    if front:
        f = run(fn, dict(base, laps=1, start_mode="ring"), onboard_sim.PROFILES["measured"], front, 8000)
        res["front_pass"] = float(np.mean([r["outcome"] == "passed" for r in f]))
    if harsh:
        h = run(fn, laps_kw, onboard_sim.PROFILES["harsh"], 9, 9000)
        res["harsh_laps"] = float(np.mean([r["passes"] for r in h]))
    return res


def label_text(r):
    kind = "laps" if r["lap_dir"] == "same" else "back-and-forth"
    return f"{r['laps']['laps']:.1f} {kind}/90s, worst zone {r['laps']['min_zone']:.1f}"[:60]


def write_label(r, recommended=False):
    d = Path(r["path"])
    if d.is_file():
        d = d.parent
    text = ("★ RECOMMENDED · " if recommended else "") + label_text(r)
    (d / "label.json").write_text(json.dumps({"label": text[:60], "starred": bool(recommended)}),
                                  encoding="utf-8")
    ev = d / "evals"
    ev.mkdir(exist_ok=True)
    (ev / "deploy_eval.json").write_text(json.dumps({
        "kind": "laps", "lap_dir": r["lap_dir"], "laps_per_90s": r["laps"]["laps"],
        "clean": r["laps"]["clean"], "min_zone": r["laps"]["min_zone"], "zones": r["laps"]["zones"],
        "front_pass": r.get("front_pass"), "harsh_laps": r.get("harsh_laps"),
        "episodes": r["laps"]["episodes"], "recommended": bool(recommended)}, indent=2),
        encoding="utf-8")
    return d


def render(results):
    L = ["# Deploy-path evaluation", "",
         "Each policy loaded by `onboard/policy.load_policy` exactly as `fly_hoop.py --mode policy` loads it, "
         "fed plain frames (history and zero padding done by the loader). Measured profile, dr 1, 2° sway, 15° "
         "prop wash, 90 s episodes from all 9 start zones. Generated by `hardware/privileged/eval_deploy.py`.", "",
         "| Policy | Laps | Laps / 90 s | Clean (≥2, no crash) | Worst zone | Best | Front-runway pass | Harsh profile laps | Outcomes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["laps"]["laps"]):
        lp = r["laps"]
        L.append(f"| `{r['path']}` | {r['lap_dir']} | **{lp['laps']:.2f}** | {lp['clean']:.0%} | {lp['min_zone']:.2f} | "
                 f"{lp['best']} | {r.get('front_pass', float('nan')):.0%} | {r.get('harsh_laps', float('nan')):.2f} | "
                 f"{lp['outcomes']} |")
    L += ["", "Per-zone laps / 90 s (zone 1-8 = the outer cells of a 3x3 grid, corners 1, 3, 5, 7; 9 = under the hoop):", ""]
    for r in sorted(results, key=lambda r: -r["laps"]["laps"]):
        L.append(f"- `{r['path']}`: {r['laps']['zones']}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--episodes", type=int, default=27)
    ap.add_argument("--front", type=int, default=18)
    ap.add_argument("--no-harsh", action="store_true")
    ap.add_argument("--write-labels", action="store_true")
    ap.add_argument("--out", default="deploy_eval")
    a = ap.parse_args(argv)
    paths = list(a.paths) + ([Path(p) for p in sorted(glob.glob(a.glob))] if a.glob else [])
    results = []
    for p in paths:
        try:
            r = evaluate(p, a.episodes, front=a.front, harsh=not a.no_harsh)
        except Exception as e:                      # one bad folder must not stop the ranking
            print(f"[SKIP] {p}: {e}", flush=True)
            continue
        results.append(r)
        print(f"[EVAL] {p}: {r['laps']['laps']:.2f} laps/90s clean {r['laps']['clean']:.0%} worst zone "
              f"{r['laps']['min_zone']:.2f} front {r.get('front_pass', 0):.0%} harsh {r.get('harsh_laps', 0):.2f}",
              flush=True)
    if not results:
        return []
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{a.out}.md").write_text(render(results), encoding="utf-8")
    (REPORTS / f"{a.out}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if a.write_labels:
        same = [r for r in results if r["lap_dir"] == "same"]
        best = max(same or results, key=lambda r: (r["laps"]["min_zone"] >= 1, r["laps"]["laps"]))
        for r in results:
            write_label(r, recommended=r is best)
        print(f"[BEST] {best['path']}")
    print(f"[DONE] {REPORTS / (a.out + '.md')}")
    return results


if __name__ == "__main__":
    main()
