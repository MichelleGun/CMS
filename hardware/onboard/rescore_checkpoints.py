"""
Re-score a run's saved policies on many more episodes before calling one "best".

best_model.zip is picked on one 36-episode evaluation. That number is noisy: on the same 36 layouts
the best O model scored 1.25 laps/ep evaluated one observation at a time and 1.08 evaluated in a batch
(float rounding alone, amplified over a 90 s flight; TRAINING_SPEEDUPS.md). Picking the maximum of
many noisy evaluations favours a lucky one. This flies every candidate of a run (best, latest, final
and the timed checkpoints/) through the same flight loader as export_policy.py, on --episodes fresh
layouts each, in parallel, and ranks them with a 95 % interval.

    python hardware/onboard/rescore_checkpoints.py results/hoop/runs/w10_n9_O_oblique
    python hardware/onboard/rescore_checkpoints.py <run> [<run> ...] --episodes 100 --workers 4

Writes <run>/rescore.json and prints a table. Then fly or export the winner, e.g.
    python hardware/onboard/fly_hoop.py --dry-run --mode policy --model <winner.zip>
"""
import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CMS = HERE.parents[1]
for p in (HERE, HERE.parent, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEED = 960_000_000           # layouts no training evaluation or export_policy run has used


def candidates(run: Path):
    """[(label, zip)] of a run's saved policies, duplicates (same bytes) dropped."""
    out, seen = [], set()
    files = [("best", run / "best_model.zip"), ("latest", run / "latest_model.zip"),
             ("final", run / "final_model.zip")]
    files += [(p.stem.replace("step_", "ckpt_"), p) for p in sorted((run / "checkpoints").glob("step_*.zip"))]
    for label, f in files:
        if not f.is_file():
            continue
        h = hashlib.sha1(f.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append((label, f))
    return out


def _meta_for(model: Path, run: Path) -> Path:
    """A checkpoint needs its run's meta.json beside it for the flight loader (newer checkpoints
    carry <stem>.meta.json; older ones get one written here)."""
    side = model.with_name(model.stem + ".meta.json")
    if model.parent != run and not side.is_file() and (run / "meta.json").is_file():
        side.write_text((run / "meta.json").read_text())
    return side


def _score_one(args):
    model, episodes, seed = args
    from export_policy import score
    s = score(Path(model), episodes=episodes, seed=seed)
    s.pop("_probe", None)
    return model, s


def interval(mean, std, n):
    half = 1.96 * std / max(1.0, np.sqrt(n))
    return round(mean - half, 3), round(mean + half, 3)


def rescore(runs, episodes=100, workers=4, seed=SEED, log=print):
    jobs = []
    for run in runs:
        for label, f in candidates(run):
            _meta_for(f, run)
            jobs.append((str(f), episodes, seed))
    if not jobs:
        log("no saved policies found")
        return []
    log(f"scoring {len(jobs)} policies x {episodes} episodes on {min(workers, len(jobs))} workers ...")
    results = {}
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        for model, s in ex.map(_score_one, jobs):
            results[model] = s
            log(f"  {Path(model).parent.name}/{Path(model).name}: {s['lpe']:.3f} laps/ep")
    rows = []
    for run in runs:
        mine = []
        for label, f in candidates(run):
            s = results.get(str(f))
            if not s:
                continue
            per_ep = np.asarray(s.get("per_episode") or [s["lpe"]], dtype=float)
            row = dict(run=run.name, label=label, path=str(f), lpe=round(s["lpe"], 3),
                       ci95=interval(s["lpe"], float(per_ep.std(ddof=1)) if len(per_ep) > 1 else 0.0,
                                     len(per_ep)),
                       two_plus=round(s["two_plus"], 3), one_plus=round(s["one_plus"], 3),
                       oblique_entry=s["oblique_entry"], best_streak=s["best_streak"],
                       episodes=s["episodes"], zones=s["zones"])
            mine.append(row)
        mine.sort(key=lambda r: (r["lpe"], r["two_plus"], r["oblique_entry"]), reverse=True)
        try:
            (run / "rescore.json").write_text(json.dumps(dict(episodes=episodes, seed=seed, ranking=mine),
                                                         indent=2))
        except OSError as e:
            log(f"could not write {run / 'rescore.json'}: {e}")
        rows += mine
    rows.sort(key=lambda r: (r["lpe"], r["two_plus"], r["oblique_entry"]), reverse=True)
    return rows


def table(rows):
    L = ["| # | Run | Policy | Laps/ep | ~95 % range | 2+ passes | Oblique entry |",
         "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['run']} | {r['label']} | {r['lpe']:.3f} | {r['ci95'][0]:.2f}-{r['ci95'][1]:.2f} "
                 f"| {r['two_plus']:.0%} | {r['oblique_entry']:.0%} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4, help="processes (~0.5 GB each)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    runs = [r if r.is_absolute() else (Path.cwd() / r) for r in a.runs]
    for r in runs:
        if not r.is_dir():
            sys.exit(f"[ERROR] not a run folder: {r}")
    rows = rescore(runs, a.episodes, a.workers, a.seed)
    if rows:
        print()
        print(table(rows))
        print(f"\nWinner: {rows[0]['path']}")
        print("If the ~95 % ranges of the top candidates overlap, the difference is noise: pick on oblique "
              "entry (the corner approach the real drone struggles with), or re-run with more --episodes.")


if __name__ == "__main__":
    main()
