"""
The hierarchical-RL pipeline, unattended: wait for the two skills (week10_experiments SE, SL; started
with loop_night.py --tag hrl), train the manager (train_hrl.py), then compare on the optimiser's referee.

    python hardware/onboard/launch_detached.py results/hoop/hrl_pipeline -- hardware/onboard/hrl_pipeline.py

Compared on the same fixed layouts (export_policy.score: the flight loader, 72 laps episodes + 60
slanted starts; score = laps/ep + 1.0 x oblique entry, as loop_forever):
  * the flat policy C9 (the skills' parent),
  * the fixed switch (enter skill until the first pass, then the loop skill),
  * the learned manager over enter, loop and C9.
Writes results/hoop/hrl_compare.md and .json.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMS = HERE.parents[1]
RUNS = CMS / "results" / "hoop" / "runs"
SE, SL = RUNS / "w10_hrl_SE_skill_enter", RUNS / "w10_hrl_SL_skill_loop"
C9 = RUNS / "w10_mg6_C9_loop_practice"
TAG = "v1"
OUT = CMS / "results" / "hoop" / "hrl_compare"


def status(run):
    try:
        return json.loads((run / "meta.json").read_text(encoding="utf-8")).get("final_status")
    except (OSError, ValueError):
        return None


def say(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(CMS / "results" / "hoop" / "hrl_pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    say("waiting for the skills SE and SL")
    while not all(status(r) in ("completed", "stopped") for r in (SE, SL)):
        time.sleep(300)
    say(f"skills done: SE {status(SE)}, SL {status(SL)}; training the manager")
    rel = lambda p: p.relative_to(CMS).as_posix()                  # noqa: E731
    run = RUNS / f"hrl_{TAG}"
    if status(run) not in ("completed", "stopped"):
        rc = subprocess.run([sys.executable, str(HERE / "train_hrl.py"), "--tag", TAG,
                             "--skills", rel(SE), rel(SL), rel(C9), "--n-envs", "4"], cwd=CMS).returncode
        say(f"manager training ended (exit {rc}, {status(run)})")
    sys.path.insert(0, str(HERE.parent))
    from onboard.export_policy import score
    rows = []
    for name, path in (("flat C9 (final)", C9 / "final_model.zip"),
                       ("fixed switch (enter -> loop)", run / "phase_bundle.json"),
                       ("learned manager (best)", run / "best_model.zip"),
                       ("learned manager (final)", run / "final_model.zip")):
        if not path.is_file():
            continue
        s = score(path, episodes=72, oblique_episodes=60)
        s = {k: v for k, v in s.items() if not k.startswith("_")}
        rows.append(dict(name=name, path=rel(path), score=s["lpe"] + s["oblique_entry"], **s))
        say(f"  {name}: laps/ep {s['lpe']:.2f}, 2+ passes {s['two_plus']:.0%}, oblique {s['oblique_entry']:.0%}")
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    L = ["# Hierarchical RL vs the flat policy", "",
         f"{datetime.now():%Y-%m-%d %H:%M}. Same fixed layouts: 72 laps episodes + 60 slanted starts, through "
         "the flight loader. Score = laps/ep + 1.0 x oblique entry (as loop_forever).", "",
         "| Policy | Laps / 90 s | 2+ passes | Best in 1 ep | Oblique entry | Score |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['lpe']:.2f} | {r['two_plus']:.0%} | {r['best_streak']} | "
                 f"{r['oblique_entry']:.0%} | {r['score']:.2f} |")
    try:
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        hist = json.loads((run / "passes.json").read_text(encoding="utf-8"))
        L += ["", f"Skill usage of the best manager evaluation (rows: before / after the first pass; columns: "
                  f"enter, loop, C9): {max(hist, key=lambda h: h['mean'])['usage_before_after_first_pass']}",
              f"Fixed-switch baseline during training: {meta.get('phase_baseline')}"]
    except (OSError, ValueError, KeyError):
        pass
    OUT.with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    say(f"compare written: {OUT.with_suffix('.md')}")


if __name__ == "__main__":
    main()
