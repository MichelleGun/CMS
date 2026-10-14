"""
Autopilot for the night of 2026-10-13: waits for the Week 10 A-F batch to finish, then runs the
next steps one at a time and writes a single morning report.

    python hardware/privileged/autopilot.py --dry-run     # what it sees and would do
    python hardware/privileged/autopilot.py               # run (started detached; keeps the PC awake)

1. Wait until results/hoop/runs/w10_front_summary.txt exists (the A-F launcher writes it last).
   If any A-F run did not complete (a crash), run the launcher once more: it archives and reruns
   only those.
2. A-F verdict: winner = best mean of the last 5 evaluations (passes per front-runway episode).
3. The goal stage: the winner on same-side laps from all 9 zones (week10_experiments --stage laps).
4. DAgger with the privileged teacher (privileged/dagger.py), 8 rounds.
5. hardware/privileged/reports/morning.md: what happened, the numbers, what to look at first.

Every step is logged to results/hoop/autopilot.log; a step that fails is logged and the next one
still runs.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HW = HERE.parent
CMS = HW.parent
RUNS = CMS / "results" / "hoop" / "runs"
LOG = CMS / "results" / "hoop" / "autopilot.log"
SUMMARY = RUNS / "w10_front_summary.txt"
REPORT = HERE / "reports" / "morning.md"
PY = sys.executable
KEYS = "ABCDEF"
NAMES = {"A": "baseline", "B": "gamma", "C": "kl", "D": "entropy", "E": "sde", "F": "lambda1"}


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def front_dir(k):
    return RUNS / f"w10_front_{k}_{NAMES[k]}"


def status(d):
    try:
        return json.loads((d / "meta.json").read_text()).get("final_status")
    except (OSError, ValueError):
        return None


def last5(d):
    try:
        m = [p["mean"] for p in json.loads((d / "passes.json").read_text())]
        return float(np.mean(m[-5:])) if m else float("nan")
    except (OSError, ValueError):
        return float("nan")


def best5(d):
    try:
        m = np.array([p["mean"] for p in json.loads((d / "passes.json").read_text())])
        return float(np.convolve(m, np.ones(5) / 5, mode="valid").max()) if len(m) >= 5 else float(m.max())
    except (OSError, ValueError):
        return float("nan")


def step(name, cmd):
    log(f"START {name}: {' '.join(cmd)}")
    t0 = time.time()
    with open(CMS / "results" / "hoop" / f"autopilot_{name}.log", "w", encoding="utf-8") as out:
        rc = subprocess.call(cmd, cwd=CMS, stdout=out, stderr=subprocess.STDOUT)
    log(f"END {name}: exit {rc} after {(time.time() - t0) / 60:.0f} min")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dagger-rounds", type=int, default=8)
    ap.add_argument("--dagger-episodes", type=int, default=24)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(HW / "onboard"))
    from week10_experiments import _keep_awake

    state = {k: (status(front_dir(k)), last5(front_dir(k))) for k in KEYS}
    if a.dry_run:
        print("A-F now:", state)
        print("summary file present:", SUMMARY.exists())
        print("would: wait for summary -> rerun crashed (if any) -> laps stage of the winner -> "
              f"DAgger {a.dagger_rounds} rounds x {a.dagger_episodes} episodes -> {REPORT}")
        return
    _keep_awake(True)
    try:
        log("autopilot started; waiting for the A-F batch")
        while not SUMMARY.exists():
            time.sleep(60)
        log("A-F summary found")
        if any(status(front_dir(k)) != "completed" for k in KEYS):
            log("some A-F runs did not complete: running the launcher once more for those")
            step("front_rerun", [PY, "hardware/onboard/week10_experiments.py", "--stage", "front"])

        scores = {k: last5(front_dir(k)) for k in KEYS if status(front_dir(k)) == "completed"}
        winner = max(scores, key=lambda k: np.nan_to_num(scores[k], nan=-1)) if scores else None
        log(f"A-F last-5 means: {scores}; winner {winner}")

        laps_dir = None
        if winner:
            parent = front_dir(winner) / "best_model.zip"
            rc = step("laps_stage", [PY, "hardware/onboard/week10_experiments.py", "--stage", "laps",
                                     "--only", winner, "--parent", str(parent.relative_to(CMS))])
            laps_dir = RUNS / f"w10_laps_{winner}_{NAMES[winner]}"
            log(f"laps stage {laps_dir.name}: {status(laps_dir)}, last-5 mean passes {last5(laps_dir):.2f}")

        step("dagger", [PY, "hardware/privileged/dagger.py", "--rounds", str(a.dagger_rounds),
                        "--episodes", str(a.dagger_episodes)])
        write_report(scores, winner, laps_dir)
        log(f"done; report {REPORT}")
    except Exception as e:                     # log, never leave the PC awake for nothing
        log(f"autopilot error: {e!r}")
        raise
    finally:
        _keep_awake(False)


def write_report(scores, winner, laps_dir):
    L = [f"# Morning report ({datetime.now():%Y-%m-%d %H:%M})", "",
         "Goal: enter the hoop, loop round the outside, re-enter, as many times as possible.", "",
         "Everything below ran on the corrected simulator (crossing fix, measured profile, no speed input, "
         "clockwise yaw). Written by `hardware/privileged/autopilot.py`; log in `results/hoop/autopilot.log`.", "",
         "## 1. Week 10 PPO settings (front runway, one pass per episode, 3M steps each)", "",
         "| Run | Status | Best (5-eval avg) | Last 5 evals |", "|---|---|---|---|"]
    for k in KEYS:
        d = front_dir(k)
        L.append(f"| {k} {NAMES[k]} | {status(d)} | {best5(d):.3f} | {last5(d):.3f}{' **(winner)**' if k == winner else ''} |")
    L += ["", "Passes per front-runway episode (max 1). Curves and diagnostics: `tutorial/week10_compare.ipynb`.", ""]
    L += ["## 2. The goal: the winner on same-side laps (3M steps)", ""]
    if laps_dir is not None:
        L.append(f"`{laps_dir.name}`: {status(laps_dir)}; laps per 90 s episode, last 5 evaluations: "
                 f"**{last5(laps_dir):.2f}**, best: {best5(laps_dir):.2f}. Notebook: set `PATTERN = \"w10_laps_*\"`.")
    else:
        L.append("Not run (no completed A-F run).")
    L += ["", "## 3. DAgger: the camera-only student taught by the privileged teacher", ""]
    dag = HERE / "reports" / "dagger.md"
    if dag.exists():
        body = dag.read_text(encoding="utf-8").split("\n")
        L += [line for line in body if line.startswith("|") or line.startswith("**Teacher")]
    else:
        L.append("DAgger did not produce a report (see results/hoop/autopilot_dagger.log).")
    L += ["", "## What to look at first", "",
          "1. Section 2: laps per 90 s of the RL policy. The teacher manages ~4.8 (it sees the true state).",
          "2. Section 3: does the student's laps column climb round by round? If it stays near 0, the camera "
          "inputs (no speed, ring often out of view on the way round) may not carry what the teacher uses; "
          "that is the question to look at next (more memory in the observation, or a privileged critic).",
          "3. The best policy is the candidate for the next real flight, after the 1-minute hand check of the "
          "yaw sign on the real drone.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
