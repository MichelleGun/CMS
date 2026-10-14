"""
Week 10 round 2 as a race: try PPO setting combinations, stop a run as soon as it is clearly behind
the best curve we have (run C), give its CPU slot to the next one, and give the extra training time
to whichever run is still ahead at the end (successive halving, with C's curve as the yardstick).

    python hardware/onboard/w10_race.py --dry-run      # queue, rule, and the rule replayed on A-F
    python hardware/onboard/w10_race.py                # run (start detached; keeps the PC awake)
    python hardware/onboard/w10_race.py --no-continue  # race only, no extra 3M steps / laps stage

The queue (front runway stage, same parent / demos / budget as A-F):
  G  kl_gamma    C + gamma 0.995, lambda 0.97   (B's change on top of C)
  H  kl_lambda1  C + lambda 1.0                 (F's change on top of C)
  C  seed 1      C again with another seed: how much C varies by chance. Never stopped.

The stop rule (checked every 5 min, at checkpoints every CHECK_EVERY evaluations = 400k steps):
  mean of the run's last 10 evaluations vs C's last 10 at the same point of training;
  stop if more than STOP_GAP (0.10) below, or more than WARN_GAP (0.05) below at two checkpoints in a
  row. Nothing is judged before the first checkpoint. Each evaluation is 30 episodes, so a 10-eval
  mean is good to about +-0.03: 0.10 is well outside noise.
A stopped run gets stop.flag: it ends cleanly (final_model.zip, meta final_status "stopped"), its
best_model.zip is kept, and the next queued run starts in its slot.

When the queue is done, the winner = highest last-10 mean among C and the runs that finished
(not stopped). It is continued for another 3M steps (w10_front_<run>_long, steps carry on), then
trained on the laps stage (w10_laps_<run>_long). Log results/hoop/w10_race.log; table
results/hoop/runs/w10_race_summary.md.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from week10_experiments import (ARCHIVE, CMS, DEFAULT_PARENT, EXPERIMENTS, RUNS, STAGES,  # noqa: E402
                                _keep_awake, _status, command, run_dir)

REF = run_dir("front", "C")                          # the yardstick: w10_front_C_kl
QUEUE = [("G", 0), ("H", 0), ("C", 1)]
NEVER_STOP = {("C", 1)}                              # the noise check must run to the end
EVAL_STEPS = 20_000                                  # one evaluation every 20k steps (A-F)
CHECK_EVERY = 20                                     # evaluations between checkpoints (400k steps)
WINDOW = 10
STOP_GAP, WARN_GAP = 0.10, 0.05
LOG = CMS / "results" / "hoop" / "w10_race.log"
SUMMARY = RUNS / "w10_race_summary.md"


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def curve(d: Path):
    """Mean passes per evaluation, in order (index i = after (i+1)*20k steps of this run)."""
    try:
        return [float(p["mean"]) for p in json.loads((d / "passes.json").read_text())]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def judge(run, ref, upto=None):
    """Replay the stop rule over a run's curve against the reference curve.

    Returns (stop_at_eval or None, reason, checkpoints) where checkpoints is a list of
    (n_evals, run_mean, ref_mean, gap). Pure function: unit-tested on the A-F curves."""
    n_max = min(len(run), len(ref)) if upto is None else min(upto, len(run), len(ref))
    checks, warned = [], False
    for n in range(CHECK_EVERY, n_max + 1, CHECK_EVERY):
        r, c = float(np.mean(run[n - WINDOW:n])), float(np.mean(ref[n - WINDOW:n]))
        gap = r - c
        checks.append((n, r, c, gap))
        if gap < -STOP_GAP:
            return n, f"{-gap:.2f} below C at {n * EVAL_STEPS // 1000}k steps", checks
        if gap < -WARN_GAP:
            if warned:
                return n, f"more than {WARN_GAP:.2f} below C at two checkpoints in a row " \
                          f"(now {-gap:.2f}, {n * EVAL_STEPS // 1000}k steps)", checks
            warned = True
        else:
            warned = False
    return None, "", checks


def last10(d):
    c = curve(d)
    return float(np.mean(c[-WINDOW:])) if c else float("nan")


def write_summary(rows, note=""):
    ref = curve(REF)
    L = [f"# Week 10 race ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"Yardstick: `{REF.name}` (C). A run is stopped when its last-{WINDOW}-evaluation mean falls "
         f"more than {STOP_GAP} below C's at the same point (or more than {WARN_GAP} below twice in a row), "
         f"checked every {CHECK_EVERY * EVAL_STEPS // 1000}k steps. Written by `hardware/onboard/w10_race.py`.", "",
         "| Run | Settings | Steps | Last-10 mean | C at same point | Gap | State |", "|---|---|---|---|---|---|---|"]
    for key, seed, d, state in rows:
        c = curve(d)
        n = len(c)
        r = float(np.mean(c[-WINDOW:])) if n else float("nan")
        cr = float(np.mean(ref[max(0, min(n, len(ref)) - WINDOW):min(n, len(ref))])) if n and ref else float("nan")
        name, flags = EXPERIMENTS[key]
        L.append(f"| {d.name} | {' '.join(flags) or 'baseline'} | {n * EVAL_STEPS // 1000}k | {r:.3f} | "
                 f"{cr:.3f} | {r - cr:+.3f} | {state} |")
    if note:
        L += ["", note]
    SUMMARY.write_text("\n".join(L) + "\n", encoding="utf-8")


def start(cmd, out):
    if out.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        dest = ARCHIVE / f"{out.name}_{datetime.now():%Y%m%d-%H%M%S}"
        shutil.move(str(out), str(dest))
        log(f"[RERUN] {out.name} was '{_status(dest)}'; old folder kept in {dest}")
    out.mkdir(parents=True, exist_ok=True)
    f = open(out / "train.log", "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=CMS, stdout=f, stderr=subprocess.STDOUT,
                            creationflags=DETACH_FLAGS), f


# Windows: each run gets its own hidden console in a new process group. A Ctrl+C that reached the
# launcher's console stopped every run of round 5 at once (2026-10-14 12:23, "^C" in its log); a
# new process group ignores Ctrl+C, and stop.flag / CTRL_BREAK still stop a run cleanly.
DETACH_FLAGS = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" else 0)


def retarget(cmd, out: Path):
    """Same training command, written to another run folder."""
    cmd = list(cmd)
    cmd[cmd.index("--run-id") + 1] = out.name
    cmd[cmd.index("--out") + 1] = str(out)
    return cmd


def race(a):
    queue = list(QUEUE)
    running = []                       # (key, seed, out, proc, logfile)
    rows = {}                          # (key, seed) -> [key, seed, out, state]
    stopped = set()
    last_check = 0.0
    while queue or running:
        while queue and len(running) < a.parallel:
            key, seed = queue.pop(0)
            cmd, out = command("front", key, DEFAULT_PARENT, a.timesteps, a.n_envs, seed, a.no_clearml)
            rows[(key, seed)] = [key, seed, out, "running"]
            if _status(out) in ("completed", "stopped"):
                rows[(key, seed)][3] = _status(out)
                log(f"[SKIP] {out.name} already {_status(out)}")
                continue
            proc, f = start(cmd, out)
            running.append((key, seed, out, proc, f))
            log(f"[START] {out.name}")
        if time.time() - last_check >= a.check_min * 60:
            last_check = time.time()
            ref = curve(REF)
            for key, seed, out, proc, f in running:
                if (key, seed) in NEVER_STOP or out in stopped:
                    continue
                at, why, checks = judge(curve(out), ref)
                if checks:
                    n, r, c, gap = checks[-1]
                    log(f"[CHECK] {out.name} {n * EVAL_STEPS // 1000}k: {r:.3f} vs C {c:.3f} ({gap:+.3f})")
                if at is not None:
                    (out / "stop.flag").write_text(why, encoding="utf-8")
                    stopped.add(out)
                    rows[(key, seed)][3] = f"stopped: {why}"
                    log(f"[STOP] {out.name}: {why}")
            write_summary(rows.values())
        for item in list(running):
            key, seed, out, proc, f = item
            if proc.poll() is not None:
                f.close()
                running.remove(item)
                if out not in stopped:
                    rows[(key, seed)][3] = _status(out) or f"exit {proc.returncode}"
                log(f"[END] {out.name} exit {proc.returncode} ({_status(out)})")
        time.sleep(10)
    write_summary(rows.values())
    return rows, stopped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-continue", action="store_true", help="race only")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    ap.add_argument("--extra-steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--check-min", type=float, default=5.0)
    ap.add_argument("--no-clearml", action="store_true")
    a = ap.parse_args(argv)

    if a.dry_run:
        print("queue:", ", ".join(f"{k}{'_s' + str(s) if s else ''} ({EXPERIMENTS[k][0]})" for k, s in QUEUE),
              f"| {a.parallel} at once | yardstick {REF.name}")
        print(f"rule: every {CHECK_EVERY * EVAL_STEPS // 1000}k steps, last-{WINDOW} mean vs C; stop if < C - "
              f"{STOP_GAP} or < C - {WARN_GAP} twice running")
        ref = curve(REF)
        for k in "ABDF":
            at, why, _ = judge(curve(run_dir("front", k)), ref)
            print(f"  replayed on {k}: " + (f"stopped after {at * EVAL_STEPS // 1000}k ({why})" if at else "never stopped"))
        for k, s in QUEUE:
            print("  " + subprocess.list2cmdline(command("front", k, DEFAULT_PARENT, a.timesteps, a.n_envs, s,
                                                         a.no_clearml)[0]))
        return
    if not curve(REF):
        sys.exit(f"no reference curve in {REF}")

    _keep_awake(True)
    try:
        log("race started")
        rows, stopped = race(a)
        finished = [(k, s, out) for k, s, out, st in rows.values()
                    if out not in stopped and _status(out) == "completed"]
        cands = [("C", 0, REF)] + finished
        key, seed, win = max(cands, key=lambda c: np.nan_to_num(last10(c[2]), nan=-1))
        verdict = (f"Winner: **{win.name}** (last-{WINDOW} mean {last10(win):.3f}); "
                   + ", ".join(f"{o.name} {last10(o):.3f}" for _, _, o in cands))
        log(verdict)
        write_summary(rows.values(), verdict)
        if a.no_continue:
            return
        long = RUNS / f"{win.name}_long"
        cmd, _ = command("front", key, str(win.relative_to(CMS)), a.extra_steps, a.n_envs, seed, a.no_clearml)
        proc, f = start(retarget(cmd, long), long)
        log(f"[START] {long.name}: {win.name} gets {a.extra_steps // 1_000_000}M more steps")
        proc.wait()
        f.close()
        log(f"[END] {long.name}: {_status(long)}, last-{WINDOW} mean {last10(long):.3f}")
        laps = RUNS / f"w10_laps_{key}_{EXPERIMENTS[key][0]}{'_s' + str(seed) if seed else ''}_long"
        src = long / "best_model.zip" if (long / "best_model.zip").exists() else win / "best_model.zip"
        cmd, _ = command("laps", key, str(src.relative_to(CMS)), a.timesteps, a.n_envs, seed, a.no_clearml)
        proc, f = start(retarget(cmd, laps), laps)
        log(f"[START] {laps.name} (laps stage from {src.parent.name})")
        proc.wait()
        f.close()
        log(f"[END] {laps.name}: {_status(laps)}, last-{WINDOW} mean laps per episode {last10(laps):.3f}")
        write_summary(rows.values(), verdict + f"\n\nThen: `{long.name}` last-{WINDOW} {last10(long):.3f}; "
                      f"laps stage `{laps.name}` last-{WINDOW} {last10(laps):.3f} laps per episode.")
    except Exception as e:
        log(f"race error: {e!r}")
        raise
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    main()
