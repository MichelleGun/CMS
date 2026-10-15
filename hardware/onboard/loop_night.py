"""
One unattended night aimed at a policy that loops: race PPO variants on the laps task (all built on
C + the ring tracker), stop the clear losers, give the 2 best a long extension, then export every
candidate as a Tello-ready, self-describing file (export_policy.py) and write a morning report.

    python hardware/onboard/loop_night.py --dry-run     # the plan, the commands, the loop demos found
    python hardware/onboard/loop_night.py               # run it (start detached; keeps the PC awake)

Race (laps stage: endless same-side laps from all 9 start zones, measured profile, dr 1), every run
from C's best model (w10_front_C_kl, widened 37 -> 41 inputs with zero weights), C + tracker plus:
  X  trk_combo     U + V + W                          the best guess, started first
  W  trk_pure      no BC term                         nothing pulls it back to demos that never loop
  Y  trk_loopdemo  U + V, BC term on the human flight(s) that really looped (skipped if none)
  U  trk_gamma     gamma 0.995, lambda 0.97           the loop is longer than gamma 0.99's horizon
  V  trk_streak    streak 1.0, cap 4                  re-entering pays more
Control: the existing T run (C + tracker, w10_laps_T_kl_tracker), not re-run.
Each run is one process (--vec dummy): on 2026-10-13 5 runs x 4 processes ran out of Windows commit
(WinError 1455). A run starts only with NEED_GB free, so runs go one after another when memory is short.
Stop rule (w10_race.judge's rule, from 800k steps on): last-10 mean laps per episode vs T's at the
same evaluation; stop if > 0.10 below, or > 0.05 below at two checkpoints in a row.
Then the best --ext-top run(s) (last-10 mean) get --ext-steps more.
Log results/hoop/loop_night.log; report results/hoop/overnight_loop_report.md; policies in
hardware/policies/ (see its README.md).
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
from week10_experiments import CMS, EXPERIMENTS, RUNS, _keep_awake, _status, command  # noqa: E402
from w10_race import STOP_GAP, WARN_GAP, WINDOW, curve, retarget, start  # noqa: E402

PARENT = "results/hoop/runs/w10_front_C_kl/best_model.zip"
RACE = ["X", "W", "Y", "U", "V"]          # start order when memory is short: the best guess first
EVAL_STEPS = 20_000
NEED_GB = 4.0                             # free commit wanted before starting one more run (headroom
#                                           for whatever else runs on the laptop)
RETRIES = 2                               # a run that dies at start (out of memory) is queued again
BACKFILL = ["Y"]                          # the run whose demo term uses the looping human flight(s)
# the yardstick: the 8-env laps run of T that already exists (same evaluation every 20k steps), so no
# memory is spent re-running the control
REF = RUNS / "w10_laps_T_kl_tracker"
CHECK_EVERY, MIN_EVALS = 20, 40          # a checkpoint every 20 evaluations, judged from the 40th
LOG = CMS / "results" / "hoop" / "loop_night.log"
REPORT = CMS / "results" / "hoop" / "overnight_loop_report.md"
FLIGHTS = CMS / "hardware" / "flights"


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


TAG = "laps3"


def out_dir(key) -> Path:
    return RUNS / f"w10_{TAG}_{key}_{EXPERIMENTS[key][0]}"


def loop_demos(min_passes=2):
    """Human flights with at least `min_passes` passes in one flight, that have a demo.npz."""
    found = []
    for d in sorted(FLIGHTS.glob("tello_*")):
        try:
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if int(s.get("passes") or 0) >= min_passes and (d / "demo.npz").is_file():
            found.append(str((d / "demo.npz").relative_to(CMS)))
    return found


EXTRA_ARGS = []                           # --extra-args: added to every training command
N_EVAL_ENVS = None                        # --n-eval-envs: None = the trainer's default


def run_cmd(key, parent, steps, n_envs, out, no_clearml, demos, vec="dummy"):
    cmd, _ = command("laps", key, parent, steps, n_envs, 0, no_clearml)
    cmd = retarget(cmd, out)
    # one process per run (each extra env process costs ~0.7 GB of commit for torch alone), and an
    # evaluation every EVAL_STEPS steps whatever the env count (eval-freq is per env)
    cmd += ["--vec", vec, "--eval-freq", str(max(1, EVAL_STEPS // n_envs))]
    if N_EVAL_ENVS:
        cmd += ["--n-eval-envs", str(N_EVAL_ENVS)]
    cmd += EXTRA_ARGS
    if key in BACKFILL:
        cmd += ["--demos", *demos]          # argparse keeps the last --demos
    return cmd


def free_commit_gb():
    """Windows commit headroom (what WinError 1455 runs out of). inf where it can't be read."""
    if sys.platform != "win32":
        return float("inf")
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullAvailPageFile / 2**30
    except Exception:
        return float("inf")


def judge_laps(run, ref):
    """w10_race.judge's rule, but nothing is judged before MIN_EVALS (laps curves start noisy)."""
    checks, warned = [], False
    for n in range(CHECK_EVERY, min(len(run), len(ref)) + 1, CHECK_EVERY):
        r, c = float(np.mean(run[n - WINDOW:n])), float(np.mean(ref[n - WINDOW:n]))
        gap = r - c
        checks.append((n, r, c, gap))
        if n < MIN_EVALS:
            continue
        if gap < -STOP_GAP:
            return n, f"{-gap:.2f} laps/episode below T after {n} evaluations", checks
        if gap < -WARN_GAP:
            if warned:
                return n, f"more than {WARN_GAP:.2f} below T at two checkpoints in a row " \
                          f"(now {-gap:.2f}, {n} evaluations)", checks
            warned = True
        else:
            warned = False
    return None, "", checks


def last10(d):
    c = curve(d)
    return float(np.mean(c[-WINDOW:])) if c else float("nan")


def steps_of(d):
    try:
        return int(json.loads((d / "passes.json").read_text())[-1]["t"])
    except (OSError, ValueError, IndexError, KeyError, TypeError):
        return 0


def race(a, demos):
    queue = [k for k in RACE if k not in BACKFILL or demos]
    running, rows, stopped, last_check = [], {}, set(), 0.0
    tries, started_at, last_start, waited = {}, {}, 0.0, False
    parents = {}                       # key -> its own crashed run, when resuming (--resume-crashed)
    ref = REF
    while queue or running:
        # one start per minute, and only with memory to spare (5 at once ran out: WinError 1455)
        if queue and len(running) < a.parallel and time.time() - last_start >= 60:
            free = free_commit_gb()
            if free < NEED_GB and running:
                if not waited:
                    log(f"[WAIT] {free:.1f} GB free commit; {queue[0]} waits for memory")
                    waited = True
            else:
                waited = False
                key = queue.pop(0)
                out = out_dir(key)
                rows[key] = [key, out, "running"]
                if _status(out) in ("completed", "stopped"):
                    rows[key][2] = _status(out)
                    log(f"[SKIP] {out.name} already {_status(out)}")
                    continue
                proc, f = start(run_cmd(key, parents.get(key, PARENT), a.timesteps, a.n_envs, out,
                                        a.no_clearml, demos, a.vec), out)
                running.append((key, out, proc, f))
                tries[key] = tries.get(key, 0) + 1
                started_at[key] = last_start = time.time()
                log(f"[START] {out.name} ({a.n_envs} envs, {a.timesteps // 1_000_000}M steps, "
                    f"{free:.1f} GB free before)")
        if time.time() - last_check >= a.check_min * 60:
            last_check = time.time()
            rc = curve(ref)
            for key, out, proc, f in running:
                if out in stopped:
                    continue
                at, why, checks = judge_laps(curve(out), rc)
                if checks:
                    n, r, c, gap = checks[-1]
                    log(f"[CHECK] {out.name} eval {n}: {r:.3f} vs T {c:.3f} ({gap:+.3f})")
                if at is not None:
                    (out / "stop.flag").write_text(why, encoding="utf-8")
                    stopped.add(out)
                    rows[key][2] = f"stopped: {why}"
                    log(f"[STOP] {out.name}: {why}")
            write_report(rows)
        for item in list(running):
            key, out, proc, f = item
            if proc.poll() is not None:
                f.close()
                running.remove(item)
                if out not in stopped:
                    rows[key][2] = _status(out) or f"exit {proc.returncode}"
                log(f"[END] {out.name} exit {proc.returncode} ({_status(out)}), last-10 {last10(out):.3f}")
                # any crash (not a stop) is retried: out-of-memory / WinError 1450 can hit mid-run too
                if (getattr(a, "resume_crashed", False) and proc.returncode != 0 and out not in stopped
                        and curve(out) and (out / "latest_model.zip").exists() and tries[key] <= RETRIES):
                    # keep what it learned: move the crashed run aside and continue from its latest model
                    part = out.with_name(f"{out.name}_part{tries[key]}")
                    shutil.move(str(out), str(part))
                    parents[key] = str(part.relative_to(CMS))
                    queue.append(key)
                    log(f"[RESUME] {out.name} crashed after {steps_of(part):,} steps; continues from {part.name}")
                elif proc.returncode != 0 and out not in stopped and tries[key] <= RETRIES:
                    queue.append(key)
                    log(f"[RETRY] {out.name} crashed (try {tries[key]}); queued again from the start")
        time.sleep(10)
    return rows, stopped


def extend(a, rows, stopped, demos):
    cands = [(k, o) for k, o, st in rows.values() if o not in stopped and (o / "latest_model.zip").exists()]
    cands.sort(key=lambda c: np.nan_to_num(last10(c[1]), nan=-1), reverse=True)
    top = cands[:a.ext_top]
    log("extension: " + ", ".join(f"{o.name} ({last10(o):.3f})" for _, o in top))
    procs = []
    if getattr(a, "relabel", False):
        # one more DAgger-style pass: the teacher labels the loop leg on the WINNER's own states
        # (they differ from the ones the race started with); the file joins the --demos glob
        sys.path.insert(0, str(HERE.parent))
        from privileged.loop_labels import collect
        for key, o in top:
            try:
                collect(o / "latest_model.zip", 60, seed=71_000, log=log)
            except BaseException as e:               # SystemExit included: never stop the night
                log(f"relabel failed for {o.name}: {e!r}")
    for key, o in top:
        ext = RUNS / f"{o.name}_ext"
        if _status(ext) == "completed":
            log(f"[SKIP] {ext.name} already completed")
            continue
        cmd = run_cmd(key, str(o.relative_to(CMS)), a.ext_steps, a.ext_envs, ext, a.no_clearml, demos, a.vec)
        proc, f = start(cmd, ext)
        procs.append((ext, proc, f))
        log(f"[START] {ext.name} (+{a.ext_steps // 1_000_000}M steps, {a.ext_envs} envs)")
    for ext, proc, f in procs:
        proc.wait()
        f.close()
        log(f"[END] {ext.name}: {_status(ext)}, last-10 {last10(ext):.3f}")
    return [RUNS / f"{o.name}_ext" for _, o in top]


def write_report(rows, exts=(), export_md=""):
    L = [f"# Overnight loop training ({datetime.now():%Y-%m-%d %H:%M})", "",
         "Goal: enter the hoop, loop round, re-enter, as many times as possible. Score = **laps per "
         "episode** (passes in a 90 s episode, same-side laps from all 9 start zones, measured profile, "
         "full randomisation). Written by `hardware/onboard/loop_night.py`.", "",
         "## Race (3 envs each, from C's best model, ring tracker on)", "",
         "| Run | Extra settings | Steps | Last-10 laps/ep | Best eval | State |", "|---|---|---|---|---|---|"]
    base = len(EXPERIMENTS["T"][1])
    rc = curve(REF)
    L.append(f"| {REF.name} (control, 8 envs, earlier) | (control) | {steps_of(REF):,} | {last10(REF):.3f} | "
             f"{(max(rc) if rc else float('nan')):.3f} | yardstick |")
    for key, o, st in rows.values():
        c = curve(o)
        extra = " ".join(EXPERIMENTS[key][1][base:]) or "(control)"
        if key in BACKFILL:
            extra += " + loop demos"
        L.append(f"| {o.name} | {extra} | {steps_of(o):,} | {last10(o):.3f} | "
                 f"{(max(c) if c else float('nan')):.3f} | {st} |")
    if exts:
        L += ["", "## Extension (the 2 best, continued)", "", "| Run | Steps | Last-10 laps/ep | Best eval | State |",
              "|---|---|---|---|---|"]
        for e in exts:
            c = curve(e)
            L.append(f"| {e.name} | {steps_of(e):,} | {last10(e):.3f} | {(max(c) if c else float('nan')):.3f} "
                     f"| {_status(e)} |")
    if export_md:
        L += ["", export_md]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--n-envs", type=int, default=3)
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="dummy",
                    help="subproc = one process per env (faster, ~0.7 GB commit each)")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--ext-steps", type=int, default=3_000_000)
    ap.add_argument("--ext-envs", type=int, default=4)
    ap.add_argument("--ext-top", type=int, default=1, help="how many of the best runs to extend")
    ap.add_argument("--check-min", type=float, default=5.0)
    ap.add_argument("--export-episodes", type=int, default=36)
    ap.add_argument("--no-clearml", action="store_true")
    ap.add_argument("--resume-crashed", action="store_true",
                    help="a run that crashes after its first evaluation continues from its own latest model "
                         "(the crashed folder is kept as <run>_partN) instead of starting again")
    ap.add_argument("--relabel", action="store_true",
                    help="before the extension, collect new teacher loop labels on the winner's states")
    ap.add_argument("--keys", nargs="*", default=None, help="runs to race (default: the round-3 set)")
    ap.add_argument("--parent", default=None, help="model or run folder every run starts from")
    ap.add_argument("--tag", default=None, help="run folders are w10_<tag>_<key>_<name>")
    ap.add_argument("--n-eval-envs", type=int, default=None,
                    help="evaluation copies per run (default: the trainer's: --n-envs up to 4 with "
                         "--vec subproc, 1 with dummy). Same layouts, faster; ~0.3-0.5 GB each")
    ap.add_argument("--extra-args", default="",
                    help="more train_onboard.py flags for every run, in quotes, e.g. "
                         "\"--checkpoint-min 10 --min-std-warn 0.08\"")
    a = ap.parse_args(argv)
    global PARENT, RACE, TAG, REPORT, EXTRA_ARGS, N_EVAL_ENVS
    import shlex
    EXTRA_ARGS = shlex.split(a.extra_args) if a.extra_args else []
    N_EVAL_ENVS = a.n_eval_envs
    if a.parent:
        PARENT = a.parent
    if a.keys:
        RACE = list(a.keys)
    if a.tag:
        TAG = a.tag
        REPORT = REPORT.with_name(f"overnight_loop_report_{TAG}.md")
    demos = loop_demos()

    if a.dry_run:
        print("loop demos:", demos or "none found (Y is skipped)")
        for key in [k for k in RACE if k not in BACKFILL or demos]:
            print(" ", subprocess.list2cmdline(run_cmd(key, PARENT, a.timesteps, a.n_envs, out_dir(key),
                                                       a.no_clearml, demos)))
        return

    _keep_awake(True)
    try:
        log(f"night started; loop demos: {demos or 'none'}")
        rows, stopped = race(a, demos)
        write_report(rows)
        exts = []
        try:
            exts = extend(a, rows, stopped, demos)
        except Exception as e:                       # the export must still happen
            log(f"extension error: {e!r}")
        write_report(rows, exts)
        runs = [REF] + [o for _, o, _ in rows.values()] + [e for e in exts if e.exists()]
        try:
            from export_policy import export
            md = export(runs, episodes=a.export_episodes, log=log)
        except Exception as e:
            md = f"Export failed: {e!r}. Run `python hardware/onboard/export_policy.py` by hand."
            log(md)
        write_report(rows, exts, md)
        log(f"night done; report {REPORT}")
    except Exception as e:
        log(f"night error: {e!r}")
        raise
    finally:
        _keep_awake(False)


if __name__ == "__main__":
    main()
