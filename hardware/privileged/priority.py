"""
Lower (or restore) the CPU priority of everything training, so a real flight gets the CPU first.

    python hardware/privileged/priority.py --low       # before flying
    python hardware/privileged/priority.py --normal    # after
    python hardware/privileged/priority.py             # just show

The flight loop (fly_hoop.py) must tick at 15 Hz and decode the camera; with all 16 cores busy
training it can be starved (laggy sticks, uneven logs). Below-normal training only runs a little
slower. Matches python processes running train_onboard.py, week10_experiments.py, autopilot.py or
dagger.py, plus all their child processes (the parallel simulators). Runs started later (the
autopilot's next job) begin at normal priority again: run --low again if you are still flying.
"""
import argparse
import sys

import psutil

MATCH = ("train_onboard.py", "week10_experiments.py", "autopilot.py", "dagger.py")
LEVELS = {"low": psutil.BELOW_NORMAL_PRIORITY_CLASS, "normal": psutil.NORMAL_PRIORITY_CLASS}
NAMES = {v: k for k, v in LEVELS.items()}


def training_processes():
    found = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "python" in (p.info["name"] or "").lower() and any(m in cmd for m in MATCH):
            found[p.pid] = p
            for c in p.children(recursive=True):
                found[c.pid] = c
    return list(found.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--low", action="store_true")
    g.add_argument("--normal", action="store_true")
    a = ap.parse_args(argv)
    procs = training_processes()
    if not procs:
        print("no training processes running")
        return
    target = LEVELS["low"] if a.low else LEVELS["normal"] if a.normal else None
    changed = 0
    for p in procs:
        try:
            if target is not None and p.nice() != target:
                p.nice(target)
                changed += 1
            print(f"{p.pid:>7}  {NAMES.get(p.nice(), p.nice()):>7}  {' '.join(p.cmdline())[:90]}")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"{p.pid:>7}  skipped ({e.__class__.__name__})")
    if target is not None:
        print(f"\n{changed} of {len(procs)} processes set to {NAMES[target]}")


if __name__ == "__main__":
    sys.exit(main())
