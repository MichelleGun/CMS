"""
Keep improving the loop policy until told to stop: round after round, race the best variant so far
against the next untried idea, export and score both, and carry the better one forward.

    python hardware/onboard/launch_detached.py results/hoop/mg_loop -- hardware/onboard/loop_forever.py
    python hardware/onboard/loop_forever.py --dry-run        # the plan for the next round
    type nul > results\\hoop\\mg_loop.stop                    # stop after the current round

Each round k:
  1. a DAgger relabelling pass of the current best policy (privileged/dagger_from_policy.py, tag mg):
     the teacher labels the states it flies now; D-type experiments' imitation term pools these with
     n8's labels (week10_experiments._MG_DEMOS);
  2. loop_night.py --tag mg<k> races two experiment keys from the best run (the winner's own key,
     continued, and the next idea from --pool), at most --parallel at once, below-normal priority,
     crashed runs resumed from their own checkpoints, then exports every run;
  3. score = laps/episode + 0.5 x oblique entry (export_policy, flight loader, sim): the corner
     clipping seen on the real drone counts as well as the loops. The best of the round's runs and
     the current parent becomes the next parent.
State: results/hoop/mg_loop_state.json; log: results/hoop/mg_loop.log; every round's runs are
results/hoop/runs/w10_mg<k>_<key>_<name>; Tello-ready files in hardware/policies/ (FLY_THIS_FIRST.txt).
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMS = HERE.parents[1]
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RUNS = CMS / "results" / "hoop" / "runs"
STATE = CMS / "results" / "hoop" / "mg_loop_state.json"
LOG = CMS / "results" / "hoop" / "mg_loop.log"
STOP = CMS / "results" / "hoop" / "mg_loop.stop"
START_PARENT = ("results/hoop/runs/w10_n9_O_oblique", "O")
POOL = ["D9", "E9", "C9", "DE9", "OF", "OS", "ON"]


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(round=0, parent=START_PARENT[0], parent_key=START_PARENT[1], parent_score=None,
                    tried=[], history=[])


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
    tmp.replace(STATE)


def score_of(run_name: str):
    """Best export score of a run's policies: (score, row) or (None, None) if never exported."""
    from export_policy import all_exported
    best = (None, None)
    for r in all_exported():
        if r["run"] != run_name or r.get("episodes", 0) < 36:
            continue
        s = r["lpe"] + 0.5 * float(r.get("oblique_entry") or 0.0)
        if best[0] is None or s > best[0]:
            best = (s, r)
    return best


def plan(st, pool):
    """This round's two keys: the parent's own key continued, and the next untried idea."""
    nxt = next((k for k in pool if k not in st["tried"] and k != st["parent_key"]), None)
    keys = [st["parent_key"]] + ([nxt] if nxt else [])
    return keys, nxt


def run_round(st, a):
    from week10_experiments import EXPERIMENTS
    k = st["round"] + 1
    tag = f"mg{k}"
    keys, nxt = plan(st, a.pool)
    keys = [x for x in keys if x in EXPERIMENTS]
    log(f"round {k}: parent {Path(st['parent']).name} ({st['parent_key']}), keys {keys}")
    if any(x.startswith("D") for x in keys):
        try:
            from privileged.dagger_from_policy import relabel
            relabel(CMS / st["parent"], "mg", episodes=a.relabel_episodes, log=log)
        except BaseException as e:                     # never let one step end the loop
            log(f"relabel failed: {e!r}")
    cmd = [sys.executable, str(HERE / "loop_night.py"), "--vec", "subproc", "--keys", *keys,
           "--parent", st["parent"], "--tag", tag, "--timesteps", str(a.timesteps),
           "--n-envs", str(a.n_envs), "--parallel", str(a.parallel), "--n-eval-envs", str(a.n_eval_envs),
           "--ext-top", "0", "--resume-crashed"]
    log("  " + subprocess.list2cmdline(cmd))
    rc = subprocess.run(cmd, cwd=CMS).returncode
    if rc != 0:
        log(f"  loop_night ended with exit {rc}")
    from week10_experiments import EXPERIMENTS as EX
    results = []
    for key in keys:
        name = f"w10_{tag}_{key}_{EX[key][0]}"
        s, row = score_of(name)
        results.append(dict(key=key, run=name, score=s, lpe=row and row["lpe"],
                            two_plus=row and row["two_plus"], oblique=row and row.get("oblique_entry"),
                            file=row and row["name"]))
        log(f"  {name}: score {s if s is None else round(s, 3)}"
            + (f" (laps/ep {row['lpe']:.2f}, 2+ passes {row['two_plus']:.0%}, oblique {row.get('oblique_entry')})"
               if row else " (not exported)"))
    if st["parent_score"] is None:
        st["parent_score"], _ = score_of(Path(st["parent"]).name)
    best = max((r for r in results if r["score"] is not None), key=lambda r: r["score"], default=None)
    if best and (st["parent_score"] is None or best["score"] > st["parent_score"]):
        log(f"  new best: {best['run']} ({best['score']:.3f} > {st['parent_score']})")
        st.update(parent=f"results/hoop/runs/{best['run']}", parent_key=best["key"], parent_score=best["score"])
    else:
        log(f"  the parent stays best ({st['parent_score']})")
    if nxt:
        st["tried"].append(nxt)
    elif a.pool:                       # every idea tried: start the pool again from the new best
        st["tried"] = []
    st["round"] = k
    st["history"].append(dict(round=k, keys=keys, results=results, parent_after=st["parent"],
                              at=datetime.now().isoformat(timespec="minutes")))
    save_state(st)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rounds", type=int, default=1000, help="stop after this many rounds")
    ap.add_argument("--pool", nargs="*", default=POOL)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-eval-envs", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=2, help="runs at once (CPU: ~5 processes each)")
    ap.add_argument("--relabel-episodes", type=int, default=36)
    a = ap.parse_args(argv)
    st = load_state()
    if a.dry_run:
        keys, nxt = plan(st, a.pool)
        print(f"round {st['round'] + 1}: parent {st['parent']} ({st['parent_key']}), keys {keys}; "
              f"tried so far {st['tried']}; stop file {STOP}")
        return
    STOP.unlink(missing_ok=True)
    log(f"loop started (round {st['round'] + 1}, parent {st['parent']})")
    for _ in range(a.rounds):
        if STOP.exists():
            log("stop file found: stopping")
            break
        try:
            run_round(st, a)
        except Exception as e:
            log(f"round error: {e!r}; waiting 5 min, then the next round")
            time.sleep(300)
    log("loop ended")


if __name__ == "__main__":
    main()
