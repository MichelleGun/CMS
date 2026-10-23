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
  3. every candidate (each run's best and final model, and the parent) is re-scored on the same fixed
     layouts with more episodes than one evaluation (72 laps episodes + 60 slanted starts, the flight
     loader): score = laps/episode + 1.0 x oblique entry (--oblique-weight; the corner clipping seen
     on the real drone counts fully). A challenger replaces the parent only if it wins by more than
     --margin 0.10 (one 36-episode number wobbles about +-0.15). Cache: mg_loop_rescore.json.
Each round trains with seed = round number (a continued run is never an exact repeat). A lock file
(mg_loop.lock) stops a second copy, so the logon task and a manual start can't both run.
State: results/hoop/mg_loop_state.json; log: results/hoop/mg_loop.log; every round's runs are
results/hoop/runs/w10_mg<k>_<key>_<name>; Tello-ready files in hardware/policies/ (FLY_THIS_FIRST.txt).
"""
import argparse
import json
import os
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
# round 1 of the optimiser (2026-10-17..19) tried D9, E9, C9, DE9, OF, OS: E9 then C9 won. Round 2
# (2026-10-23) builds on C9 (see week10_experiments): corners (CZ, CO, CE), loops (CD), stability (CL)
POOL = ["CZ", "CE", "CD", "CL", "CO"]
RESCORE = CMS / "results" / "hoop" / "mg_loop_rescore.json"
LOCK = CMS / "results" / "hoop" / "mg_loop.lock"
MARGIN = 0.10          # a challenger must beat the parent by this much (one 36-episode number wobbles +-0.15)


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


def _pid_alive(pid) -> bool:
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def take_lock() -> bool:
    """One optimiser at a time (a logon task and a manual start must not both run)."""
    try:
        pid = int(LOCK.read_text().strip())
        if pid != os.getpid() and _pid_alive(pid):
            return False
    except (OSError, ValueError):
        pass
    LOCK.write_text(str(os.getpid()))
    return True


def candidates(run: Path):
    out = []
    for name in ("best_model.zip", "final_model.zip", "latest_model.zip"):
        p = run / name
        if p.is_file() and not (name == "latest_model.zip" and (run / "final_model.zip").is_file()):
            out.append(p)
    return out


def rescore(model: Path, episodes=72, oblique_episodes=60) -> dict:
    """Score one model on the same fixed layouts as every other (export_policy.score: the flight
    loader, the laps task, slanted starts), with more episodes than an evaluation, cached by file."""
    from export_policy import score
    cache = {}
    try:
        cache = json.loads(RESCORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    key = f"{model.relative_to(CMS).as_posix()}@{int(model.stat().st_mtime)}@{episodes}/{oblique_episodes}"
    if key not in cache:
        s = score(model, episodes=episodes, oblique_episodes=oblique_episodes)
        cache[key] = {k: v for k, v in s.items() if not k.startswith("_")}
        tmp = RESCORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        tmp.replace(RESCORE)
    return cache[key]


def total(s: dict, oblique_weight: float) -> float:
    return float(s["lpe"]) + oblique_weight * float(s.get("oblique_entry") or 0.0)


def best_of_run(run: Path, a):
    """(score, model, scores) of a run's best candidate model, or (None, None, None)."""
    best = (None, None, None)
    for m in candidates(run):
        try:
            s = rescore(m, a.rescore_episodes, a.oblique_episodes)
        except Exception as e:
            log(f"  rescore failed for {m.relative_to(CMS)}: {e!r}")
            continue
        t = total(s, a.oblique_weight)
        log(f"    {m.relative_to(CMS).as_posix()}: laps/ep {s['lpe']:.2f}, 2+ passes {s['two_plus']:.0%}, "
            f"oblique {s['oblique_entry']:.0%} -> {t:.3f}")
        if best[0] is None or t > best[0]:
            best = (t, m, s)
    return best


def choose(parent_score, challengers, margin=MARGIN):
    """The challenger (dict with 'score') that replaces the parent, or None: only a clear win counts."""
    best = max((c for c in challengers if c.get("score") is not None), key=lambda c: c["score"], default=None)
    if best is None:
        return None
    if parent_score is None or best["score"] > parent_score + margin:
        return best
    return None


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
    if any("dagger_mg_r" in " ".join(EXPERIMENTS[x][1]) for x in keys):
        try:
            from privileged.dagger_from_policy import relabel
            relabel(CMS / st["parent"], "mg", episodes=a.relabel_episodes, log=log)
        except BaseException as e:                     # never let one step end the loop
            log(f"relabel failed: {e!r}")
    cmd = [sys.executable, str(HERE / "loop_night.py"), "--vec", "subproc", "--keys", *keys,
           "--parent", st["parent"], "--tag", tag, "--timesteps", str(a.timesteps),
           "--n-envs", str(a.n_envs), "--parallel", str(a.parallel), "--n-eval-envs", str(a.n_eval_envs),
           "--ext-top", "0", "--resume-crashed",
           # a different seed every round: a continued run is never a repeat (rounds 4 and 5 of
           # 2026-10-18 trained the identical C9 run, both 2.155)
           "--extra-args", f"--seed {k}"]
    log("  " + subprocess.list2cmdline(cmd))
    rc = subprocess.run(cmd, cwd=CMS).returncode
    if rc != 0:
        log(f"  loop_night ended with exit {rc}")
    from week10_experiments import EXPERIMENTS as EX
    # fair re-score: every candidate on the same layouts, more episodes than one evaluation
    log(f"  re-scoring ({a.rescore_episodes} laps episodes + {a.oblique_episodes} slanted starts, "
        f"score = laps/ep + {a.oblique_weight} x oblique entry)")
    p_score, p_model, p_s = best_of_run(CMS / st["parent"], a)
    results = []
    for key in keys:
        name = f"w10_{tag}_{key}_{EX[key][0]}"
        sc, model, s = best_of_run(RUNS / name, a)
        results.append(dict(key=key, run=name, score=sc, model=model and model.relative_to(CMS).as_posix(),
                            lpe=s and s["lpe"], two_plus=s and s["two_plus"],
                            oblique=s and s.get("oblique_entry")))
    st["parent_score"] = p_score
    win = choose(p_score, results, a.margin)
    if win:
        log(f"  new best: {win['run']} ({win['score']:.3f} > parent {p_score} + {a.margin})")
        st.update(parent=f"results/hoop/runs/{win['run']}", parent_key=win["key"], parent_score=win["score"],
                  parent_model=win["model"])
    else:
        log(f"  the parent stays best ({p_score if p_score is None else round(p_score, 3)}; "
            f"no challenger beat it by more than {a.margin})")
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
    ap.add_argument("--rescore-episodes", type=int, default=72)
    ap.add_argument("--oblique-episodes", type=int, default=60)
    ap.add_argument("--oblique-weight", type=float, default=1.0,
                    help="score = laps/ep + this x oblique entry (Michelle, 2026-10-23: corners count fully)")
    ap.add_argument("--margin", type=float, default=MARGIN)
    a = ap.parse_args(argv)
    st = load_state()
    if a.dry_run:
        keys, nxt = plan(st, a.pool)
        print(f"round {st['round'] + 1}: parent {st['parent']} ({st['parent_key']}), keys {keys}, "
              f"seed {st['round'] + 1}; tried so far {st['tried']}; stop file {STOP}")
        return
    if not take_lock():
        print(f"another loop_forever is running (pid in {LOCK}); not starting a second one")
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
