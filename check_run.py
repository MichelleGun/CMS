"""Quick progress check for a training run, compared with the current best (O).

    python check_run.py results/hoop/runs/w10_steven_OZ_oblique_clearance

Reads only the run's passes.json and live_status.json; it never touches the training.
Zone 9 (start under the hoop) is reported separately, since the real drone never starts there.
"""
import json
import statistics as st
import sys
from pathlib import Path

BASELINE = Path("results/hoop/runs/w10_n9_O_oblique")
ZONES_REAL = [str(z) for z in range(1, 9)]


def load(run):
    p = run / "passes.json"
    if not p.exists():
        sys.exit(f"No passes.json in {run} yet: wait for the first evaluation (every 20,000 steps).")
    return json.loads(p.read_text(encoding="utf-8"))


def no9(e):
    z = e.get("zones") or {}
    vals = [z[k] for k in ZONES_REAL if k in z]
    return st.mean(vals) if vals else float("nan")


def summary(evals, n=10):
    last = evals[-n:]
    return {
        "all": st.mean(e["mean"] for e in last),
        "no9": st.mean(no9(e) for e in last),
        "z9": st.mean((e.get("zones") or {}).get("9", 0) for e in last),
        "zones": {k: st.mean((e.get("zones") or {}).get(k, 0) for e in last) for k in ZONES_REAL},
    }


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    ev, base = load(run), load(BASELINE)

    status = {}
    ls = run / "live_status.json"
    if ls.exists():
        status = json.loads(ls.read_text(encoding="utf-8"))
    start = ev[0]["t"] if ev and "t" in ev[0] else None
    now = status.get("current_timestep")
    print(f"Run: {run.name}")
    print(f"  status: {status.get('status', '?')}, evaluations so far: {len(ev)}")
    if start and now:
        print(f"  steps trained in this run: {now - start + 20000:,}")

    n = min(10, len(ev))
    me, ref = summary(ev, n), summary(base, 10)
    print(f"\nAverage laps per episode over the last {n} evaluations (higher is better):")
    print(f"  {'':28}{'this run':>10}{'O (best)':>10}")
    print(f"  {'all 9 zones':28}{me['all']:>10.2f}{ref['all']:>10.2f}")
    print(f"  {'zones 1-8 (real starts)':28}{me['no9']:>10.2f}{ref['no9']:>10.2f}   <- the one to watch")
    print(f"  {'zone 9 (under the hoop)':28}{me['z9']:>10.2f}{ref['z9']:>10.2f}")
    print("\n  per zone:")
    for k in ZONES_REAL:
        print(f"    zone {k}: {me['zones'][k]:>5.2f}   (O {ref['zones'][k]:.2f})")

    if len(ev) >= 20:
        first = st.mean(no9(e) for e in ev[:10])
        print(f"\nTrend (zones 1-8): first 10 evaluations {first:.2f} -> last 10 {me['no9']:.2f}")
    if n < 10:
        print(f"\nOnly {n} evaluations so far: scores jump about +-0.15 between evaluations, so wait "
              "for 10+ before reading much into it.")
    gap = me["no9"] - ref["no9"]
    verdict = ("ahead of O" if gap > 0.15 else "behind O" if gap < -0.15 else
               "about level with O (within the noise)")
    print(f"\nVerdict so far (zones 1-8): {gap:+.2f} -> {verdict}")
    print("Laps don't show clean corner entries: when the run is done, export it to see 'Oblique entry'.")


if __name__ == "__main__":
    main()
