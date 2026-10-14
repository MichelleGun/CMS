"""
Upload every run and flight made so far to ClearML (the backfill). New runs sync themselves
(see clearml_sync.py); this is for everything from before that, and is safe to re-run: a folder
whose meta.json already has a clearml_task_id, or whose name already exists in its project, is
skipped.

    python scripts/clearml_upload_runs.py --dry-run          # list what would be uploaded
    python scripts/clearml_upload_runs.py                     # upload all of it
    python scripts/clearml_upload_runs.py --only onboard-rl clones

What goes where (ClearML project <- folder):
    Tello/onboard-rl <- results/hoop/runs/onboard*        (camera policies, the ones the Tello flies)
    Tello/pybullet   <- results/hoop/runs/<everything else> (PyBullet state policies, the teachers)
    Tello/clones     <- hardware/flights/bc/*               (behaviour clones)
    Tello/flights    <- hardware/flights/tello_*            (real and dry-run Tello flights)

Order matters for the links: runs first (oldest first, so a continued run finds its parent),
clones, then flights (so a flight can point at the policy it flew).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMS = HERE.parent
sys.path.insert(0, str(HERE))
import clearml_sync  # noqa: E402

GROUPS = ("onboard-rl", "pybullet", "clones", "flights")


def _meta(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def collect():
    """[(group, folder)] in upload order."""
    runs = CMS / "results" / "hoop" / "runs"
    out = []
    run_dirs = sorted((p for p in runs.iterdir() if p.is_dir()),
                      key=lambda p: (_meta(p).get("created_at") or "", p.name)) if runs.is_dir() else []
    for p in run_dirs:
        out.append(("onboard-rl" if p.name.startswith("onboard") or _meta(p).get("task") == "onboard_hoop"
                    else "pybullet", p))
    bc = CMS / "hardware" / "flights" / "bc"
    if bc.is_dir():
        out += [("clones", p) for p in sorted(bc.iterdir()) if p.is_dir()]
    flights = CMS / "hardware" / "flights"
    out += [("flights", p) for p in sorted(flights.glob("tello_*")) if p.is_dir()]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    ap.add_argument("--only", nargs="*", choices=GROUPS, default=None)
    a = ap.parse_args(argv)

    items = [(g, p) for g, p in collect() if not a.only or g in a.only]
    if a.dry_run:
        for g, p in items:
            m = _meta(p)
            files = [f.name for f in p.iterdir() if f.is_file()]
            done = " (already in ClearML)" if m.get("clearml_task_id") else ""
            print(f"{'Tello/' + g:18s} {p.name:40s} {m.get('final_status') or '-':18s} "
                  f"{len(files):3d} files{done}")
        counts = {g: sum(1 for x, _ in items if x == g) for g in GROUPS}
        print(f"\n{len(items)} folders: {counts}")
        reason = clearml_sync.disabled_reason()
        print("ClearML: " + (f"OFF - {reason}" if reason else "ready"))
        return

    if not clearml_sync.enabled():
        sys.exit("[clearml] nothing uploaded; fix the reason above and run again")
    tally = {}
    for i, (g, p) in enumerate(items, 1):
        res = clearml_sync.record(g, p)
        tally[res] = tally.get(res, 0) + 1
        print(f"[{i}/{len(items)}] Tello/{g}/{p.name}: {res}")
    print(f"\n[done] {tally}")


if __name__ == "__main__":
    main()
