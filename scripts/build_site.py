"""
Build the results shown by the dashboard's Overview, Policies and Training tabs, and the snapshot
the online copy (GitHub Pages) reads instead of a server. There is one frontend, webapp/frontend:
the local server serves it as is, and scripts/build_pages.py packs it for Pages.

    python scripts/build_site.py              # refresh the data (leaderboard, curves, rounds, timeline)
    python scripts/build_site.py --record     # also re-record the 3D replays (needs the simulator)

What it writes into webapp/frontend/:
  data/leaderboard.json   every exported policy's sim scores (hardware/policies/*.meta.json)
  data/curves.json        learning curves (laps per episode per evaluation) of the milestone runs
  data/optimiser.json     the loop optimiser's rounds and fair re-scores (results/hoop/mg_loop_*)
  data/hrl.json           the hierarchical-RL comparison (results/hoop/hrl_compare.json), if done
  data/timeline.json      "what changed, why, impact" (HANDOVER.md section 7)
  data/milestones.json    the milestone policies, with their replay files
  data/replays/<id>.json  (--record) each milestone policy flying the same 3 start layouts, every
                          tick: drone x, y, z, yaw; ring centre and axis; passes; HRL skill
  data/api/...            the dashboard's own GET answers (config, a few runs and their configs) for
                          the online copy: /api/runs/x is saved as data/api/runs/x.json (js/api.js)
No camera frames, flight logs, demos or report drafts are published.
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "webapp" / "frontend"
DATA = FRONT / "data"
RUNS = ROOT / "results" / "hoop" / "runs"
sys.path.insert(0, str(ROOT / "hardware"))

# (id, label, what it shows, model path relative to the repo)
MILESTONES = [
    ("T", "T: ring tracker", "C + the ring tracker; learning the laps task from demos", "results/hoop/runs/w10_laps_T_kl_tracker/best_model.zip"),
    ("W", "W: no imitation", "pure PPO: the demos never loop, so imitating them was dropped", "results/hoop/runs/w10_laps3_W_trk_pure/best_model.zip"),
    ("n8", "n8: memory", "1 s of memory + asymmetric critic + DAgger teacher: the first policy that loops", "results/hoop/runs/n8_dapg_nodrain/best_model.zip"),
    ("O", "O: slanted starts", "practice at corner approaches", "results/hoop/runs/w10_n9_O_oblique/best_model.zip"),
    ("E9", "E9: stress drone", "trained with a slower, driftier drone in 30 % of episodes", "results/hoop/runs/w10_mg2_E9_stress/best_model.zip"),
    ("C9", "C9: loop practice (best)", "more slanted starts and starts just before the hoop", "results/hoop/runs/w10_mg6_C9_loop_practice/final_model.zip"),
    ("HRLp", "HRL: fixed switch", "skills: enter until the first pass, then loop", "results/hoop/runs/hrl_v1/phase_bundle.json"),
    ("HRLm", "HRL: learned manager", "a PPO manager picks a skill every 0.53 s", "results/hoop/runs/hrl_v1/best_model.zip"),
]
CURVE_RUNS = [("T", "w10_laps_T_kl_tracker"), ("W", "w10_laps3_W_trk_pure"), ("n8", "n8_dapg_nodrain_cont2"),
              ("O", "w10_n9_O_oblique"), ("E9", "w10_mg2_E9_stress"), ("C9", "w10_mg6_C9_loop_practice"),
              ("HRLm", "hrl_v1")]
REPLAY_EPISODES = 3
REPLAY_SEED = 950_000_000          # export_policy's layouts: every policy flies the same starts


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write(name, obj):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def leaderboard():
    from onboard.export_policy import all_exported
    rows = []
    for r in all_exported():
        if r.get("episodes", 0) < 36:
            continue
        rows.append(dict(file=r["name"], run=r["run"], kind=r["kind"], lpe=round(r["lpe"], 3),
                         two_plus=round(r["two_plus"], 3), best=r["best_streak"],
                         oblique=r.get("oblique_entry"), memory=bool(r.get("memory"))))
    rows.sort(key=lambda r: r["lpe"], reverse=True)
    return rows


def curves():
    out = []
    for key, run in CURVE_RUNS:
        p = _json(RUNS / run / "passes.json", [])
        if not p:
            continue
        t0 = p[0]["t"]
        pts = [[round((e["t"] - t0) / 1e6, 3), round(float(e["mean"]), 3)] for e in p]
        step = max(1, len(pts) // 150)
        out.append(dict(id=key, run=run, points=pts[::step]))
    return out


def optimiser():
    st = _json(ROOT / "results" / "hoop" / "mg_loop_state.json", {}) or {}
    rounds = []
    for h in st.get("history", []):
        rounds.append(dict(round=h["round"], keys=h["keys"], at=h.get("at"), parent_after=Path(h["parent_after"]).name,
                           results=[dict(key=r["key"], score=r.get("score"), lpe=r.get("lpe"),
                                         two_plus=r.get("two_plus"), oblique=r.get("oblique")) for r in h["results"]]))
    return dict(parent=Path(st.get("parent", "")).name, parent_score=st.get("parent_score"), rounds=rounds)


def timeline():
    text = (ROOT / "HANDOVER.md").read_text(encoding="utf-8")
    sec = text.split("## 7.", 1)[-1]
    rows = []
    for line in sec.splitlines():
        if not re.match(r"^\| 20\d\d-", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4:
            rows.append(dict(date=cells[0], change=cells[1], why=cells[2], impact=cells[3]))
    return rows


def record(mid, path, episodes=REPLAY_EPISODES):
    """Fly one policy through the flight loader on the export layouts and keep every tick."""
    from onboard import spec
    from onboard.export_policy import env_kwargs
    from onboard.gym_env import OnboardHoopEnv
    from onboard.policy import load_policy
    fn = load_policy(ROOT / path)
    env = OnboardHoopEnv(seed=REPLAY_SEED, **env_kwargs(fn.meta, fn.obs_width == spec.TRACKER_OBS_DIM))
    env.set_episode_seed(REPLAY_SEED)
    eps = []
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            if hasattr(fn, "reset"):
                fn.reset()
            ring = env.sim.ring
            ticks, events, done, info = [], [], False, {}
            while not done:
                a = fn(obs)
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
                s, r = env.sim, env.sim.ring
                n = r.normal
                t = [round(float(v), 3) for v in (*s.pos, math.radians(s.yaw_deg), *r.centre, *n)]
                t.append(int(info.get("passes", 0)))
                t.append(int(getattr(fn, "current", -1) if getattr(fn, "current", None) is not None else -1))
                ticks.append(t)
                if info.get("phase") not in (None, "flying"):
                    events.append([len(ticks) - 1, info["phase"]])
            eps.append(dict(start=str(info.get("start_type", "?")), passes=int(info.get("passes", 0)),
                            end=info.get("phase") if term else "timeout", ring_radius=round(ring.radius, 3),
                            ring_noodle=round(ring.noodle_m, 3), events=events, ticks=ticks))
    finally:
        env.close()
    skills = None
    if getattr(fn, "skill_paths", None):
        skills = [Path(p).name.replace("w10_hrl_", "").replace("w10_", "") for p in fn.skill_paths]
    (DATA / "replays").mkdir(parents=True, exist_ok=True)
    (DATA / "replays" / f"{mid}.json").write_text(json.dumps(dict(
        id=mid, hz=spec.CTRL_HZ, fields=["x", "y", "z", "yaw", "rx", "ry", "rz", "nx", "ny", "nz", "passes", "skill"],
        skills=skills, episodes=eps), separators=(",", ":")), encoding="utf-8")
    return [e["passes"] for e in eps]


# Runs the online copy lists in its Simulator / History tabs: the milestones and the optimiser's rounds.
SNAPSHOT_RUNS = re.compile(r"^(w10_mg\d+_[A-Z0-9]+_[a-z_]+|hrl_v\d+)$")


def api_snapshot(run_ids):
    """Ask the dashboard backend itself (in-process, no server) and save its answers, so the online
    copy reads exactly what the local dashboard shows. Local paths are cut down to repo paths."""
    sys.path.insert(0, str(ROOT / "webapp" / "backend"))
    from fastapi.testclient import TestClient
    import main as backend
    c = TestClient(backend.app)
    out = DATA / "api"
    shutil.rmtree(out, ignore_errors=True)
    roots = [str(ROOT) + os.sep, str(ROOT).replace("\\", "/") + "/"]

    def save(path, obj):
        text = json.dumps(obj, separators=(",", ":"))
        for r in roots:
            text = text.replace(json.dumps(r)[1:-1], "")
        f = DATA / (path + ".json")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")

    def get(path):
        r = c.get(path)
        r.raise_for_status()
        return r.json()

    save("api/config", get("/api/config"))
    save("api/drone/status", get("/api/drone/status"))
    runs = get("/api/runs")
    keep = [r for r in runs["runs"] if r["run_id"] in run_ids or SNAPSHOT_RUNS.match(r["run_id"])]
    for r in keep:                      # nothing is training in a snapshot
        if r.get("status") == "training":
            r["status"] = "completed"
    runs.update(runs=keep, current_run_id=None,
                capacity=dict(runs["capacity"] or {}, live_runs=0) if runs.get("capacity") else None)
    runs["best_run_ids"] = {k: v for k, v in (runs.get("best_run_ids") or {}).items() if any(x["run_id"] == v for x in keep)}
    if runs.get("best_run_id") not in [x["run_id"] for x in keep]:
        runs["best_run_id"] = None
    save("api/runs", runs)
    for r in keep:
        rid = r["run_id"]
        one = get(f"/api/runs/{rid}")
        if one.get("status") == "training":
            one["status"] = "completed"
        if (one.get("live_status") or {}).get("status") == "training":
            one["live_status"]["status"] = "completed"
        save(f"api/runs/{rid}", one)
        try:
            save(f"api/runs/{rid}/config", get(f"/api/runs/{rid}/config"))
        except Exception as e:          # an old run without a readable config: its tab says so
            print(f"[site]   no config for {rid}: {e}")
    return [r["run_id"] for r in keep]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="re-record the 3D replays (slow: needs the simulator)")
    ap.add_argument("--only", nargs="*", default=None, help="with --record: just these milestone ids")
    a = ap.parse_args(argv)
    from onboard.sim import CAGE_XY, CAGE_Z

    board = leaderboard()
    write("leaderboard.json", board)
    write("curves.json", curves())
    write("optimiser.json", optimiser())
    write("hrl.json", _json(ROOT / "results" / "hoop" / "hrl_compare.json", []))
    write("timeline.json", timeline())
    ms = []
    for mid, label, what, path in MILESTONES:
        if not (ROOT / path).is_file():
            print(f"[site] skip {mid}: no {path}")
            continue
        if a.record and (a.only is None or mid in a.only):
            print(f"[site] recording {mid} ...", flush=True)
            print(f"[site]   passes per episode: {record(mid, path)}", flush=True)
        if (DATA / "replays" / f"{mid}.json").is_file():
            ms.append(dict(id=mid, label=label, what=what, model=path))
    write("milestones.json", ms)
    best = board[0] if board else None
    write("meta.json", dict(built=datetime.now().isoformat(timespec="minutes"), cage_xy=CAGE_XY, cage_z=list(CAGE_Z),
                            repo="https://github.com/MichelleGun/CMS", best=best))
    snap = api_snapshot({run for _, run in CURVE_RUNS} | {Path(m["model"]).parent.name for m in ms})
    print(f"[site] built {DATA} ({len(board)} policies, {len(ms)} replays, {len(snap)} runs in the API snapshot)")


if __name__ == "__main__":
    main()
