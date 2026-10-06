"""
Run a trained policy (from the sim, or a behaviour-cloned one) on the Tello, or on FakeTello with --dry-run.

    python hardware/control/run_policy.py --run-dir results/hoop/runs/<run> --dry-run --headless --hoop 0.4 -0.3 1.0
    python hardware/control/run_policy.py --run-dir results/hoop/runs/<run> --hoop 0.4 -0.3 1.0 --speed-scale 0.5
    python hardware/control/run_policy.py --model hardware/flights/bc/my_bc/bc_model.zip --algo ppo --dry-run

A run dir is the sim's results/hoop/runs/<algo>_<timestamp>/ (best_model.zip / final_model.zip +
meta.json), so models trained in the dashboard load directly. The policy sees exactly the
observation it was trained on (see sim_check/obs_parity.py); the only differences are the state
source (camera + Tello fusion instead of simulator ground truth) and the real drone's dynamics.

Safety: the policy never talks to the drone directly. Its action goes through the geofence, the speed
ramp (--speed-scale), the safety monitor, and the same keyboard kill/land/hold keys as teleop.
Pressing any movement key takes over manually for that moment; --record keeps those corrections as
demonstrations (DAgger-style).
Reaching the hoop (within 0.15 m) = hover 1 s, then land by itself.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_setup import HW, add_common_args, build, confirm_hoop, resolve_hoop    # noqa: E402

from common.action_mapper import discrete_to_action          # noqa: E402
from common.flight_session import FlightSession              # noqa: E402
from common.logger import FlightLogger                        # noqa: E402
from common.safety import GlobalKillSwitch                     # noqa: E402
from control.inputs import NullInput, PygameInput              # noqa: E402
from control.loop import drive                                 # noqa: E402
from imitation.record_demo import DemoRecorder                 # noqa: E402

ALGOS = ("ppo", "a2c", "sac", "td3", "dqn")


def load_policy(run_dir=None, model_path=None, variant="best", algo=None):
    """Returns (policy_fn(obs)->action(4,), description)."""
    import stable_baselines3 as sb3
    discrete = False
    if run_dir is not None:
        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").is_file() else {}
        algo = meta.get("algo", algo or "ppo")
        discrete = meta.get("env_variant") == "discrete_vel"
        path = run_dir / f"{variant}_model.zip"
        if not path.is_file():
            path = run_dir / "final_model.zip" if (run_dir / "final_model.zip").is_file() else run_dir / "best_model.zip"
        status = meta.get("final_status")
        if status == "crashed":
            sys.exit(f"[policy] {run_dir.name} has final_status 'crashed'; pick another run")
    else:
        path = Path(model_path)
        algo = algo or "ppo"
        mj = path.parent / "meta.json"
        if mj.is_file():
            meta = json.loads(mj.read_text())
            algo = meta.get("algo", algo)
            discrete = meta.get("env_variant") == "discrete_vel"
    if not path.is_file():
        sys.exit(f"[policy] model not found: {path}")
    cls = getattr(sb3, algo.upper())
    model = cls.load(str(path), device="cpu")
    shape = model.observation_space.shape
    if shape != (1, 75):
        sys.exit(f"[policy] {path.name} expects observations of shape {shape}, but this robot builds (1, 75). "
                 "Was it trained on HoopFindAviary with KIN obs and VEL actions?")

    def policy_fn(obs):
        act, _ = model.predict(obs, deterministic=True)
        if discrete:
            return discrete_to_action(int(np.asarray(act).reshape(-1)[0]))
        return np.asarray(act, dtype=float).reshape(-1)[:4]
    return policy_fn, f"{algo.upper()} {path}"


def main(argv=None, inputs=None, realtime=None, policy_fn=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    src = ap.add_mutually_exclusive_group(required=policy_fn is None)
    src.add_argument("--run-dir", type=Path, help="results/hoop/runs/<run>")
    src.add_argument("--model", type=Path, help="a .zip model (meta.json next to it is used if present)")
    ap.add_argument("--variant", choices=["best", "final"], default="best")
    ap.add_argument("--algo", choices=ALGOS, default=None)
    ap.add_argument("--headless", action="store_true", help="no window; takes off by itself (dry-run only)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--record", action="store_true", help="record human corrections as demonstrations")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    if a.headless and not a.dry_run:
        sys.exit("--headless is only for --dry-run: a real flight needs the window with the kill keys")
    if not 0.0 < a.speed_scale <= 1.0:
        sys.exit("--speed-scale must be in (0, 1]")

    desc = "custom policy"
    if policy_fn is None:
        policy_fn, desc = load_policy(a.run_dir, a.model, a.variant, a.algo)
    print(f"[policy] {desc}")
    cfg, backend, camera = build(a)
    hoop = resolve_hoop(a, cfg, required=True)
    confirm_hoop(hoop, a)

    logger = FlightLogger("policy", enabled=not a.no_log)
    session = FlightSession(cfg, backend, camera, hoop=hoop, logger=logger, speed_scale=a.speed_scale)
    rec = DemoRecorder({"source": "dagger", "hoop": hoop.tolist(), "dry_run": a.dry_run}) if a.record else None
    window = None
    if inputs is None:
        inputs = NullInput() if a.headless else PygameInput("Tello policy run")
        window = None if a.headless else inputs
    kill = GlobalKillSwitch(session.monitor)
    if not a.no_hotkeys and not a.headless:
        kill.start()
    rt = (not a.headless) if realtime is None else realtime
    summary = {}
    try:
        summary = drive(session, inputs, policy_fn=policy_fn, recorder=rec, realtime=rt, max_seconds=a.max_seconds,
                        auto_takeoff=a.headless, window=window)
    except KeyboardInterrupt:
        print("\n[policy] Ctrl+C: landing")
        session.request_land("ctrl_c")
        try:
            drive(session, NullInput(), realtime=rt, max_seconds=0.0)
        except KeyboardInterrupt:
            print("[policy] second Ctrl+C: leaving. The Tello lands by itself ~15 s after commands stop.")
    finally:
        kill.stop()
        session.close()
        if rec is not None and len(rec):
            print(f"[policy] saved {len(rec)} correction samples -> {rec.save()}")
        try:
            inputs.close()
            backend.close()
            if hasattr(camera, "close"):
                camera.close()
        except Exception:
            pass
    if summary:
        print(f"[policy] finished: phase={summary['phase']} reached_hoop={summary['reached']} "
              f"ticks={summary['ticks']}" + (f" log={logger.path}" if logger.path else ""))
    return summary, rec, session


if __name__ == "__main__":
    main()
