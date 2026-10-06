"""Argument parsing and object construction shared by the control and calibration scripts."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
for p in (HW, HW.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import constants as C                      # noqa: E402
from common.config import FLIGHTS_DIR, load_effective_config   # noqa: E402

HOOP_FILE = FLIGHTS_DIR / "hoop.json"


def add_common_args(ap):
    g = ap.add_argument_group("backend")
    g.add_argument("--dry-run", action="store_true",
                   help="no drone: FakeTello + fake overhead camera (props never spin; safe anywhere)")
    g.add_argument("--profile", default="typical", choices=["ideal", "typical", "harsh"],
                   help="FakeTello realism preset (dry-run only)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--fake-axis-flip", action="store_true",
                   help="dry-run only: emulate a drone whose left/right is mirrored relative to the config "
                        "(shows what the axis calibration / runtime axis check catch)")
    g.add_argument("--camera-index", type=int, default=None)
    g.add_argument("--speed-scale", type=float, default=0.5,
                   help="scales the commanded speed fraction (0-1). Keep low for first flights")
    g.add_argument("--hoop", type=float, nargs=3, metavar=("X", "Y", "Z"), help="hoop position in the cage frame (m)")
    g.add_argument("--force", action="store_true", help="allow a hoop outside the range the policy was trained on")
    g.add_argument("--no-log", action="store_true", help="do not write flights/*.jsonl")
    g.add_argument("--no-hotkeys", action="store_true", help="do not register the global kill hotkeys")
    return ap


def check_hoop_range(hoop, force=False):
    h = np.asarray(hoop, dtype=float)
    lim = C.HOOP_XY_MAX
    problems = []
    if np.any(np.abs(h[:2]) > lim + 1e-9):
        problems.append(f"hoop xy {h[:2].round(2).tolist()} is outside +-{lim.tolist()} m, which the policy never saw in training")
    if not (C.HOOP_Z_RANGE[0] - 1e-9 <= h[2] <= C.HOOP_Z_RANGE[1] + 1e-9):
        problems.append(f"hoop z {h[2]:.2f} is outside {C.HOOP_Z_RANGE}")
    if problems and not force:
        sys.exit("[hoop] " + "; ".join(problems) + "  (use --force to override)")
    return h


def resolve_hoop(args, cfg, required=False, log=print):
    if args.hoop:
        return check_hoop_range(args.hoop, args.force)
    if HOOP_FILE.exists():
        d = json.loads(HOOP_FILE.read_text())
        age = time.time() - d["unix_time"]
        if age <= cfg["hoop"]["max_age_s"]:
            log(f"[hoop] using {HOOP_FILE.name} ({age:.0f} s old): {d['xyz']}")
            return check_hoop_range(d["xyz"], args.force)
        log(f"[hoop] {HOOP_FILE.name} is {age:.0f} s old (limit {cfg['hoop']['max_age_s']}); ignoring it")
    if required:
        sys.exit("[hoop] no hoop position: pass --hoop X Y Z or run calibration/hoop_locator.py")
    return None


def build(args, log=print):
    """Returns (cfg, backend, camera). Connects the backend."""
    cfg = load_effective_config()
    if args.dry_run:
        from common.fake_tello import FakeOverheadCamera, FakeTello
        backend = FakeTello(cfg, args.profile, seed=args.seed, axis_flip=getattr(args, 'fake_axis_flip', False))
        camera = FakeOverheadCamera(backend, seed=args.seed + 1)
        log(f"[dry-run] FakeTello profile '{args.profile}', no hardware is touched")
    else:
        if not cfg["conventions"].get("axes_calibrated", False):
            log("[warning] conventions.axes_calibrated is false: takeoff will be refused. "
                "Run calibration/calibrate_axes.py first (dry-run flights are unaffected).")
        from common.tello_link import TelloLink
        from state_estimation.aruco_tracker import OverheadCamera
        backend = TelloLink(cfg)
        camera = OverheadCamera(cfg, HW, args.camera_index)
    backend.connect()
    return cfg, backend, camera


def confirm_hoop(hoop, args, log=print):
    log(f"[hoop] target: x={hoop[0]:+.2f} y={hoop[1]:+.2f} z={hoop[2]:+.2f} m")
    if args.dry_run or getattr(args, "yes", False):
        return
    if input("Is the hoop really there? Type y to continue: ").strip().lower() != "y":
        sys.exit("aborted")
