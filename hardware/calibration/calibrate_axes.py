"""
Find out how YOUR Tello's rc channels, velocity telemetry and yaw reading relate to the cage frame,
instead of trusting the assumed conventions in config.yaml.

    python hardware/calibration/calibrate_axes.py --dry-run                    # rehearse with FakeTello (writes nothing)
    python hardware/calibration/calibrate_axes.py --dry-run --fake-axis-flip   # see a mismatch being detected
    python hardware/calibration/calibrate_axes.py                              # real drone (needs the overhead camera)

The drone takes off, then makes eight small hops (forward/back, left/right, up/down, yaw both ways) with
RAW rc values, so the action mapper's assumptions are not in the loop. The overhead camera (and Tello's
own height reading for up/down) tell us what each channel really does. Result: corrected
`conventions.rc_signs / vel_signs / att_signs / vel_to_mps`, written to hardware/config.local.yaml with
`axes_calibrated: true` only if every channel was resolved. Until that flag is true, real takeoff is refused
by the rest of the toolchain.

Before a real run: props guards on, clear 2x2 m cage, someone with a hand on the X key
(or Ctrl+Shift+X anywhere). Hops are ~20-30 cm.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_flight import CalibFlight                                  # noqa: E402

from calibration.analysis import analyze_axes                         # noqa: E402
from common.config import save_config_values                          # noqa: E402
from control.session_setup import add_common_args, build              # noqa: E402


def run_trials(cf: CalibFlight, rc_mag=30, seconds=1.2):
    trials = {}
    plan = [("fb", (0, rc_mag, 0, 0)), ("lr", (rc_mag, 0, 0, 0)),
            ("ud", (0, 0, rc_mag, 0)), ("yaw", (0, 0, 0, rc_mag + 10))]
    for name, rc in plan:
        cf.log(f"[axes] hop '{name}' rc={rc}")
        trials[name] = cf.run(rc, seconds if name != "ud" else 0.8, name)
        if cf.aborted:
            cf.log("[axes] a safety fault interrupted the calibration")
            return trials, False
        cf.log(f"[axes] and back")
        cf.run(tuple(-c for c in rc), seconds if name != "ud" else 0.8, name + "_back")
        if cf.aborted:
            return trials, False
    return trials, True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--rc", type=int, default=30)
    ap.add_argument("--yes", action="store_true", help="do not ask before writing / flying")
    ap.add_argument("--write", action="store_true", help="write results even with --dry-run (testing only)")
    ap.add_argument("--config-out", type=Path, default=None)
    a = ap.parse_args(argv)

    if not a.dry_run and not a.yes:
        print("About to FLY the drone through eight small hops. Props guards on, cage clear, kill key ready.")
        if input("Type 'fly' to continue: ").strip().lower() != "fly":
            sys.exit("aborted")
    cfg, backend, camera = build(a)
    cf = CalibFlight(cfg, backend, camera, hotkeys=not a.dry_run and not a.no_hotkeys)
    result = None
    try:
        if not cf.wait_ready():
            sys.exit("[axes] estimator never became ready (is the drone marker visible to the overhead camera?)")
        if not cf.takeoff():
            sys.exit("[axes] takeoff refused or failed")
        for _ in range(60):
            cf.tick()
        trials, finished = run_trials(cf, a.rc)
        result = analyze_axes(trials, cfg["conventions"])
    finally:
        if cf.session.phase in ("flying", "warmup", "takeoff"):
            cf.land()
        cf.close()
        try:
            backend.close()
            if hasattr(camera, "close"):
                camera.close()
        except Exception:
            pass

    print("\n[axes] results")
    for n in result.notes:
        print("  -", n)
    for p in result.problems:
        print("  ! UNRESOLVED:", p)
    conv = dict(result.conventions)
    print("  conventions ->", conv)
    if result.ok:
        conv["axes_calibrated"] = True
        if (not a.dry_run) or a.write:
            if a.yes or input("Write these to config.local.yaml? [y/N] ").strip().lower() == "y":
                path = save_config_values({"conventions": conv}, a.config_out)
                print(f"[axes] wrote {path}")
        else:
            print("[axes] dry-run: nothing written (use --write to test the write path)")
    else:
        print("[axes] NOT marking axes_calibrated: fix the problems above and re-run.")
    return result


if __name__ == "__main__":
    main()
