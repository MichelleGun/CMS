"""
Measure how fast the drone really moves for a given rc value, and how quickly it responds.

    python hardware/calibration/calibrate_speed.py --dry-run
    python hardware/calibration/calibrate_speed.py                 # real drone; run calibrate_axes.py first

The simulator's action 'speed fraction 1.0' means exactly 0.25 m/s. On the Tello that only holds if
`control.rc_per_mps` is right: the video this project started from found rc 15 gave ~0.117 m/s, i.e. the
nominal numbers are not reliable. This script flies three forward/back passes at different rc levels,
measures the speed with the overhead camera, fits `rc = k * speed`, and also reports the response time
constant and the camera latency the filter's online tracker converged to.
Writes control.rc_per_mps (and estimator.cam_latency_s if the tracker is confident) to config.local.yaml.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_flight import CalibFlight                                  # noqa: E402

from calibration.analysis import fit_rc_per_mps, steady_speed_and_tau  # noqa: E402
from common.config import save_config_values                          # noqa: E402
from control.session_setup import add_common_args, build              # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--levels", type=int, nargs="+", default=[15, 25, 35])
    ap.add_argument("--seconds", type=float, default=2.2)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--config-out", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.dry_run and not a.yes:
        if input("The drone will fly three short forward/back passes. Cage clear, kill key ready? Type 'fly': ").strip().lower() != "fly":
            sys.exit("aborted")
    cfg, backend, camera = build(a)
    cf = CalibFlight(cfg, backend, camera, hotkeys=not a.dry_run and not a.no_hotkeys)
    levels, speeds, taus = [], [], []
    latency = sigma = None
    try:
        if not cf.wait_ready() or not cf.takeoff():
            sys.exit("[speed] could not take off")
        for _ in range(60):
            cf.tick()
        sign = 1
        for rc_level in a.levels:
            tr = cf.run((0, sign * rc_level, 0, 0), a.seconds, f"fb{rc_level}")
            if cf.aborted:
                break
            v, tau = steady_speed_and_tau(tr)
            if v:
                levels.append(rc_level)
                speeds.append(v)
                if tau:
                    taus.append(tau)
                print(f"[speed] rc {rc_level:3d} -> {v:.3f} m/s" + (f", 63 % in {tau:.2f} s" if tau else ""))
            cf.run((0, -sign * rc_level, 0, 0), a.seconds, "back")        # return toward the centre
            sign = -sign
        lt = cf.session.est_ctor.latency
        if len(lt._raw) >= 3:
            latency, sigma = lt.latency, lt.sigma
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

    if len(levels) < 2:
        sys.exit("[speed] not enough valid passes to fit")
    k = fit_rc_per_mps(levels, speeds)
    print(f"\n[speed] rc_per_mps = {k:.1f}   (config currently {cfg['control']['rc_per_mps']})")
    print(f"        => sim's 0.25 m/s needs rc {0.25 * k:.0f}; rc_max is {cfg['control']['rc_max']}")
    if taus:
        print(f"        response time constant ~ {np.median(taus):.2f} s (put into a FakeTello profile for realistic dry-runs)")
    updates = {"control": {"rc_per_mps": round(k, 1)}}
    if latency is not None:
        print(f"        camera latency tracker: {latency * 1000:.0f} ms (jitter {sigma * 1000:.0f} ms)")
        updates["estimator"] = {"cam_latency_s": round(latency, 3)}
    if 0.25 * k > cfg["control"]["rc_max"]:
        print("  ! the sim's full speed needs more rc than rc_max allows: raise control.rc_max deliberately, or keep --speed-scale < 1")
    if (not a.dry_run) or a.write:
        if a.yes or input("Write to config.local.yaml? [y/N] ").strip().lower() == "y":
            print(f"[speed] wrote {save_config_values(updates, a.config_out)}")
    else:
        print("[speed] dry-run: nothing written")
    return k, taus, latency


if __name__ == "__main__":
    main()
