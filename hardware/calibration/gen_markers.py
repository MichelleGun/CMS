"""
Generate printable ArUco markers for the cage.

    python hardware/calibration/gen_markers.py            # writes hardware/calibration/markers/*.png

Print at 100 % scale (no 'fit to page'). What goes where (ids come from config.yaml `aruco`):
  reference markers (ids 0-3)  flat on the floor at the four cage positions in config `aruco.reference`
                               (default: the corners at (+-1, +-1) m). Centre of each marker on the spot.
  drone marker (id 10)         on TOP of the Tello, with the marker's TOP edge pointing at the drone's NOSE
                               (the camera is at the front). Keep it flat; cut to about 7 x 7 cm.
  hoop marker (id 20)          flat on the floor directly under the hoop centre (calibration/hoop_locator.py).
Matte paper, white border at least one marker-module wide, no glare under the room lights.
"""
import sys
from pathlib import Path

import numpy as np

HW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HW))
from common.config import load_effective_config      # noqa: E402

OUT = HW / "calibration" / "markers"


def render(aruco, dictionary, marker_id, px=600, border=100):
    if hasattr(aruco, "generateImageMarker"):
        img = aruco.generateImageMarker(dictionary, marker_id, px)
    else:                                                       # OpenCV < 4.7
        img = aruco.drawMarker(dictionary, marker_id, px)
    out = np.full((px + 2 * border, px + 2 * border), 255, dtype=np.uint8)
    out[border:border + px, border:border + px] = img
    return out


def main():
    try:
        import cv2
    except ImportError:
        sys.exit("opencv is needed: pip install opencv-contrib-python")
    cfg = load_effective_config()
    a = cfg["aruco"]
    aruco = cv2.aruco
    d = aruco.getPredefinedDictionary(getattr(aruco, a["dictionary"]))
    OUT.mkdir(parents=True, exist_ok=True)
    names = {int(i): f"reference_{i}" for i in a["reference"]}
    names[int(a["drone_id"])] = "drone_top"
    names[int(a["hoop_id"])] = "hoop_floor"
    for mid, name in sorted(names.items()):
        img = render(aruco, d, mid)
        cv2.putText(img, f"id {mid}  {name}", (110, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
        cv2.imwrite(str(OUT / f"{mid:02d}_{name}.png"), img)
        print("wrote", OUT / f"{mid:02d}_{name}.png")
    print("\nPrint at 100 % scale. Drone marker ~7 cm, floor markers 15-20 cm. See this file's docstring for placement.")


if __name__ == "__main__":
    main()
