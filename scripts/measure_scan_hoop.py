"""
Measure the blue noodle ring in the phone scan (assets/3D_net&hoop.glb), in the sim frame.

The scan is placed exactly the way the dashboard's Environment tab places it (webapp/frontend/js/
envview.js fitScan(), sliders at their defaults): its footprint is scaled so it matches the sim net,
squared to the net's axes, centred, and its floor put at z = 0. The blue texels of that mesh are
then fitted with a plane (-> the ring's facing / lean) and a circle in that plane (-> its centre and
radius). The numbers printed at the end are what HoopSpec in hoop_net_env.py should use.

    python scripts/measure_scan_hoop.py

Needs only numpy + Pillow (no trimesh): a GLB is a JSON header plus one binary buffer.
"""
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CMS_DIR = Path(__file__).resolve().parent.parent
GLB = CMS_DIR / "assets" / "3D_net&hoop.glb"
ROOM = (2.0, 2.0)       # sim net footprint, as hoop_net_env.ROOM[:2]

_DTYPE = {5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}


def load_glb(path):
    """Positions (N, 3), UVs (N, 2) and the base-colour texture (H, W, 3 uint8) of the single mesh."""
    data = path.read_bytes()
    jlen = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + jlen])
    bin_start = 20 + jlen + 8                 # skip the BIN chunk's length + type
    blob = data[bin_start:]

    def view(i):
        bv = gltf["bufferViews"][i]
        off = bv.get("byteOffset", 0)
        return blob[off:off + bv["byteLength"]]

    def accessor(i):
        a = gltf["accessors"][i]
        arr = np.frombuffer(view(a["bufferView"]), dtype=_DTYPE[a["componentType"]],
                            offset=a.get("byteOffset", 0), count=a["count"] * _WIDTH[a["type"]])
        return arr.reshape(a["count"], _WIDTH[a["type"]]).astype(float)

    if len(gltf["meshes"]) != 1 or any(k in n for n in gltf["nodes"] for k in ("matrix", "rotation", "scale",
                                                                                "translation")):
        sys.exit("unexpected GLB layout (several meshes or node transforms); update load_glb()")
    prim = gltf["meshes"][0]["primitives"][0]
    pos = accessor(prim["attributes"]["POSITION"])
    uv = accessor(prim["attributes"]["TEXCOORD_0"])
    mat = gltf["materials"][prim["material"]]
    tex = gltf["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    img = Image.open(io.BytesIO(view(gltf["images"][tex["source"]]["bufferView"]))).convert("RGB")
    return pos, uv, np.asarray(img)


def pct(sorted_vals, q):
    """Same percentile as envview.js pct(): index floor(q * (n - 1)) of a sorted array."""
    return sorted_vals[min(len(sorted_vals) - 1, max(0, int(np.floor(q * (len(sorted_vals) - 1)))))]


def fit_scan(pos):
    """Port of envview.js fitScan(): returns a function mapping scan (three.js, y-up) points to sim (z-up)."""
    step = max(1, len(pos) // 15000)
    pts = pos[::step]
    ys = np.sort(pts[:, 1])
    floor = pct(ys, 0.02)
    best = None
    for deg in np.arange(0, 90, 0.5):
        r = np.radians(deg)
        c, s = np.cos(r), np.sin(r)
        us = np.sort(c * pts[:, 0] + s * pts[:, 2])
        ws = np.sort(-s * pts[:, 0] + c * pts[:, 2])
        lu, hu, lw, hw = pct(us, 0.02), pct(us, 0.98), pct(ws, 0.02), pct(ws, 0.98)
        area = (hu - lu) * (hw - lw)
        if best is None or area < best["area"]:
            best = dict(area=area, r=r, c=c, s=s, w=hu - lu, d=hw - lw, uc=(lu + hu) / 2, wc=(lw + hw) / 2)
    cx = best["c"] * best["uc"] - best["s"] * best["wc"]
    cz = best["s"] * best["uc"] + best["c"] * best["wc"]
    scale = (sum(ROOM) / 2) / ((best["w"] + best["d"]) / 2)
    r = best["r"]

    def to_sim(p):
        x, y, z = p[:, 0] - cx, p[:, 1] - floor, p[:, 2] - cz
        # fitGroup.rotation.y = r: (x, z) -> (x cos r + z sin r, -x sin r + z cos r), then uniform scale
        xr = (x * np.cos(r) + z * np.sin(r)) * scale
        zr = (-x * np.sin(r) + z * np.cos(r)) * scale
        return np.column_stack([xr, zr, y * scale])          # three (x, y-up, z) -> sim (x, y, z-up)

    print(f"scan footprint {best['w']:.2f} x {best['d']:.2f} m, turned {np.degrees(r):.1f} deg, scale x{scale:.3f}")
    return to_sim


def blue_mask(rgb):
    """Saturated blue: hue 195-255 deg, saturation > 0.35, value > 0.2."""
    f = rgb.astype(float) / 255.0
    mx, mn = f.max(1), f.min(1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
    r, g, b = f[:, 0], f[:, 1], f[:, 2]
    d = np.maximum(mx - mn, 1e-9)
    hue = np.where(mx == b, 240 + 60 * (r - g) / d, np.where(mx == g, 120 + 60 * (b - r) / d, 60 * (g - b) / d))
    hue = np.mod(hue, 360)
    return (hue > 195) & (hue < 255) & (sat > 0.35) & (mx > 0.2)


def fit_plane(p):
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c)
    return c, vt[2]


def fit_circle(p, centre, n):
    """Algebraic (Kasa) circle fit in the plane through `centre` with normal `n`."""
    u = np.cross(n, [0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])
    v = np.cross(n, u)
    q = p - centre
    a, b = q @ u, q @ v
    A = np.column_stack([a, b, np.ones_like(a)])
    sol, *_ = np.linalg.lstsq(A, a ** 2 + b ** 2, rcond=None)
    ca, cb = sol[0] / 2, sol[1] / 2
    rad = np.sqrt(sol[2] + ca ** 2 + cb ** 2)
    resid = np.abs(np.hypot(a - ca, b - cb) - rad)
    return centre + ca * u + cb * v, rad, resid


def main():
    pos, uv, img = load_glb(GLB)
    h, w = img.shape[:2]
    px = np.clip((uv[:, 0] % 1.0) * w, 0, w - 1).astype(int)
    py = np.clip((uv[:, 1] % 1.0) * h, 0, h - 1).astype(int)    # glTF UV origin is the top-left
    rgb = img[py, px]
    sim = fit_scan(pos)(pos)

    pts = sim[blue_mask(rgb) & (sim[:, 2] > 0.5)]
    print(f"{len(pts)} blue vertices above 0.5 m")
    if len(pts) < 50:
        sys.exit("too few blue vertices to fit the ring")
    # keep the main blob: iteratively drop points far from the median
    for radius in (0.6, 0.4, 0.35):
        med = np.median(pts, 0)
        pts = pts[np.linalg.norm(pts - med, axis=1) < radius]
    # plane + circle, then one pass rejecting outliers of the circle fit
    for _ in range(3):
        c, n = fit_plane(pts)
        centre, rad, resid = fit_circle(pts, c, n)
        keep = resid < max(0.04, 2.5 * np.median(resid))
        dist_plane = np.abs((pts - c) @ n)
        keep &= dist_plane < max(0.05, 2.5 * np.median(dist_plane))
        pts = pts[keep]
    c, n = fit_plane(pts)
    centre, rad, resid = fit_circle(pts, c, n)
    if n[0] < 0:                     # report the facing that points along +x, like HoopSpec's nominal
        n = -n
    yaw = np.degrees(np.arctan2(n[1], n[0]))
    tilt = np.degrees(np.arcsin(np.clip(n[2], -1, 1)))
    print(f"{len(pts)} vertices in the fit, circle residual median {np.median(resid) * 100:.1f} cm")
    print(f"ring centre      x={centre[0]:+.3f}  y={centre[1]:+.3f}  z={centre[2]:.3f} m")
    print(f"ring facing      {n.round(3).tolist()}  (yaw {yaw:+.1f} deg, tilt {tilt:+.1f} deg)")
    print(f"tube-centre radius {rad:.3f} m  (expected ~0.18 m: 0.15 opening + 0.03 tube)")


if __name__ == "__main__":
    main()
