"""
Build a virtual basketball court inside a running PyBullet world.

Two public helpers:
    make_court_texture(png_path, ...)   -> paints a top-down court image (PIL)
    build_court(client, scale=0.5, ...) -> adds floor + 2 hoops to the sim,
                                           returns a dict of body ids + dimensions

The court markings are baked into the floor texture; the 3D scene only adds the
textured floor slab and the two hoop assemblies (pole + backboard + rim).

Real FIBA court is 28 x 15 m with a 3.05 m rim. `scale` shrinks everything
uniformly; the default 0.5 gives a 14 x 7.5 m court with the rim at ~1.5 m, which
sits nicely just above the Crazyflie's 1 m hover.
"""
from pathlib import Path

import numpy as np
import pybullet as p

# --- real court dimensions (metres), scaled by `scale` in build_court ----------
FULL_LEN, FULL_WID = 28.0, 15.0
RIM_HEIGHT = 3.05
RIM_RADIUS = 0.23
BACKBOARD_W, BACKBOARD_H = 1.8, 1.05
RIM_FROM_BASELINE = 1.2          # rim centre this far inside the baseline


# ----------------------------------------------------------------------------- #
# texture
# ----------------------------------------------------------------------------- #
def make_court_texture(png_path, px_per_m=54):
    """Draw a top-down basketball court and save it as a PNG."""
    from PIL import Image, ImageDraw

    W, H = int(FULL_LEN * px_per_m), int(FULL_WID * px_per_m)
    img = Image.new("RGB", (W, H), (198, 140, 78))      # hardwood
    d = ImageDraw.Draw(img)

    def m2px(x, y):
        # court coords: origin at centre, +x toward one hoop, +y toward a sideline
        return int((x + FULL_LEN / 2) * px_per_m), int((y + FULL_WID / 2) * px_per_m)

    white = (255, 255, 255)
    paint = (168, 74, 44)
    lw = max(4, px_per_m // 7)

    # subtle plank lines
    for gx in range(0, W, px_per_m * 2):
        d.line([(gx, 0), (gx, H)], fill=(190, 140, 84), width=1)

    # boundary + halfway line
    d.rectangle([m2px(-FULL_LEN / 2, -FULL_WID / 2), m2px(FULL_LEN / 2, FULL_WID / 2)],
                outline=white, width=lw)
    d.line([m2px(0, -FULL_WID / 2), m2px(0, FULL_WID / 2)], fill=white, width=lw)

    # centre circle
    r = 1.8 * px_per_m
    cx, cy = m2px(0, 0)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=white, width=lw)

    key_len, key_wid = 5.8, 4.9
    three_r = 6.75
    for sign in (-1, 1):
        base_x = sign * FULL_LEN / 2
        rim_x = base_x - sign * RIM_FROM_BASELINE

        # painted key
        x0, x1 = sorted([base_x, base_x - sign * key_len])
        d.rectangle([m2px(x0, -key_wid / 2), m2px(x1, key_wid / 2)], fill=paint)
        d.rectangle([m2px(x0, -key_wid / 2), m2px(x1, key_wid / 2)], outline=white, width=lw)

        # free-throw circle
        ftx, fty = m2px(base_x - sign * key_len, 0)
        fr = 1.8 * px_per_m
        d.ellipse([ftx - fr, fty - fr, ftx + fr, fty + fr], outline=white, width=lw)

        # three-point arc (semicircle around the rim, clipped to the court)
        rr = int(three_r * px_per_m)
        rcx, rcy = m2px(rim_x, 0)
        if sign > 0:
            d.arc([rcx - rr, rcy - rr, rcx + rr, rcy + rr], 90, 270, fill=white, width=lw)
        else:
            d.arc([rcx - rr, rcy - rr, rcx + rr, rcy + rr], -90, 90, fill=white, width=lw)

        # rim + backboard marks
        rimpx = int(RIM_RADIUS * px_per_m)
        d.ellipse([rcx - rimpx, rcy - rimpx, rcx + rimpx, rcy + rimpx],
                  outline=(230, 120, 40), width=lw)

    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)
    return png_path


# ----------------------------------------------------------------------------- #
# 3D scene
# ----------------------------------------------------------------------------- #
def _hoop(client, rim_x, face_sign, scale, rim_h):
    """One pole + backboard + ring of markers, facing toward court centre."""
    rim_r = RIM_RADIUS * scale
    bb_w, bb_h = BACKBOARD_W * scale, BACKBOARD_H * scale
    base_x = rim_x + face_sign * RIM_FROM_BASELINE * scale
    pole_x = base_x + face_sign * 0.25 * scale
    ids = []

    # pole
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05 * scale, height=rim_h,
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05 * scale, length=rim_h,
                              rgbaColor=[0.15, 0.15, 0.15, 1], physicsClientId=client)
    ids.append(p.createMultiBody(0, col, vis, [pole_x, 0, rim_h / 2],
                                 physicsClientId=client))

    # backboard
    bb = p.createVisualShape(p.GEOM_BOX,
                             halfExtents=[0.02 * scale, bb_w / 2, bb_h / 2],
                             rgbaColor=[1, 1, 1, 0.85], physicsClientId=client)
    bbc = p.createCollisionShape(p.GEOM_BOX,
                                 halfExtents=[0.02 * scale, bb_w / 2, bb_h / 2],
                                 physicsClientId=client)
    ids.append(p.createMultiBody(0, bbc, bb, [base_x, 0, rim_h + 0.15 * scale],
                                 physicsClientId=client))

    # rim: a ring of small orange spheres
    marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.02 * scale,
                                 rgbaColor=[0.95, 0.45, 0.1, 1], physicsClientId=client)
    for a in np.linspace(0, 2 * np.pi, 18, endpoint=False):
        ids.append(p.createMultiBody(
            0, -1, marker,
            [rim_x + np.cos(a) * rim_r, np.sin(a) * rim_r, rim_h],
            physicsClientId=client))
    return ids


def build_court(client, scale=0.5, texture_png=None):
    """Add the court to an existing PyBullet client. Returns geometry info."""
    L, Wd = FULL_LEN * scale, FULL_WID * scale
    rim_h = RIM_HEIGHT * scale
    rim_x = (L / 2 - RIM_FROM_BASELINE * scale)

    if texture_png is None:
        texture_png = str(Path(__file__).resolve().parent.parent / "assets" / "court_texture.png")
    if not Path(texture_png).is_file():
        make_court_texture(texture_png)

    # floor slab
    floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L / 2, Wd / 2, 0.01],
                                       physicsClientId=client)
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L / 2, Wd / 2, 0.01],
                                    rgbaColor=[1, 1, 1, 1], physicsClientId=client)
    floor = p.createMultiBody(0, floor_col, floor_vis, [0, 0, -0.01],
                              physicsClientId=client)
    try:
        tex = p.loadTexture(texture_png, physicsClientId=client)
        p.changeVisualShape(floor, -1, textureUniqueId=tex, physicsClientId=client)
    except Exception as e:                       # texture is cosmetic only
        print(f"[court] texture skipped: {e}")

    hoops = {
        "right": _hoop(client, rim_x, -1, scale, rim_h),
        "left": _hoop(client, -rim_x, +1, scale, rim_h),
    }

    return {
        "floor_id": floor,
        "hoops": hoops,
        "length": L, "width": Wd,
        "rim_height": rim_h,
        "rim_x": rim_x,          # right hoop at (+rim_x, 0, rim_h); left at (-rim_x, 0, rim_h)
        "scale": scale,
    }
