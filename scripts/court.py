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
RING_BLUE = [0.12, 0.38, 0.95, 1]  # every simulated hoop: the real ring is a blue pool noodle


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

    # rim: a ring of small blue spheres (the real ring is a blue pool noodle)
    marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.02 * scale,
                                 rgbaColor=RING_BLUE, physicsClientId=client)
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


# ----------------------------------------------------------------------------- #
# hardware-demo cage: a bare 2x2x2 m room with one hoop, moved on every episode
# ----------------------------------------------------------------------------- #
def _hoop_at(client, center, rim_r=0.12, bb_w=0.4, bb_h=0.25, pole_r=0.02, face_dir=(0, 1)):
    """One pole + backboard + ring of rim markers at an absolute (x, y, rim_z).

    `face_dir` is the horizontal direction the open (approach) side of the
    hoop points toward; the backboard is mounted on the opposite side.
    """
    cx, cy, rim_h = center
    fx, fy = np.asarray(face_dir, dtype=float)
    norm = np.hypot(fx, fy)
    fx, fy = (fx / norm, fy / norm) if norm > 1e-6 else (0.0, 1.0)
    ids = []

    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=pole_r, height=rim_h,
                                 physicsClientId=client)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=pole_r, length=rim_h,
                              rgbaColor=[0.15, 0.15, 0.15, 1], physicsClientId=client)
    ids.append(p.createMultiBody(0, col, vis, [cx, cy, rim_h / 2], physicsClientId=client))

    bb_x, bb_y = cx - fx * (rim_r + 0.03), cy - fy * (rim_r + 0.03)
    orn = p.getQuaternionFromEuler([0, 0, np.arctan2(fy, fx)])
    bb = p.createVisualShape(p.GEOM_BOX, halfExtents=[bb_w / 2, 0.02, bb_h / 2],
                             rgbaColor=[1, 1, 1, 0.85], physicsClientId=client)
    bbc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[bb_w / 2, 0.02, bb_h / 2],
                                 physicsClientId=client)
    ids.append(p.createMultiBody(0, bbc, bb, [bb_x, bb_y, rim_h + 0.15 * rim_r / 0.23], orn,
                                 physicsClientId=client))

    marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.015,
                                 rgbaColor=RING_BLUE, physicsClientId=client)
    for a in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        ids.append(p.createMultiBody(
            0, -1, marker,
            [cx + np.cos(a) * rim_r, cy + np.sin(a) * rim_r, rim_h],
            physicsClientId=client))
    return ids


def _draw_room_wireframe(client, room, color=(0.2, 0.6, 1.0)):
    """12-edge wireframe box marking the physical cage bounds (visual only)."""
    rx, ry, rz = np.array(room) / 2
    corners = np.array([[x, y, z] for x in (-rx, rx) for y in (-ry, ry) for z in (0, 2 * rz)])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(corners[i] != corners[j]) == 1:
                p.addUserDebugLine(corners[i], corners[j], lineColorRGB=color, lineWidth=1.5,
                                   physicsClientId=client)


def build_room(client, room=(2.0, 2.0, 2.0), show_wireframe=True):
    """Floor slab + wireframe bounds for the hardware-demo cage. Build once per run."""
    rx, ry, rz = room
    floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[rx / 2, ry / 2, 0.01],
                                       physicsClientId=client)
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[rx / 2, ry / 2, 0.01],
                                    rgbaColor=[0.5, 0.5, 0.55, 1], physicsClientId=client)
    floor = p.createMultiBody(0, floor_col, floor_vis, [0, 0, -0.01], physicsClientId=client)
    if show_wireframe:
        _draw_room_wireframe(client, room)
    return {"floor_id": floor, "room": np.array(room, dtype=float)}


def build_hoop(client, hoop_pos, rim_r=0.12):
    """Single hoop at `hoop_pos = (x, y, rim_height)`. Rebuild every episode to move it."""
    cx, cy, _ = hoop_pos
    face_dir = (cx, cy) if (abs(cx) > 1e-6 or abs(cy) > 1e-6) else (0.0, 1.0)
    return _hoop_at(client, hoop_pos, rim_r=rim_r, face_dir=face_dir)


def build_mini_hoop(client, room=(2.0, 2.0, 2.0), hoop_pos=(0.5, 0.5, 0.6), rim_r=0.12):
    """Convenience: room + one hoop in a single call, for a static (non-episodic) view."""
    info = build_room(client, room=room)
    info["hoop"] = build_hoop(client, hoop_pos, rim_r=rim_r)
    info["hoop_pos"] = np.array(hoop_pos, dtype=float)
    info["rim_radius"] = rim_r
    return info


def build_net(client, room=(2.0, 2.0, 2.0), show_grid=True, spacing=0.25,
              frame_rgba=(0.92, 0.92, 0.95, 1), net_rgba=(0.55, 0.55, 0.6, 1)):
    """Visual-only cage like the real one: white frame pipes on the 12 edges and a grey net grid on
    the four walls and the roof, built from thin bars (not translucent panels: PyBullet's
    TinyRenderer -- the drone camera -- ignores alpha and would draw solid walls).
    No collision shapes: net contact is handled geometrically by HoopNetAviary.
    `show_grid` is kept for API compatibility; the grid is always built so the POV shows it."""
    rx, ry, rz = np.array(room, dtype=float)
    hx, hy = rx / 2, ry / 2
    ids = []

    def bars(half, positions, rgba):
        if not positions:
            return
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=list(rgba), physicsClientId=client)
        out = p.createMultiBody(0, -1, vis, batchPositions=[list(map(float, q)) for q in positions],
                                physicsClientId=client)
        ids.extend(out if isinstance(out, (list, tuple)) else [out])

    t = 0.0025                     # net cord half-thickness
    xs = np.arange(-hx, hx + 1e-9, spacing)[1:-1]
    ys = np.arange(-hy, hy + 1e-9, spacing)[1:-1]
    zs = np.arange(0, rz + 1e-9, spacing)[1:-1]
    # vertical cords on the +-y walls (varying x) and the +-x walls (varying y)
    bars([t, t, rz / 2], [(x, sy * hy, rz / 2) for x in xs for sy in (-1, 1)]
         + [(sx * hx, y, rz / 2) for y in ys for sx in (-1, 1)], net_rgba)
    # horizontal cords: along x on the +-y walls and the roof, along y on the +-x walls and the roof
    bars([hx, t, t], [(0, sy * hy, z) for z in zs for sy in (-1, 1)] + [(0, y, rz) for y in ys], net_rgba)
    bars([t, hy, t], [(sx * hx, 0, z) for z in zs for sx in (-1, 1)] + [(x, 0, rz) for x in xs], net_rgba)
    # frame pipes on the 12 edges
    f = 0.012
    bars([f, f, rz / 2], [(sx * hx, sy * hy, rz / 2) for sx in (-1, 1) for sy in (-1, 1)], frame_rgba)
    bars([hx, f, f], [(0, sy * hy, z) for sy in (-1, 1) for z in (0, rz)], frame_rgba)
    bars([f, hy, f], [(sx * hx, 0, z) for sx in (-1, 1) for z in (0, rz)], frame_rgba)
    return ids


def _ring_points(centre, normal, R, n_beads):
    """n_beads points on the circle of radius R around `centre`, in the plane perpendicular to `normal`."""
    centre, n = np.asarray(centre, dtype=float), np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    u = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return [centre + R * (np.cos(a) * u + np.sin(a) * v)
            for a in np.linspace(0, 2 * np.pi, n_beads, endpoint=False)]


def build_ring(client, centre, normal, inner_radius=0.15, tube_radius=0.03, n_beads=28):
    """Visual-only ring: a necklace of blue spheres around the circle of radius inner_radius+tube_radius
    in the plane perpendicular to `normal`. Rebuild every episode (the ring is re-posed on reset)."""
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=tube_radius, rgbaColor=RING_BLUE,
                              physicsClientId=client)
    return [p.createMultiBody(0, -1, vis, pt.tolist(), physicsClientId=client)
            for pt in _ring_points(centre, normal, inner_radius + tube_radius, n_beads)]


def move_ring(client, ids, centre, normal, R):
    """Move the beads of a ring made by build_ring() to a new pose (the ring sways within an episode)."""
    for i, pt in zip(ids, _ring_points(centre, normal, R, len(ids))):
        p.resetBasePositionAndOrientation(i, pt.tolist(), [0, 0, 0, 1], physicsClientId=client)
