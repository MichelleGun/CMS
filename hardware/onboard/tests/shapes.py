"""
Synthetic ring images for the detector tests.

Drawn rather than loaded from photographs, so each test states its own geometry and can
cover cases a fixture set never would: a deliberately irregular ring, one clipped by the
frame edge, a same-coloured decoy.
"""
import numpy as np

BLUE_BGR = (235, 120, 40)        # inside the detector's default hue window
RED_BGR = (40, 40, 220)
BG = 70


def frame(w=480, h=360, bg=BG):
    img = np.full((h, w, 3), bg, np.uint8)
    img[: h // 3] = (90, 80, 75)         # a low-saturation "horizon", so the blue mask is
    return img                            # not trivially empty everywhere else


def draw_ring(img, centre, radius, *, ratio=1.0, thickness=None, colour=BLUE_BGR,
              angle=0.0):
    """An elliptical annulus: the ring as the camera sees it.

    thickness=-1 fills it, giving a disc -- a ring with no hole, which must be rejected.
    """
    import cv2
    t = thickness if thickness is not None else max(2, int(radius * 0.22))
    cv2.ellipse(img, (int(centre[0]), int(centre[1])),
                (int(radius), max(1, int(radius * ratio))), angle, 0, 360, colour, t)
    return img


def draw_wobbly_ring(img, centre, radius, *, wobble=0.14, lobes=5, colour=BLUE_BGR,
                     thickness=None, seed=0):
    """A ring whose radius varies with angle: a pool noodle bent by hand.

    This is the case the kit's detector rejects: its gates require the contour to fill
    0.8-1.15 of its fitted ellipse and to have an ellipse residual under 0.12, both of
    which assume a near-perfect circle.
    """
    import cv2
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    r = radius * (1.0 + wobble * np.sin(lobes * th + phase))
    pts = np.stack([centre[0] + r * np.cos(th), centre[1] + r * np.sin(th)], axis=1)
    t = thickness if thickness is not None else max(3, int(radius * 0.22))
    cv2.polylines(img, [pts.astype(np.int32)], True, colour, t)
    return img
