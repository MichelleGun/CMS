"""
The onboard-camera hoop-traversal task: observation layout, action layout, and the
normalisation constants.

This module is the single source of truth. The simulator (sim.py), the real flight
(fly_hoop.py), the demonstration recorder (record.py) and the trainers all import it,
so the interface cannot drift between "what the expert was recorded doing" and "what a
policy is asked to do". A test asserts OBS_DIM equals the sum of the field widths.

Why a feature vector and not the image: a 33-float observation trains an MLP on a CPU in
seconds, keeps demonstration files small, and runs at 15 Hz on this machine. The cost is
that the policy only ever sees what ring_detect.py chose to measure.

Why this is NOT the simulator's task (scripts/hoop_env.py):
  * The sim rewards reaching a *point* (-dist, +10 inside 0.15 m) and TERMINATES on
    arrival, so a policy trained there stops at the centre instead of passing through.
  * The sim never commands yaw (a PD loop holds it), so it cannot orient to a ring plane.
  * The sim's action is a direction plus a speed fraction; here it is four independent
    rc channels, because that is what send_rc_control takes and what a human flies with.
Those three differences are why this task, its observation and its action are new.
"""
import numpy as np

# --- control rate -----------------------------------------------------------
CTRL_HZ = 15                      # Tello telemetry is ~10 Hz and ring detection costs
CTRL_DT = 1.0 / CTRL_HZ           # real CPU; 15 Hz leaves headroom for both

# --- action -----------------------------------------------------------------
# [lr, fb, ud, yaw], each in [-1, 1], mapped to send_rc_control by action_to_rc().
ACT_DIM = 4
ACT_LR, ACT_FB, ACT_UD, ACT_YAW = 0, 1, 2, 3
ACT_BUFFER = 4                    # past actions in the observation (~0.27 s at 15 Hz)

# --- observation layout -----------------------------------------------------
# Field widths, in order. OBS_DIM is their sum; a test checks that.
OBS_FIELDS = (
    ("found", 1),        # 1.0 when the tracker holds a lock this tick, else 0.0
    ("centre", 2),       # ring centre offset from image centre, in half-frames [-1, 1]
    ("scale", 1),        # apparent size: major axis / frame width, x SCALE_NORM
    ("axis_ratio", 1),   # minor/major in [0, 1]; 1.0 = square-on, low = seen edge-on
    ("range", 1),        # estimated metres to the ring / RANGE_NORM
    ("age", 1),          # ticks since the last real detection / AGE_NORM
    # Where the ring was when it was last actually seen. Without these the task is
    # partially observable at the one moment that matters: during the blind final push
    # the ring is not visible, so found=0 and range reads "far" -- which is exactly what
    # the observation also looks like while searching at the start of a flight. The two
    # demand opposite actions (push forward vs turn and look), so a memoryless policy
    # cannot tell them apart and stalls short of the ring. Measured: a cloned policy got
    # to within 17 cm of the ring plane and then hovered until the episode timed out,
    # 0/12. These fields persist after the lock is lost, which separates the two cases.
    ("last_range", 1),   # range at the last fresh detection / RANGE_NORM (1.0 if never)
    ("last_centre", 2),  # cx, cy at the last fresh detection
    ("height", 1),       # Tello 'h' (cm) / HEIGHT_NORM
    ("vel", 3),          # Tello vgx/vgy/vgz (cm/s) / VEL_NORM
    ("att", 3),          # pitch, roll (deg) / ATT_NORM, and yaw rate / YAWRATE_NORM
    ("actions", ACT_BUFFER * ACT_DIM),   # executed actions, oldest first, zeros at takeoff
)
OBS_DIM = sum(w for _, w in OBS_FIELDS)        # 33

# Slice for each field, so consumers never hardcode an index.
OBS_SLICES = {}
_off = 0
for _name, _w in OBS_FIELDS:
    OBS_SLICES[_name] = slice(_off, _off + _w)
    _off += _w
del _off, _name, _w

# --- normalisation ----------------------------------------------------------
# Chosen so a typical flight keeps every field inside about [-1, 1]. They are not
# clamps (except where noted); a policy can still see an out-of-range value.
SCALE_NORM = 2.0          # major/width of 0.5 (ring filling half the frame) -> 1.0
RANGE_NORM = 2.0          # metres; the cage is 2 m across
LOCK_MAX_AGE = 15         # ticks (1 s at 15 Hz) before the TRACKER declares a lock lost
# The observation's `age` is normalised by a longer span than LOCK_MAX_AGE on purpose:
# the blind push lasts about 3 s, so an age saturating at 1 s would read identically
# one second and three seconds into it, losing the information a policy needs to know
# how far through the push it is.
AGE_NORM = 60             # ticks (4 s at 15 Hz)
HEIGHT_NORM = 150.0       # cm; takeoff settles near 80
VEL_NORM = 50.0           # cm/s
ATT_NORM = 30.0           # deg
YAWRATE_NORM = 90.0       # deg/s

# --- geometry / hardware ----------------------------------------------------
# Two diameters, and confusing them is a real bug: the camera sees the noodle's OUTER
# edge, so ranging must use the outer diameter, while whether the drone fits through is
# decided by the opening. MEASURED: a 30 cm opening in a 6 cm noodle, so outer = 42 cm.
RING_OPENING_M = 0.30     # MEASURED: the clear gap the drone has to fly through
RING_NOODLE_M = 0.06      # MEASURED: the noodle's thickness
RING_OUTER_M = RING_OPENING_M + 2 * RING_NOODLE_M   # what the detector measures
RING_DIAMETER_M = RING_OUTER_M                      # used by the range estimate

# How wide the aircraft is, which is a DIFFERENT question from what the camera sees and
# the reason the two must not be conflated. The opening gives 15 cm of radius; this gives
# the drone 7 cm of it, leaving 8 cm of radial clearance from the drone's centre to the
# noodle. Every alignment gate in expert.py has to live inside that 8 cm, and sim.py uses
# it to decide pass vs strike -- without it the simulator scores the drone as a point and
# would call a 14 cm miss a clean pass through a 30 cm hole.
#   Tello + prop guards  ~18 cm -> 6.0 cm clearance
#   Tello + props only   ~14 cm -> 8.0 cm clearance   <- flown configuration
#   Tello body only       9.8 cm -> 10.1 cm clearance
DRONE_WIDTH_M = 0.14      # flying without prop guards
DRONE_RADIUS_M = DRONE_WIDTH_M / 2.0
RING_CLEARANCE_M = RING_OPENING_M / 2.0 - DRONE_RADIUS_M   # 0.08: the whole error budget

CAMERA_HFOV_DEG = 70.0    # Tello front camera: 82.6 deg diagonal on 4:3 -> ~70 deg
                          # horizontal. Approximate; checked against a tape at stage 1.
RC_MAX = 30               # hard clamp per channel. The Tello accepts +-100; 30 is a
                          # deliberate project limit (hardware/config.yaml uses 35 for
                          # the same reason). It cannot go much lower: at an assumed
                          # ~125 rc per m/s, rc 30 is only ~0.24 m/s, and the blind
                          # commit push has to cover ~0.42 m with it.

TASK_NAME = "onboard_hoop"   # written into demo meta so trainers pick the right loss


def focal_px(frame_width: int, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Pinhole focal length in pixels, from the horizontal field of view."""
    return (frame_width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)


def offset_metres(norm_offset: float, range_m: float,
                  hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """A normalised image offset (in half-frame-widths) -> metres at that range.

    Needed because an image offset means nothing on its own: cx = 0.08 is a 2.2 cm
    lateral error at 0.42 m but a 9 cm one at 1.6 m. Any threshold about "is the drone
    lined up well enough to fly through a 30 cm hole" is a statement about metres, so it
    has to be expressed in metres and converted, not fixed in image units.
    """
    if not np.isfinite(range_m) or not np.isfinite(norm_offset):
        return float("inf")
    return float(abs(norm_offset) * range_m * np.tan(np.radians(hfov_deg) / 2.0))


def range_from_major_axis(major_px: float, frame_width: int,
                          ring_diameter_m: float = RING_DIAMETER_M,
                          hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Distance to the ring (m) from its apparent major axis in pixels.

    The MAJOR axis is used, not the minor or the area: viewed off-axis a circle projects
    to an ellipse whose major axis stays the true diameter while the minor foreshortens.
    So this estimate is insensitive to the approach angle, which the area would not be.
    Returns inf for a degenerate measurement rather than raising.
    """
    if major_px <= 1e-6:
        return float("inf")
    return float(focal_px(frame_width, hfov_deg) * ring_diameter_m / major_px)


def action_to_rc(action, rc_max: int = RC_MAX):
    """[lr, fb, ud, yaw] in [-1, 1] -> four ints for send_rc_control, hard-clamped.

    Clamping happens here and nowhere else, so no controller (scripted, cloned or
    learned) can command more than rc_max even if it returns garbage. NaNs become 0.
    """
    a = np.nan_to_num(np.asarray(action, dtype=float).reshape(-1)[:ACT_DIM], nan=0.0,
                      posinf=0.0, neginf=0.0)
    a = np.clip(a, -1.0, 1.0)
    return [int(round(v * rc_max)) for v in a]


def empty_obs() -> np.ndarray:
    return np.zeros(OBS_DIM, dtype=np.float32)
