"""
Turn a policy trained in the PyBullet net env (scripts/train_hoop_multi.py, HoopNetAviary)
into demonstrations for the onboard-camera task, so the simulator runs and the human
flights can train one policy together.

    python hardware/onboard/make_teacher_demos.py --run results/hoop/runs/ppo_20261010-023538 \
        --episodes 100 --out hardware/flights/demos/teacher_ppo.npz
    python hardware/imitation/bc_train.py --demos hardware/flights/tello_*/demo.npz \
        hardware/flights/demos/teacher_ppo.npz --source-weight human=3 --out hardware/flights/bc/mixed

Why this direction, and not the other: the two tasks see different things. The PyBullet
policy reads 82 numbers of TRUE state (position, velocity, the ring's centre, facing, velocity
and size; policies trained before 2026-10-10 evening read 78, without the last two);
the onboard policy reads 33 numbers from the camera (where the ring is in the image, how
big). A human flight records only the camera numbers, so it cannot be turned into a PyBullet
observation. The onboard simulator, though, knows the true state at every tick. So the
PyBullet policy can fly the onboard simulator as a "teacher" that is told the truth, while
the recorder writes down what the camera-only "student" would have seen and what the
teacher did. That is ordinary privileged-teacher distillation.

The bridge, per tick (onboard sim runs at 15 Hz, PyBullet at 30 Hz):
  * teacher obs = [pos, rpy=(0,0,0), world vel, ang vel=0, last 15 PyBullet-rate actions,
    ring normal (the direction to pass through), [82-D teachers only: ring velocity = 0 (the
    onboard ring hangs still), ring opening radius], ring centre - pos]. rpy is zero because
    PyBullet's velocity controller holds yaw at 0, which is all the teacher has ever seen;
    its actions are world-frame, so the onboard drone's own yaw does not change their meaning.
  * teacher action [dx, dy, dz, s] -> world velocity SPEED_LIMIT * |s| * unit(d) (exactly
    what BaseRLAviary's VEL action does), then rotated into the body frame and divided by
    the onboard sim's MAX_SPEED to give rc lr / fb / ud.
  * yaw: the PyBullet task never turns, but the camera must keep the ring in view, so a
    small P-controller turns the nose toward the ring centre (frozen in the last 35 cm so
    the passage itself is not disturbed).
Only passed episodes are kept, as in make_demos.py: a clone copies its teacher's mistakes.
How many pass is printed, because the two simulators' dynamics differ and the yield is the
honest measure of how well the teacher transfers.
"""
import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

HW_ROOT = Path(__file__).resolve().parents[1]
CMS = HW_ROOT.parent
for p in (HW_ROOT, CMS / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# flake8: noqa: E402
from imitation.record_demo import DEMO_DIR, DemoRecorder
from onboard import spec
from onboard.gym_env import OnboardHoopEnv
from onboard.sim import MAX_SPEED, MAX_YAW_DPS

PYB_CTRL_HZ = 30                       # HoopNetAviary's ctrl_freq
PYB_SPEED_LIMIT = 0.03 * 30.0 / 3.6    # BaseRLAviary VEL: 3 % of the CF2X's 30 km/h, ~0.25 m/s
PYB_ACTION_BUFFER = PYB_CTRL_HZ // 2   # BaseRLAviary.ACTION_BUFFER_SIZE
TEACHER_OBS_DIM = 12 + 4 * PYB_ACTION_BUFFER + 10         # 82: + normal, ring vel, radius, rel
LEGACY_TEACHER_OBS_DIM = 12 + 4 * PYB_ACTION_BUFFER + 6   # 78: + normal, rel (older runs)
ARMED_TEACHER_OBS_DIM = TEACHER_OBS_DIM + 1               # 83: + the lap "armed" flag (v4 runs)
YAW_K = 1.0 / 35.0                     # rc per degree of heading error
YAW_MAX = 0.7
YAW_FREEZE_M = 0.35


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _rotation_between(a, b) -> np.ndarray:
    """Smallest rotation matrix taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-9:                       # opposite: half-turn about any axis normal to a
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


class SimTeacher:
    """Wraps a HoopNetAviary-trained SB3 model as an onboard-sim controller with privileged state.

    Call `act(sim)` once per onboard tick (it reads sim.pos/vel/yaw/ring); returns the rc
    action in [-1, 1]^4. `reset()` between episodes."""

    def __init__(self, model, deterministic=True, frame="canonical"):
        self.model = model
        self.deterministic = deterministic
        if frame not in ("canonical", "world"):
            raise ValueError("frame must be 'canonical' or 'world'")
        self.frame = frame
        shape = tuple(model.observation_space.shape)
        if shape[-1] not in (TEACHER_OBS_DIM, LEGACY_TEACHER_OBS_DIM, ARMED_TEACHER_OBS_DIM):
            raise ValueError(f"teacher expects observations of width {shape[-1]}; a HoopNetAviary "
                             f"(net env) policy has {TEACHER_OBS_DIM} (or {LEGACY_TEACHER_OBS_DIM} "
                             "for older runs). Is this a legacy-env or onboard-task model?")
        self.obs_shape = shape
        self.reset()

    def reset(self):
        self.buf = deque([np.zeros(4, dtype=np.float32)] * PYB_ACTION_BUFFER,
                         maxlen=PYB_ACTION_BUFFER)

    # -- the two halves of the bridge, separately testable ---------------------------
    def frame_of(self, sim):
        """-> (R, c_t): teacher-frame point = R @ (p - ring centre) + c_t, vector = R @ v.

        "world" is the identity. "canonical" rotates and shifts the scene so the onboard ring
        sits exactly at the PyBullet ring's nominal pose (scripts/hoop_net_env.py HoopSpec),
        which is where the teacher has spent its whole training: a PyBullet policy partly
        memorises that pose (measured: in the world frame it aims ~15 cm low and to one side
        of a ring hung where the onboard sim hangs it, and strikes). Its velocity commands
        are mapped back with R.T, so only the frame changes, not what the teacher decided."""
        c, n = sim.ring.centre, self.pass_dir(sim)
        if self.frame == "world":
            return np.eye(3), c
        from hoop_net_env import DEFAULT_SPEC, ring_normal
        target = ring_normal(DEFAULT_SPEC.normal_yaw_deg, DEFAULT_SPEC.normal_tilt_deg)
        return _rotation_between(n, target), np.asarray(DEFAULT_SPEC.centre, dtype=float)

    @staticmethod
    def pass_dir(sim) -> np.ndarray:
        """The direction the drone must pass through: away from the side it is on now."""
        n = sim.ring.normal
        return -n if float(np.dot(sim.pos - sim.ring.centre, n)) > 0 else n

    def teacher_obs(self, sim) -> np.ndarray:
        R, c_t = self.frame_of(sim)
        c = sim.ring.centre
        pos = R @ (sim.pos - c) + c_t
        kin = np.concatenate([pos, np.zeros(3), R @ sim.vel, np.zeros(3)])
        ring = [R @ self.pass_dir(sim)]
        if self.obs_shape[-1] in (TEACHER_OBS_DIM, ARMED_TEACHER_OBS_DIM):   # the ring hangs still
            ring += [np.zeros(3), [sim.ring.radius]]
        if self.obs_shape[-1] == ARMED_TEACHER_OBS_DIM:  # always armed: each demo is one pass
            ring += [[1.0]]
        obs = np.concatenate([kin, np.concatenate(list(self.buf)), *ring, R @ (c - sim.pos)])
        return obs.astype(np.float32).reshape(self.obs_shape)

    def action_to_rc(self, action, sim) -> np.ndarray:
        """PyBullet VEL action (in the teacher's frame) -> onboard rc [lr, fb, ud, yaw=0]."""
        a = np.clip(np.asarray(action, dtype=float).reshape(-1)[:4], -1, 1)
        R, _ = self.frame_of(sim)
        d = R.T @ a[:3]
        nd = float(np.linalg.norm(d))
        v = PYB_SPEED_LIMIT * abs(a[3]) * (d / nd) if nd > 1e-9 else np.zeros(3)
        rc = np.zeros(4)
        rc[spec.ACT_FB] = float(np.dot(v, sim.forward)) / MAX_SPEED
        rc[spec.ACT_LR] = float(np.dot(v, sim.right)) / MAX_SPEED
        rc[spec.ACT_UD] = float(v[2]) / MAX_SPEED
        return np.clip(rc, -1, 1)

    @staticmethod
    def yaw_to_ring(sim) -> float:
        d = sim.ring.centre - sim.pos
        if abs(float(np.dot(d, sim.ring.normal))) < YAW_FREEZE_M:
            return 0.0
        want = math.degrees(math.atan2(d[1], d[0]))
        err = _wrap_deg(want - sim.yaw_deg)
        # The sim turns by -action * MAX_YAW_DPS, so a positive error needs a negative action.
        return float(np.clip(-YAW_K * err, -YAW_MAX, YAW_MAX))

    def act(self, sim) -> np.ndarray:
        a, _ = self.model.predict(self.teacher_obs(sim), deterministic=self.deterministic)
        a = np.clip(np.asarray(a, dtype=np.float32).reshape(-1)[:4], -1, 1)
        # Two PyBullet control periods per onboard tick: the buffer keeps covering 0.5 s.
        for _ in range(max(1, round(PYB_CTRL_HZ / spec.CTRL_HZ))):
            self.buf.append(a.copy())
        rc = self.action_to_rc(a, sim)
        rc[spec.ACT_YAW] = self.yaw_to_ring(sim)
        return rc


def load_teacher(path, frame="canonical"):
    from stable_baselines3 import PPO, A2C, SAC, TD3
    sys.path.insert(0, str(CMS / "scripts"))
    from vec_util import resolve_model_path
    zp = resolve_model_path(path)
    meta_p = zp.parent / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    if meta.get("env", "net") != "net":
        raise ValueError(f"{zp} was trained on env '{meta.get('env')}'; the teacher needs a "
                         "net-env (HoopNetAviary) policy")
    if meta.get("algo") == "dqn":
        raise ValueError("DQN's discrete moves are not supported as a teacher; use a "
                         "continuous-action run (ppo/a2c/sac/td3)")
    cls = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3}[meta.get("algo", "ppo")]
    return SimTeacher(cls.load(str(zp), device="cpu"), frame=frame), zp, meta


def generate(teacher, episodes=100, profile="typical", seed=0, keep_failures=False,
             verbose=True, source_info=None):
    env = OnboardHoopEnv(profile=profile, seed=seed)
    rec = DemoRecorder(meta={"source": "sim_teacher", "profile": profile, "sim": True,
                             "episodes_requested": episodes, **(source_info or {})},
                       obs_dim=spec.OBS_DIM, task=spec.TASK_NAME, obs_shape=(spec.OBS_DIM,))
    tally = {"passed": 0, "struck": 0, "missed": 0, "out_of_cage": 0, "timeout": 0}
    kept = 0
    try:
        for e in range(episodes):
            obs, _ = env.reset(seed=seed + e)
            teacher.reset()
            episode, t = [], 0.0
            while True:
                action = teacher.act(env.sim)
                episode.append((obs.copy(), action.astype(np.float32), t))
                obs, _r, term, trunc, info = env.step(action)
                t += spec.CTRL_DT
                if term or trunc:
                    phase = info["phase"] if term else "timeout"
                    break
            tally[phase] += 1
            if phase == "passed" or keep_failures:
                for o, a, ts in episode:
                    rec.add(o, a, a, ts)
                rec.next_episode()
                kept += 1
            if verbose and (e + 1) % max(1, episodes // 10) == 0:
                print(f"[teacher] {e + 1}/{episodes} episodes: {tally}", flush=True)
    finally:
        env.close()
    tally["kept_episodes"] = kept
    return rec, tally


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="a net-env run directory (latest -> best -> final model) or a model .zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh"], default="typical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"default: {DEMO_DIR}/teacher_<run>.npz")
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--frame", choices=["canonical", "world"], default="canonical",
                    help="canonical = show the teacher the onboard ring at the PyBullet ring's "
                         "nominal pose (see SimTeacher.frame_of); world = the raw coordinates")
    a = ap.parse_args(argv)

    teacher, zp, meta = load_teacher(a.run, a.frame)
    run_id = meta.get("run_id", zp.parent.name)
    print(f"[teacher] {zp} ({meta.get('algo', 'ppo')}, run {run_id})")
    rec, tally = generate(teacher, a.episodes, a.profile, a.seed, a.keep_failures,
                          source_info={"teacher_run": run_id, "teacher_model": zp.name,
                                       "teacher_frame": a.frame})
    print(f"[teacher] passed {tally['passed']}/{a.episodes} "
          f"({tally['passed'] / max(1, a.episodes):.0%}); struck {tally['struck']}, "
          f"missed {tally['missed']}, out of cage {tally['out_of_cage']}, "
          f"timeout {tally['timeout']}")
    if not len(rec):
        sys.exit("[teacher] no episode passed, so there is nothing to learn from. The PyBullet "
                 "policy does not transfer to the onboard simulator as it is; train the clone "
                 "on the human and scripted-expert demos instead (or try --profile ideal).")
    out = a.out or (DEMO_DIR / f"teacher_{run_id}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec.meta.update(tally)
    path = rec.save(out)
    print(f"[teacher] {len(rec)} samples from {tally['kept_episodes']} episodes -> {path}")
    return path, tally


if __name__ == "__main__":
    main()
