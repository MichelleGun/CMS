"""
A training run's configuration, for the dashboard's Config card: what a finished or live run was
started with (read from its meta.json and model), which of those values can be edited for the next
run, and the exact train_onboard.py command line a set of edited values becomes.

The editable fields are described once, in FIELDS and REWARD_FIELDS. The card renders its form from
that description (GET /api/runs/{id}/config), POST /api/runs takes the same field names, and
onboard_argv() turns them into the command. So the form, the API and the command cannot drift apart.
Only the onboard-camera task is described here (the net / legacy simulator runs keep their own panel).
"""
import json
import shlex
from pathlib import Path
from typing import Any, Optional

from config import CMS_DIR, SCRIPTS_DIR

ONBOARD_TRAIN = CMS_DIR / "hardware" / "onboard" / "train_onboard.py"

# name -> default of every reward value (hardware/onboard/gym_env.REWARD_WEIGHTS). A test pins these
# to the trainer's own table, so a new weight there fails loudly here.
REWARD_DEFAULTS = {
    "w_shape": 3.0, "axis_pull": 1.0, "w_time": 0.02, "r_passed": 20.0, "r_struck": -8.0,
    "r_missed": -5.0, "r_out_of_cage": -10.0, "r_found": 0.5, "w_rim": 0.3, "rim_margin_m": 0.08,
    "r_wrong_way": -5.0, "arm_dist_m": 0.25, "drain": True, "w_face": 0.5, "face_ahead_m": 0.3,
    "pass_facing_min": 0.5, "return_side_m": 0.55, "w_dwell": 0.0,
}
# streak / streak_cap are reward values too, but they are ordinary fields (and trainer flags) of
# their own, so they sit with the environment settings below.
REWARD_HELP = {
    "w_shape": "weight of the progress shaping (metres gained towards the next waypoint)",
    "axis_pull": "extra pull onto the ring's axis near the plane (x radial metres)",
    "w_time": "time penalty per tick, and the 'drain' paid when an episode ends early",
    "r_passed": "bonus for a counted pass (x the streak multiplier)",
    "r_struck": "penalty for touching the noodle (the drain is added)",
    "r_missed": "penalty for crossing the plane outside the opening (the drain is added)",
    "r_out_of_cage": "penalty for leaving the cage (the drain is added)",
    "r_found": "once per search: first time the ring comes into view",
    "w_rim": "penalty slope as the drone nears the noodle (full at touching)",
    "rim_margin_m": "clearance (m) at which the rim penalty starts",
    "r_wrong_way": "penalty for passing the wrong way / before re-arming",
    "arm_dist_m": "how far clear on the exit side before the next pass counts (m)",
    "drain": "failures also cost w_time x the ticks left, so ending early is never cheaper",
    "w_face": "weight of the facing potential (camera towards the next waypoint)",
    "face_ahead_m": "the facing target sits this far behind the ring (m)",
    "pass_facing_min": "a pass only counts with forward . pass direction at least this",
    "return_side_m": "same-side laps: the way round passes this far beside the ring (m)",
    "w_dwell": "per tick loitering in the opening without pushing forward (train_onboard sets 0.05)",
}

CHOICES_PROFILE = ["ideal", "typical", "harsh", "measured"]

# key = the POST /api/runs field name. `src` = where the current run's value is in meta.json
# (dotted path; a tuple is tried in order). `flag` = the train_onboard.py flag, None = built
# specially in onboard_argv. type: int / float / bool / choice. null_ok: empty = the trainer's default.
FIELDS = [
    # ---- run
    dict(key="timesteps", group="Run", label="Timesteps", type="int", min=1000, max=10**10, step=100000,
         src="timesteps_requested", help="steps this run adds (a continued run's step count carries on)"),
    dict(key="until_stopped", group="Run", label="Until stopped", type="bool", src="until_stopped",
         help="ignore the step budget and train until Stop"),
    dict(key="n_envs", group="Run", label="Parallel envs", type="int", min=1, max=16, src="n_envs"),
    dict(key="seed", group="Run", label="Seed", type="int", min=0, max=2**31 - 1, src="seed"),
    dict(key="eval_freq", group="Run", label="Eval every (per env)", type="int", min=100, max=10**6,
         src="eval_freq", help="x n_envs timesteps between evaluations"),
    dict(key="n_eval_episodes", group="Run", label="Eval episodes", type="int", min=1, max=500,
         src="n_eval_episodes"),
    # ---- environment
    dict(key="profile", group="Environment", label="Drone profile", type="choice", choices=CHOICES_PROFILE,
         src="profile", help="how the pretend drone behaves; 'measured' is fitted to the real flight logs"),
    dict(key="onboard_dr", group="Environment", label="Domain randomisation", type="float", min=0, max=1,
         step=0.1, src="robust.dr", help="0 = off, 1 = every episode draws a drone between ideal and harsh"),
    dict(key="onboard_sway_deg", group="Environment", label="Ring swing (max deg)", type="float", min=0,
         max=45, step=1, src="robust.sway_deg"),
    dict(key="onboard_wash_deg", group="Environment", label="Prop-wash push (max deg)", type="float", min=0,
         max=45, step=1, src="robust.wash_deg"),
    dict(key="hard_start_p", group="Environment", label="Hard starts", type="float", min=0, max=1, step=0.05,
         src="robust.hard_start_p", help="share of starts behind / beside / below the ring"),
    dict(key="dr_steps", group="Environment", label="Randomisation ramp (steps)", type="int", min=0,
         max=10**8, step=50000, src="robust.dr_steps", help="0 = full strength from the start"),
    dict(key="onboard_laps", group="Environment", label="Passes per episode", type="int", min=0, max=20,
         src="laps", help="1 = single pass, 0 = endless laps"),
    dict(key="same_side", group="Environment", label="Same side every time", type="bool", src="lap_dir=same",
         help="laps round the outside (else back-and-forth through the ring)"),
    dict(key="start_zones", group="Environment", label="Start from all 9 zones", type="bool",
         src="robust.start_mode=zones", help="else the front runway"),
    dict(key="streak", group="Environment", label="Streak bonus", type="float", min=0, max=5, step=0.1,
         src="streak", help="pass k pays r_passed x min(1 + streak x (k-1), streak cap)"),
    dict(key="streak_cap", group="Environment", label="Streak cap", type="float", min=1, max=20, step=0.5,
         src="streak_cap"),
    dict(key="ring_tracker", group="Environment", label="Ring tracker (41 inputs)", type="bool",
         src="ring_tracker", help="dead-reckoned ring position while the ring is out of view; "
                                  "a 37-input parent is widened with zero weights"),
    dict(key="bc_augment", group="Environment", label="Augment demos (mirror + noise)", type="bool",
         src="robust.bc_augment"),
    # ---- PPO
    dict(key="learning_rate", group="PPO", label="Learning rate", type="float", min=1e-6, max=1e-2,
         src="learning_rate", sci=True),
    dict(key="gamma", group="PPO", label="Discount (gamma)", type="float", min=0.8, max=0.9999, step=0.001,
         src="ppo.gamma", flag="--gamma"),
    dict(key="gae_lambda", group="PPO", label="GAE lambda", type="float", min=0.8, max=1.0, step=0.01,
         src="ppo.gae_lambda", flag="--gae-lambda", help="1.0 = unbiased advantages (a baseline adds no bias)"),
    dict(key="ent_coef", group="PPO", label="Entropy bonus", type="float", min=0, max=0.1, step=0.001,
         src="ppo.ent_coef", flag="--ent-coef"),
    dict(key="target_kl", group="PPO", label="Target KL (trust region)", type="float", min=0.001, max=0.5,
         step=0.005, src="ppo.target_kl", flag="--target-kl", null_ok=True,
         help="stops an update's epochs early when the policy moved this far; empty = off"),
    dict(key="batch_size", group="PPO", label="Batch size", type="int", min=16, max=65536, step=64,
         src="ppo.batch_size", flag="--batch-size"),
    dict(key="n_epochs", group="PPO", label="Epochs per update", type="int", min=1, max=50,
         src="ppo.n_epochs", flag="--n-epochs"),
    dict(key="critic_warmup_steps", group="PPO", label="Critic warm-up (steps)", type="int", min=0, max=10**7,
         step=10000, src="critic_warmup_steps", flag="--critic-warmup-steps",
         help="the actor is frozen while the critic learns the parent's values"),
    dict(key="log_std_reset", group="PPO", label="Reset log-std to", type="float", min=-5, max=1, step=0.1,
         src="ppo.log_std_reset", flag="--log-std-reset", null_ok=True,
         help="a continued run can restart its action noise; empty = keep the parent's"),
    dict(key="sde", group="PPO", label="gSDE exploration", type="bool", src="ppo.use_sde", flag="--sde"),
    # ---- imitation
    dict(key="bc_coef", group="Imitation", label="Demo term strength (start)", type="float", min=0, max=10,
         step=0.1, src="bc.coef"),
    dict(key="bc_coef_min", group="Imitation", label="Demo term strength (floor)", type="float", min=0,
         max=10, step=0.05, src="bc.coef_min"),
    dict(key="bc_decay_steps", group="Imitation", label="Demo term decay (steps)", type="int", min=1,
         max=10**8, step=50000, src="bc.decay_steps"),
    dict(key="human_weight", group="Imitation", label="Human demo weight", type="float", min=0, max=20,
         step=0.5, src="bc.source_weights.human"),
]
FIELD_BY_KEY = {f["key"]: f for f in FIELDS}
# numbers the trainer uses when the flag is left out (meta from before 2026-10-13 has no `ppo` block)
PPO_FALLBACK = {"gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0, "target_kl": None, "batch_size": 64,
                "n_epochs": 10, "log_std_reset": None, "use_sde": False}


def _dig(d: Any, path: str):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _from_meta(meta: dict, f: dict):
    """The value of one field in a run's meta.json (None if the run did not record it)."""
    src = f["src"]
    if "=" in src:                                     # a boolean: does this path equal that value
        path, want = src.split("=", 1)
        got = _dig(meta, path)
        return None if got is None else got == want
    return _dig(meta, src)


def current_values(meta: dict, arch: Optional[dict] = None) -> dict:
    """{field key: value} for a run: its meta, else the model's own hyperparameters, else the
    trainer's default. `timesteps` is what the run ADDED (meta records start + budget)."""
    out = {}
    for f in FIELDS:
        v = _from_meta(meta, f)
        if v is None and f["key"] == "ring_tracker":
            v = (meta.get("obs_dim") or 0) > 37
        if v is None and f["key"] in PPO_FALLBACK:
            v = PPO_FALLBACK[f["key"]]
        if v is None and f["key"] == "sde":
            v = False
        if v is None and arch:
            v = arch.get(f"ppo.{f['key']}")
        out[f["key"]] = v
    ts, init = meta.get("timesteps_requested"), meta.get("init_timesteps")
    if ts and init:
        out["timesteps"] = max(1000, int(ts) - int(init))
    if meta.get("until_stopped"):
        out["timesteps"] = 1_000_000
    sde = _dig(meta, "ppo.use_sde")
    if sde is not None:
        out["sde"] = bool(sde)
    return out


def current_reward(meta: dict) -> dict:
    rw = meta.get("reward") or {}
    return {k: rw.get(k, d) for k, d in REWARD_DEFAULTS.items()}


def field_schema() -> list:
    """The form description the frontend renders (no values)."""
    return [dict(f) for f in FIELDS]


def reward_schema() -> list:
    return [dict(key=k, default=d, type="bool" if isinstance(d, bool) else "float",
                 help=REWARD_HELP.get(k, "")) for k, d in REWARD_DEFAULTS.items()]


def _num(v):
    return repr(float(v)) if isinstance(v, float) else str(v)


def onboard_argv(python_exe: str, run_id: str, run_dir, p: dict, init_dir=None, demos=()) -> list:
    """The train_onboard.py command for one set of field values `p` (the POST /api/runs fields).
    A flag is only added when its value was given, so a payload from the old form builds exactly
    the command it always did."""
    argv = [python_exe, str(ONBOARD_TRAIN),
            "--timesteps", str(p["timesteps"]), "--n-envs", str(p["n_envs"]), "--vec", p.get("vec", "auto"),
            "--eval-freq", str(p["eval_freq"]), "--n-eval-episodes", str(p["n_eval_episodes"]),
            "--run-id", run_id, "--out", str(run_dir)]
    if p.get("until_stopped"):
        argv.append("--until-stopped")
    if p.get("seed") is not None:
        argv += ["--seed", str(p["seed"])]
    if init_dir is not None:
        argv += ["--init", str(init_dir)]
    argv += ["--profile", p["profile"], "--dr", str(p["onboard_dr"]), "--sway-deg", str(p["onboard_sway_deg"]),
             "--wash-deg", str(p["onboard_wash_deg"]),
             "--lap-dir", "same" if p["same_side"] else "alternate",
             "--start-mode", "zones" if p["start_zones"] else "ring", "--streak", str(p["streak"]),
             "--hard-start-p", str(p["hard_start_p"]), "--dr-steps", str(int(p["dr_steps"])),
             "--laps", str(int(p["onboard_laps"])),
             "--bc-augment" if p["bc_augment"] else "--no-bc-augment"]
    if p.get("streak_cap") is not None:
        argv += ["--streak-cap", str(p["streak_cap"])]
    if p.get("ring_tracker"):
        argv.append("--ring-tracker")
    if demos:
        argv += ["--demos", *(str(d) for d in demos), "--source-weight", f"human={p['human_weight']}",
                 "--bc-coef", str(p["bc_coef"])]
        if p.get("bc_coef_min") is not None:
            argv += ["--bc-coef-min", str(p["bc_coef_min"])]
        if p.get("bc_decay_steps") is not None:
            argv += ["--bc-decay-steps", str(int(p["bc_decay_steps"]))]
    if p.get("learning_rate") is not None:
        argv += ["--lr", _num(float(p["learning_rate"]))]
    for key, f in FIELD_BY_KEY.items():
        flag = f.get("flag")
        if not flag or f["group"] != "PPO":
            continue
        v = p.get(key)
        if v is None or v is False:
            continue
        argv.append(flag) if v is True else argv.extend([flag, _num(v)])
    over = {k: v for k, v in (p.get("reward_weights") or {}).items() if k in REWARD_DEFAULTS
            and v != REWARD_DEFAULTS[k]}
    if over:
        argv += ["--reward-weights", json.dumps(over, separators=(",", ":"))]
    return argv


def _relative(arg: str, root: str) -> str:
    """A path inside the project shown relative to it, with forward slashes (what you type from CMS/)."""
    if arg.startswith(root):
        return arg[len(root):].lstrip("\\/").replace("\\", "/")
    return arg


def command_line(argv: list) -> str:
    """The argv as one pasteable line, with the python and script shortened to what you would type."""
    parts = list(argv)
    if len(parts) > 1 and Path(parts[1]) == ONBOARD_TRAIN:
        parts = ["python", "hardware/onboard/train_onboard.py", *parts[2:]]
    root = str(CMS_DIR)
    parts = [_relative(str(x), root) for x in parts]
    return " ".join(shlex.quote(str(x)) if " " in str(x) or "{" in str(x) or '"' in str(x) else str(x)
                    for x in parts)


_ARCH_CACHE: dict = {}
_ARCH_SNIPPET = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
                 "from clearml_sync import model_architecture; print(json.dumps(model_architecture(sys.argv[2])))")


def architecture(run_dir: Path) -> dict:
    """Layer shapes, activation, log-std and the saved PPO numbers of the run's newest model, read
    from its zip without building an env (scripts/clearml_sync.model_architecture). That needs torch,
    which the dashboard process never imports, so it runs in a short subprocess; the answer is cached
    by file and modification time."""
    import subprocess
    import sys
    for name in ("latest_model.zip", "final_model.zip", "best_model.zip", "bc_model.zip"):
        z = run_dir / name
        if z.is_file():
            key = (str(z), z.stat().st_mtime_ns)
            if key not in _ARCH_CACHE:
                try:
                    r = subprocess.run([sys.executable, "-c", _ARCH_SNIPPET, str(SCRIPTS_DIR), str(z)],
                                       capture_output=True, text=True, timeout=90)
                    arch = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else                         {"architecture_error": (r.stderr or "no output").strip()[-300:]}
                except Exception as e:                       # never let this stop the card from opening
                    arch = {"architecture_error": str(e)[:300]}
                _ARCH_CACHE.clear()
                _ARCH_CACHE[key] = dict(arch, model_file=name)
            return _ARCH_CACHE[key]
    return {}
