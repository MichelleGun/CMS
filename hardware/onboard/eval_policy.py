"""
Score a trained onboard-camera policy in the simulator and keep the result next to it.

    python hardware/onboard/eval_policy.py --model hardware/flights/bc/my_clone/bc_model.zip
    python hardware/onboard/eval_policy.py --model ... --episodes 12 --render

Writes <model folder>/evals/<timestamp>.json so the dashboard can show every evaluation a
model has had. `--render` puts the real OpenCV detector in the loop (slower, stricter).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HW_ROOT = Path(__file__).resolve().parents[1]
if str(HW_ROOT) not in sys.path:
    sys.path.insert(0, str(HW_ROOT))

# flake8: noqa: E402
from onboard.gym_env import rollout
from onboard.policy import load_policy


def evaluate(model: Path, episodes=12, seed=100, profile="typical", render=False) -> dict:
    fn = load_policy(model)
    res = rollout(fn, episodes=episodes, seed=seed, profile=profile, render_frames=render)
    res.update(model=str(model), seed=seed, profile=profile, render=bool(render),
               saved=time.strftime("%Y-%m-%d %H:%M:%S"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--profile", choices=["ideal", "typical", "harsh"], default="typical")
    ap.add_argument("--render", action="store_true",
                    help="score against the real OpenCV detector instead of the analytic model")
    a = ap.parse_args(argv)
    if a.episodes < 1:
        ap.error("--episodes must be at least 1")
    if not a.model.is_file():
        sys.exit(f"[eval] no model at {a.model}")

    res = evaluate(a.model, a.episodes, a.seed, a.profile, a.render)
    out = a.model.parent / "evals"
    out.mkdir(exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}{'_render' if a.render else ''}.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[eval] passed {res['passed']}/{res['episodes']}  struck {res['struck']}  "
          f"missed {res['missed']}  ({'real detector' if a.render else 'analytic'}) -> {path}")
    return res


if __name__ == "__main__":
    main()
