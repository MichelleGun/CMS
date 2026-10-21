"""
Named camera calibrations ("presets") for the ring detector: day, night, or anything else.

The ring detector's colour window and shape gates (ring_detect.DetectorParams) depend on the
light. On 2026-10-14 a daytime flight locked onto the window instead of the hoop with the
calibration that worked at night. So each lighting situation gets its own preset, saved from the
dashboard's Calibrate camera panel (or tune_detector.py --preset), and the flight picks one:

    python hardware/onboard/fly_hoop.py --mode policy --model <zip> --detector day

Files: hardware/onboard/detector_presets/<slug>.json = {name, condition, notes, created_at,
params, thumb}. The classic hardware/onboard/detector.json is listed as the preset "default";
"set as default" copies a preset into it, so flights started without --detector use it.

Plain JSON only (no OpenCV), so the dashboard can use it without loading the detector.
"""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
PRESET_DIR = HERE / "detector_presets"
DEFAULT_FILE = HERE / "detector.json"
DEFAULT_SLUG = "default"
CONDITIONS = ("day", "night", "other")
# DetectorParams' fields (ring_detect.py); anything else in a saved file is ignored
PARAM_KEYS = ("hsv_lo", "hsv_hi", "min_area_px", "min_side_px", "opening_ratio", "fill_ratio",
              "residual_max", "min_axis_ratio", "allow_edge", "close_kernel", "border_tol",
              "min_arc_deg", "min_major_px")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return s[:48] or "preset"


def clean_params(params: dict) -> dict:
    """Only known keys, with the colour window checked (OpenCV hue 0-179, sat/val 0-255)."""
    out = {k: params[k] for k in PARAM_KEYS if k in params}
    for key in ("hsv_lo", "hsv_hi"):
        if key in out:
            v = [int(x) for x in out[key]]
            if len(v) != 3 or not (0 <= v[0] <= 179 and 0 <= v[1] <= 255 and 0 <= v[2] <= 255):
                raise ValueError(f"{key} must be [hue 0-179, sat 0-255, val 0-255], got {out[key]}")
            out[key] = v
    if "hsv_lo" in out and "hsv_hi" in out and any(a > b for a, b in zip(out["hsv_lo"], out["hsv_hi"])):
        raise ValueError("each hsv_lo value must be <= its hsv_hi value")
    return out


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def default_preset() -> dict:
    params = _read(DEFAULT_FILE) or {}
    return dict(slug=DEFAULT_SLUG, name="Default (detector.json)", condition="other",
                notes="what flights use when no preset is chosen", created_at=None,
                params=clean_params(params), thumb=None, is_default=True,
                path=str(DEFAULT_FILE))


def list_presets() -> list:
    """The default first, then saved presets, newest first."""
    out = []
    for p in PRESET_DIR.glob("*.json"):
        d = _read(p)
        if not d or "params" not in d:
            continue
        d = dict(d, slug=p.stem, path=str(p), is_default=False)
        thumb = p.with_suffix(".jpg")
        d["thumb"] = thumb.name if thumb.is_file() else None
        out.append(d)
    out.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    return [default_preset()] + out


def get_preset(name_or_path: str) -> dict:
    """A preset by slug, by display name, or a path to a JSON file of detector params."""
    key = str(name_or_path)
    if slugify(key) == DEFAULT_SLUG:
        return default_preset()
    for d in list_presets():
        if d["slug"] == slugify(key) or d.get("name") == key:
            return d
    p = Path(key)
    if p.suffix == ".json" and p.is_file():
        d = _read(p) or {}
        params = d.get("params", d)
        return dict(slug=p.stem, name=d.get("name", p.stem), condition=d.get("condition", "other"),
                    params=clean_params(params), path=str(p), is_default=False)
    names = ", ".join(d["slug"] for d in list_presets())
    raise KeyError(f"no camera preset '{name_or_path}' (have: {names})")


def save_preset(name: str, params: dict, condition: str = "other", notes: str = "",
                thumb_jpg: Optional[bytes] = None, overwrite: bool = True) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    slug = slugify(name)
    if slug == DEFAULT_SLUG:
        raise ValueError("'default' is reserved: use 'set as default' instead")
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESET_DIR / f"{slug}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"a preset '{slug}' already exists")
    doc = dict(name=str(name).strip() or slug, condition=condition, notes=str(notes or ""),
               created_at=datetime.now().isoformat(timespec="seconds"), params=clean_params(params))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)
    if thumb_jpg:
        path.with_suffix(".jpg").write_bytes(thumb_jpg)
    return get_preset(slug)


def set_default(slug: str) -> dict:
    """Copy a preset's params into detector.json (what a flight without --detector uses)."""
    d = get_preset(slug)
    if d["slug"] == DEFAULT_SLUG:
        return d
    tmp = DEFAULT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d["params"], indent=2), encoding="utf-8")
    tmp.replace(DEFAULT_FILE)
    return default_preset()


def delete_preset(slug: str) -> None:
    if slugify(slug) == DEFAULT_SLUG:
        raise ValueError("the default can't be deleted")
    path = PRESET_DIR / f"{slugify(slug)}.json"
    if not path.is_file():
        raise KeyError(f"no camera preset '{slug}'")
    path.unlink()
    thumb = path.with_suffix(".jpg")
    if thumb.is_file():
        thumb.unlink()


def suggest(presets: list, now: Optional[datetime] = None) -> Optional[str]:
    """The newest preset whose condition matches the time of day (day 07:00-18:00). Only a hint:
    the preset is always chosen by hand."""
    now = now or datetime.now()
    want = "day" if 7 <= now.hour < 18 else "night"
    for d in presets:
        if d.get("condition") == want and not d.get("is_default"):
            return d["slug"]
    return None
