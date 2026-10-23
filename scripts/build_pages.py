"""
Pack the dashboard (webapp/frontend) as the online copy for GitHub Pages. Run by
.github/workflows/pages.yml on every push; you can run it locally to look at the result:

    python scripts/build_pages.py _site
    python -m http.server 8002 --directory _site

The page is the same one the local server serves; only js/mode.js changes (DQ_STATIC = true),
so the tabs read the snapshot in data/ (scripts/build_site.py) instead of a server. The 3D cage
model and the "How it works" page are copied in here rather than kept twice in the repo.
Standard library only: the workflow needs no installs.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "webapp" / "frontend"


def build(out):
    out = Path(out).resolve()
    if out == FRONT or FRONT in out.parents:
        sys.exit("build_pages: pick an output folder outside webapp/frontend")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(FRONT, out, dirs_exist_ok=True)     # (a folder a local server is showing stays)
    mode = out / "js" / "mode.js"
    text = mode.read_text(encoding="utf-8")
    assert "window.DQ_STATIC = false;" in text, "js/mode.js changed: update build_pages.py"
    mode.write_text(text.replace("window.DQ_STATIC = false;", "window.DQ_STATIC = true;"), encoding="utf-8")
    (out / "assets").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "assets" / "3D_net&hoop.glb", out / "assets" / "3D_net&hoop.glb")
    shutil.copy2(ROOT / "docs" / "Onboard Hoop Policy Anatomy.html", out / "anatomy.html")
    (out / ".nojekyll").write_text("", encoding="utf-8")     # serve files as they are
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[pages] {out}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else ROOT / "_site")
