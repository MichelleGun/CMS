"""pytest bootstrap: put hardware/ and CMS/scripts on sys.path so tests import like the scripts do."""
import sys
from pathlib import Path

HW = Path(__file__).resolve().parent
for p in (HW, HW.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
