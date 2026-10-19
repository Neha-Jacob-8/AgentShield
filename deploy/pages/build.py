"""
Build the GitHub Pages site: the console page in static mode plus a recording of the real models.

  python -m src.defense.api --record deploy/pages/recording.json --public   # re-record (needs the models)
  python deploy/pages/build.py [OUT]                                         # writes _site/ (stdlib only)

The page plays the recording in the browser (no server, no models), so uploads are not available there.
The recording holds only the samples in demo_samples/. .github/workflows/pages.yml runs this on every push.
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARKER = "<!-- static recording -->"


def build(out: Path) -> Path:
    page = (ROOT / "src" / "defense" / "demo_ui" / "index.html").read_text(encoding="utf-8")
    if page.count(MARKER) != 1:
        raise SystemExit(f"the console page must contain {MARKER!r} once")
    rec = json.loads((HERE / "recording.json").read_text(encoding="utf-8"))
    if not rec.get("steps") or not rec.get("samples"):
        raise SystemExit("deploy/pages/recording.json has no steps or samples: re-record it")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page.replace(MARKER, '<script>window.AGENTSHIELD_RECORDING = "recording.json";</script>'),
                                    encoding="utf-8")
    shutil.copy(HERE / "recording.json", out / "recording.json")
    (out / ".nojekyll").touch()
    return out


if __name__ == "__main__":
    print("built", build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_site"))
