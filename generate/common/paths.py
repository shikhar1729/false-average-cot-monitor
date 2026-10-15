"""Where things live. Run every generator from the repository root as a module, e.g.
`python -m generate.attack.headline`."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MONITORING = ROOT / "monitoring"                       # the dataset authors' monitor.py, unmodified
TW = Path(os.environ.get("TW_REPO", ROOT / "repo"))    # git clone of github.com/few-sh/terminal-wrench
