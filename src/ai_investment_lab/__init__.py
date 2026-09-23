"""AI Investment Lab milestone 0.1."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("LUMIBOT_CACHE_FOLDER", str(_PROJECT_ROOT / ".cache" / "lumibot"))

__version__ = "0.1.0"
