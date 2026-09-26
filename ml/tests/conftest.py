"""pytest-конфигурация ML: путь к исходникам."""
from __future__ import annotations

import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / "src"))
