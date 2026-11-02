"""Run project commands directly without installing the local package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from uob_regime.cli import main

if __name__ == "__main__":
    main()
