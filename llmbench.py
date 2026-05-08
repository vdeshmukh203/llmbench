#!/usr/bin/env python3
"""
Standalone entry-point for llmbench.

Run directly without installation::

    python llmbench.py [command] [options]

Or install the package for the ``llmbench`` console script::

    pip install .
    llmbench [command] [options]
"""
import sys
from pathlib import Path

# Make the src/ package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llmbench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
