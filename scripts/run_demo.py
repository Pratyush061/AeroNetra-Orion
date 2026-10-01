#!/usr/bin/env python3
"""Run the download-free synthetic demo.

    python scripts/run_demo.py [--config configs/default.yaml] [--out outputs]
"""

from __future__ import annotations

import sys

from orion.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
