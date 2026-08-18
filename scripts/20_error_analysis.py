#!/usr/bin/env python3
"""Per-typology and error-transition analysis.

Thin wrapper. The stage lives in :mod:`amlc.analysis.error_analysis`.

It takes no arguments: the paths come from the environment through
`amlc.paths`, so set AMLC_ARCHIVE before running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.analysis.error_analysis import main

if __name__ == "__main__":
    main()
