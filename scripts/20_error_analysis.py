#!/usr/bin/env python3
"""Per-typology and error-transition analysis on the frozen coreset.

Compare the two temporal-training boosters, pooled over five seeds, with
the available seed-42 LLM diagnostics. The original construction scores
remain attached to the frozen target selection and importance weights.

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
