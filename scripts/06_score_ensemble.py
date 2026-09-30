#!/usr/bin/env python3
"""Score the reported two-booster ensemble at fixed thresholds.

Thin wrapper. The stage lives in :mod:`amlc.baselines.ml.ensemble`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.baselines.ml.ensemble import main

if __name__ == "__main__":
    main()
