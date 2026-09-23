#!/usr/bin/env python3
"""Train the GCPAL graph baseline.

Thin wrapper. The stage lives in :mod:`amlc.baselines.ml.gcpal`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.baselines.ml.gcpal import main

if __name__ == "__main__":
    main()
