#!/usr/bin/env python3
"""Optuna search. Optional: data/tuned_params/ already holds the results.

Thin wrapper. The stage lives in :mod:`amlc.baselines.ml.tuning`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.baselines.ml.tuning import main

if __name__ == "__main__":
    main()
