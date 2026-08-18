#!/usr/bin/env python3
"""Re-infer GCPAL on the temporal test split.

Thin wrapper. The stage lives in :mod:`amlc.baselines.ml.gcpal_infer`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.baselines.ml.gcpal_infer import main

if __name__ == "__main__":
    main()
