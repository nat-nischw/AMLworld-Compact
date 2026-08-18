#!/usr/bin/env python3
"""The seven sampling baselines, B1 to B7.

Thin wrapper. The stage lives in :mod:`amlc.coreset.ablation`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.coreset.ablation import main

if __name__ == "__main__":
    main()
