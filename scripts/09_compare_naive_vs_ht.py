#!/usr/bin/env python3
"""Naive versus HT-Coreset, per model.

Thin wrapper. The stage lives in :mod:`amlc.coreset.compare`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.coreset.compare import main

if __name__ == "__main__":
    main()
