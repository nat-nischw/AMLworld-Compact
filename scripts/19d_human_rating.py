#!/usr/bin/env python3
"""The three-rater human validation tables.

Thin wrapper. The stage lives in :mod:`amlc.audit.human_rating`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.audit.human_rating import main

if __name__ == "__main__":
    main()
