#!/usr/bin/env python3
"""Draw the 1,000-trace rubric sample and run the regex annotator.

Thin wrapper. The stage lives in :mod:`amlc.audit.sample_traces`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.audit.sample_traces import main

if __name__ == "__main__":
    main()
