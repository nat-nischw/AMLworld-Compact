#!/usr/bin/env python3
"""Stage 17b. Paired DT-versus-ML statistics.

Thin wrapper. The stage lives in :mod:`amlc.triage.dt_stats`; this exists so the
pipeline order is visible from `scripts/`. Arguments are passed straight through.

    python scripts/17b_dt_stats.py --results <dir>/dt_results_ht.csv --out <dir>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.triage.dt_stats import main

if __name__ == "__main__":
    main()
