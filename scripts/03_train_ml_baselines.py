#!/usr/bin/env python3
"""Train LightGBM+GFP and XGBoost+GFP over 5 seeds.

Thin wrapper. The stage lives in :mod:`amlc.llm.runner`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.llm.runner import main

if __name__ == "__main__":
    # This stage is the supervised mode of the shared runner.
    sys.argv[1:1] = ['--mode', 'supervised']
    main()
