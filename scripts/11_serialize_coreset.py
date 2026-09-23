#!/usr/bin/env python3
"""Serialise the coreset into typed-graph prompts.

Thin wrapper. The stage lives in :mod:`amlc.serialize.build_coreset_prompts`; this exists so the pipeline
order is visible from `scripts/`. Arguments are passed straight through.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.serialize.build_coreset_prompts import main

if __name__ == "__main__":
    main()
