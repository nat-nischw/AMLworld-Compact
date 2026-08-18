#!/usr/bin/env python3
"""Stage 23. The LLM predictions under HT weighting, and the predict-all floor.

Thin wrapper. The stage lives in :mod:`amlc.analysis.llm_ht`; this exists so
the pipeline order is visible from `scripts/`. Needs AMLC_ARCHIVE.

    python scripts/23_score_llm_ht.py
    python scripts/23_score_llm_ht.py --dataset HI-Small

Writes ``results/analysis/llm_ht_weighted.csv`` (one row per model, prompting
and seed, both framings side by side) and ``llm_ht_weighted_summary.csv`` (the
spans quoted in Tables 2 and 14).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.analysis.llm_ht import main

if __name__ == "__main__":
    main()
