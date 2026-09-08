#!/usr/bin/env python3
"""Stage 23. The LLM predictions under HT weighting, and the predict-all floor.

Thin wrapper. The stage lives in :mod:`amlc.analysis.llm_ht`; this exists so
the pipeline order is visible from `scripts/`.

    python scripts/23_score_llm_ht.py --runs-dir runs/demo \\
        --models GPT-OSS-20B --promptings ICL-ZS --seeds 42 \\
        --dataset HI-Small --out runs/demo/evaluation
    python scripts/23_score_llm_ht.py  # original AMLC_ARCHIVE layout

Writes ``results/analysis/llm_ht_weighted.csv`` (one row per model, prompting
and seed, HT and compact metrics side by side) and
``llm_ht_weighted_summary.csv`` (min/max over selected cells' seed means).
``--out`` overrides that output directory. All metrics are percentages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.analysis.llm_ht import main

if __name__ == "__main__":
    main()
