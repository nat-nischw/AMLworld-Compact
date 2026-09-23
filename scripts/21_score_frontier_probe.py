#!/usr/bin/env python3
"""Stage 21. Score the frontier-API context-engineering probe.

Reproduces the eighteen cells of the appendix probe table from
``results/frontier_probe/predictions.csv`` and checks every published
number against it. Needs nothing but the repository.

    python scripts/21_score_frontier_probe.py

The stage lives in :mod:`amlc.audit.frontier_probe`; this exists so the
pipeline reads as a numbered sequence. Pass ``--extract`` only if you hold the
raw API responses, which the release cannot redistribute; see that module for
the scoring conventions.
"""
from amlc.audit.frontier_probe import main

if __name__ == "__main__":
    main()
