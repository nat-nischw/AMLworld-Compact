#!/usr/bin/env python3
"""Describe cached LLM outputs by presence of the intended target transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from amlc.analysis.target_presence import align_presence, summarize_presence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=ROOT / "results/analysis/sampling_uncertainty/inputs")
    parser.add_argument("--cases", type=Path,
                        default=ROOT / "results/analysis/target_integrity/cases.csv")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/analysis/target_presence")
    args = parser.parse_args()
    cases = pd.read_csv(args.cases)
    per_run, summaries, hashes = [], [], []
    paths = [args.cases]
    for dataset in ["HI-Small", "LI-Small"]:
        path = args.inputs / f"{dataset}.npz"
        paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            present = align_presence(cases[cases.dataset == dataset], data["case_ids"],
                                     data["center_edge_ids"])
            runs, summary = summarize_presence(
                data["predictions"], data["labels"], present, data["models"],
                data["promptings"], data["seeds"])
        for frame, target in [(runs, per_run), (summary, summaries)]:
            frame.insert(0, "dataset", dataset)
            target.append(frame)
    for path in paths:
        hashes.append({"file": path.name,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(per_run, ignore_index=True).to_csv(args.out / "per_run.csv", index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "inputs": hashes, "metric_units": "fractions",
        "scoring": "Frozen binary predictions; all requests remain in denominators.",
        "scope": "Unweighted subset cohorts; present targets remain unmarked. "
                 "These observational rates are not a corrected-prompt evaluation.",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
