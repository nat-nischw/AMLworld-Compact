#!/usr/bin/env python3
"""Build and cache the Graph-Feature-Preprocessor tensors for one or both splits.

This is the slow first stage: it streams every AMLworld edge through IBM Snap
ML's GraphFeaturePreprocessor in temporal order and caches the result, so the
supervised baselines and the coreset can start from features rather than CSVs.
Budget two to three hours per split.

Requires the raw AMLworld files, so run `00_download_amlworld.py` first.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc import config
from amlc.data.gfp import load_amlworld_for_snapml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS),
                    choices=config.DATASETS)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="streaming batch. 128 preserves temporal causality; "
                         "a value <= 0 fits the preprocessor on the whole edge "
                         "set at once and silently destroys it")
    args = ap.parse_args()

    for ds in args.datasets:
        out = load_amlworld_for_snapml(ds, gfp_batch_size=args.batch_size)
        n, d = out["X_test"].shape
        print(f"{ds}: cached, test split {n:,} x {d} features, "
              f"{out['gfp_time']:.0f}s in the preprocessor")


if __name__ == "__main__":
    main()
