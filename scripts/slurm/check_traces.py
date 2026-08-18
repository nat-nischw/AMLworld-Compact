#!/usr/bin/env python3
"""Check reasoning-trace coverage in the LLM prediction outputs.

Usage:
    python scripts/slurm/check_traces.py                       # every model
    python scripts/slurm/check_traces.py --model GPT-OSS-20B   # one model
    python scripts/slurm/check_traces.py --verbose             # show example traces
"""

import argparse
import json
from pathlib import Path

from amlc import archive
from amlc.config import DATASETS, LLM_MODELS, PROMPTINGS

OUTPUT_DIR = Path("outputs")

# Every evaluated model. This used to be a hand-kept list of the six served on a
# 4-GPU node, which left Qwen3.5-397B-A17B out of the default sweep, so a run
# with no --model silently reported nothing about it.
MODELS = list(LLM_MODELS)

# ICL-V is excluded: it was run for 2 models at one seed, so it has no coverage
# to check across the grid.
METHODS = [archive.prompting_dir(p) for p in PROMPTINGS if p != "ICL-V"]


def check_model(model_name: str, verbose: bool = False):
    """Check reasoning trace coverage for a model."""
    model_dir = OUTPUT_DIR / model_name
    if not model_dir.exists():
        print(f"  [SKIP] {model_name}: no output directory")
        return

    total_preds = 0
    total_with_trace = 0
    trace_lengths = []
    examples = []

    for method in METHODS:
        for dataset in DATASETS:
            files = sorted(model_dir.glob(f"{method}/{dataset}/seed_*.json"))

            for f in files:
                seed = f.stem  # e.g., seed_42
                try:
                    data = json.loads(f.read_text())
                except Exception as e:
                    print(f"  [ERROR] {f}: {e}")
                    continue

                # Handle both list and dict formats
                predictions = data if isinstance(data, list) else data.get("predictions", [])

                n_preds = len(predictions)
                n_with_trace = 0

                for pred in predictions:
                    llm_raw = pred.get("llm_raw_response", {})
                    reasoning = None

                    if isinstance(llm_raw, dict):
                        reasoning = llm_raw.get("reasoning")

                    # Also check reasoning_content field
                    if not reasoning:
                        reasoning = pred.get("reasoning_content")

                    if reasoning and len(str(reasoning).strip()) > 10:
                        n_with_trace += 1
                        trace_lengths.append(len(str(reasoning)))

                        if verbose and len(examples) < 3:
                            examples.append({
                                "model": model_name,
                                "method": method,
                                "dataset": dataset,
                                "seed": seed,
                                "case_id": pred.get("case_id", pred.get("center_edge_id", "?")),
                                "trace_len": len(str(reasoning)),
                                "trace_preview": str(reasoning)[:500],
                            })

                total_preds += n_preds
                total_with_trace += n_with_trace

                pct = (n_with_trace / n_preds * 100) if n_preds > 0 else 0
                status = "OK" if pct > 50 else "LOW" if pct > 0 else "NONE"
                print(f"  [{status:4s}] {method}/{dataset}/{seed}: "
                      f"{n_with_trace}/{n_preds} traces ({pct:.1f}%)")

    # Summary
    pct_total = (total_with_trace / total_preds * 100) if total_preds > 0 else 0
    avg_len = (sum(trace_lengths) / len(trace_lengths)) if trace_lengths else 0

    print(f"\n  SUMMARY: {total_with_trace}/{total_preds} predictions with traces "
          f"({pct_total:.1f}%)")
    if trace_lengths:
        print(f"  Trace length: mean={avg_len:.0f} chars, "
              f"min={min(trace_lengths)}, max={max(trace_lengths)}")

    if verbose and examples:
        print("\n  === Example traces ===")
        for ex in examples:
            print(f"\n  [{ex['model']}] {ex['method']}/{ex['dataset']}/{ex['seed']} "
                  f"case={ex['case_id']} ({ex['trace_len']} chars)")
            print(f"  {ex['trace_preview']}")
            print("  ...")

    return {
        "model": model_name,
        "total": total_preds,
        "with_trace": total_with_trace,
        "pct": pct_total,
        "avg_trace_len": avg_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Check reasoning trace coverage")
    parser.add_argument("--model", type=str, help="Specific model to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show example traces")
    parser.add_argument("--all", action="store_true", help="Check all models (not just 4-GPU)")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    elif args.all:
        models = sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()])
    else:
        models = MODELS

    print("=" * 60)
    print("Reasoning Trace Coverage Report")
    print("=" * 60)

    results = []
    for model in models:
        print(f"\n{model}:")
        r = check_model(model, verbose=args.verbose)
        if r:
            results.append(r)

    # Final summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Traces':>10} {'Total':>10} {'Coverage':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<25} {r['with_trace']:>10} {r['total']:>10} "
              f"{r['pct']:>9.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
