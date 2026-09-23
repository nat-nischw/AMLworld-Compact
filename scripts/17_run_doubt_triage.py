#!/usr/bin/env python3
"""Run Doubt Triage across every (dataset, model, prompting, strategy, seed) cell.

Four selective-deferral rules (see amlc.triage.doubt_triage.STRATEGIES):

- ``dt``          defer to the LLM where the ensemble says legit, inside the
                  census stratum. This is the paper's Doubt Triage.
- ``confidence``  defer where the ensemble is near its threshold.
- ``disagree``    defer only where ML and LLM disagree.
- ``union``       ``dt`` OR ``confidence``.

Usage:
    python scripts/17_run_doubt_triage.py                        # all cells, released draw
    python scripts/17_run_doubt_triage.py --dataset HI-Small
    python scripts/17_run_doubt_triage.py --strategy dt

The evaluation scorer uses the two temporal-trained boosters. The frozen
construction draw and its repaired design weights are unchanged.

Outputs ``dt_results.csv`` (per seed) and ``dt_summary.csv`` (mean over seeds).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.config import CORESET_DRAWS, PROMPTINGS
from amlc.config import DATASETS, SEEDS
from amlc.paths import results
from amlc.config import LLM_MODELS as CONFIG_LLM_MODELS
from amlc.triage.doubt_triage import (
    STRATEGIES,
    DoubtTriage,
    ht_weighted_prf,
    load_coreset_from_archive,
    load_ensemble_typology,
    load_llm_predictions,
    resolve_archive,
    verify_alignment,
)

# Grouped by family rather than by size. This is the row order of the shipped
# results/triage/dt_results.csv, so changing it changes that file.
# config.LLM_MODELS holds the same seven in the paper's by-size order; the
# assertion keeps the two from drifting apart.
LLM_MODELS = [
    "GPT-OSS-120B", "GPT-OSS-20B",
    "Nemotron-3-Super-120B", "Nemotron-3-Nano-30B",
    "Qwen3.5-397B-A17B", "Qwen3.5-35B-A3B",
    "Qwen3.5-27B",
]
if set(LLM_MODELS) != set(CONFIG_LLM_MODELS):
    raise SystemExit(
        "LLM_MODELS here must be a reordering of config.LLM_MODELS, not a "
        f"different set. Only here: {sorted(set(LLM_MODELS) - set(CONFIG_LLM_MODELS))}; "
        f"only in config: {sorted(set(CONFIG_LLM_MODELS) - set(LLM_MODELS))}"
    )

# Every prompting except ICL-V, which exists for only 2 models at seed 42 and is
# therefore opt-in via --prompting.
DEFAULT_PROMPTINGS = [p for p in PROMPTINGS if p != "ICL-V"]


def run_cell(archive, data, dataset, model, prompting, strategy, confidence_delta,
             ml_prf, verbose=True):
    """Run one (dataset, model, prompting, strategy) cell across all seeds."""
    dt = DoubtTriage(
        ml_probs=data["ml_probs"],
        weights=data["weights"],
        ml_threshold=data["ml_threshold"],
        strategy=strategy,
        confidence_delta=confidence_delta,
    )
    ml_p, ml_r, ml_f1 = ml_prf

    rows = []
    for seed in SEEDS:
        loaded = load_llm_predictions(archive, dataset, model, prompting, seed, data["n"])
        if loaded is None:
            continue
        llm_bin, llm_typ, tok = loaded
        ml_typ = load_ensemble_typology(archive, dataset, data["subset_idx"], seed)

        res = dt.evaluate(
            labels=data["labels"],
            llm_preds=llm_bin,
            llm_typologies=llm_typ,
            ml_typologies=ml_typ,
            gt_typologies=data["gt_typo_str"],
        )

        rows.append({
            "draw": data["draw"],
            "weights_repaired": data.get("weights_repaired", True),
            "dataset": dataset,
            "llm_model": model,
            "prompting": prompting,
            "strategy": strategy,
            "confidence_delta": confidence_delta if strategy in ("confidence", "union") else np.nan,
            "seed": seed,
            "ml_threshold": data["ml_threshold"],
            "evaluation_members": ";".join(data["evaluation_members"]),
            "ml_f1": ml_f1,
            "ml_precision": ml_p,
            "ml_recall": ml_r,
            "dt_f1": res.w_f1,
            "dt_precision": res.w_precision,
            "dt_recall": res.w_recall,
            "dt_typ_f1": res.w_typ_macro_f1,
            "dt_typ_acc": res.w_typ_accuracy,
            "delta_f1": res.w_f1 - ml_f1,
            "n_consult": res.n_consult,
            "consult_pct": res.consult_ratio * 100,
            "n_cases_matched": tok.get("n_cases_matched", 0),
            "avg_input_tokens": tok.get("avg_input_tokens", 0) * res.consult_ratio,
            "avg_output_tokens": tok.get("avg_output_tokens", 0) * res.consult_ratio,
        })

    if verbose and rows:
        f1s = [r["dt_f1"] for r in rows]
        print(f"  {dataset:10s} | {model:24s} | {prompting:7s} | "
              f"{strategy:10s} | F1={np.mean(f1s)*100:5.1f}+/-{np.std(f1s)*100:.1f} | "
              f"d={np.mean([r['delta_f1'] for r in rows])*100:+6.2f} | "
              f"consult={np.mean([r['consult_pct'] for r in rows]):.1f}% | n={len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=None,
                    help="archived experiment tree (default: $AMLC_ARCHIVE)")
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--draw", choices=CORESET_DRAWS, default="ht-coreset",
                    help="ht-coreset = the released draw (default); "
                         "ablation-redraw = the same-size re-draw behind the "
                         "published ablation numbers")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the draw/serialisation alignment check "
                         "(required for --draw ablation-redraw)")
    ap.add_argument("--archived-weights", action="store_true",
                    help="use the archived HT weights instead of the repaired "
                         "ones; the archived LI-Small vector sums to 2x the "
                         "population (see coreset/ht_weights.py)")
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--model", choices=LLM_MODELS)
    ap.add_argument("--prompting", choices=PROMPTINGS)
    ap.add_argument("--strategy", choices=list(STRATEGIES))
    ap.add_argument("--confidence-delta", type=float, default=0.10)
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    args = ap.parse_args()

    archive = resolve_archive(args.archive)
    out_dir = args.out or (results() / "triage")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else DATASETS
    models = [args.model] if args.model else LLM_MODELS
    promptings = [args.prompting] if args.prompting else DEFAULT_PROMPTINGS
    strategies = [args.strategy] if args.strategy else list(STRATEGIES)

    all_rows = []
    for ds in datasets:
        data = load_coreset_from_archive(archive, ds, draw=args.draw,
                                         verify=not args.no_verify,
                                         repair_weights=not args.archived_weights)
        rep = verify_alignment(archive, ds, args.draw, strict=False)
        ml_preds = (data["ml_probs"] >= data["ml_threshold"]).astype(int)
        ml_prf = ht_weighted_prf(ml_preds, data["labels"], data["weights"])

        print(f"\n{'='*104}")
        print(f"  {ds} | draw={args.draw} | n={data['n']} | "
              f"weights={'repaired' if data['weights_repaired'] else 'archived'} | "
              f"threshold={data['ml_threshold']} | ensemble-only HT F1={ml_prf[2]*100:.1f}")
        if rep.get("checked"):
            status = "aligned" if rep["labels_match"] and rep["weights_match"] else \
                     f"MISALIGNED, {rep['label_mismatch_frac']:.1%} of labels differ"
            print(f"  alignment vs serialised prompts: {status}")
        print(f"{'='*104}")

        for model in models:
            for prompting in promptings:
                for strategy in strategies:
                    all_rows += run_cell(archive, data, ds, model, prompting, strategy,
                                         args.confidence_delta, ml_prf)

    if not all_rows:
        print("No results. Check that the LLM prediction files exist.")
        return

    suffix = f"_{args.tag}" if args.tag else ""
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / f"dt_results{suffix}.csv", index=False)

    agg = df.groupby(["draw", "dataset", "llm_model", "prompting", "strategy"]).agg(
        n_seeds=("seed", "count"),
        evaluation_members=("evaluation_members", "first"),
        ml_threshold=("ml_threshold", "first"),
        ml_f1=("ml_f1", "first"),
        dt_f1_mean=("dt_f1", "mean"), dt_f1_std=("dt_f1", "std"),
        dt_p_mean=("dt_precision", "mean"), dt_p_std=("dt_precision", "std"),
        dt_r_mean=("dt_recall", "mean"), dt_r_std=("dt_recall", "std"),
        dt_tf1_mean=("dt_typ_f1", "mean"), dt_tf1_std=("dt_typ_f1", "std"),
        dt_tacc_mean=("dt_typ_acc", "mean"),
        delta_f1_mean=("delta_f1", "mean"),
        avg_in=("avg_input_tokens", "mean"), avg_out=("avg_output_tokens", "mean"),
        consult_pct=("consult_pct", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / f"dt_summary{suffix}.csv", index=False)

    print(f"\nSaved {out_dir / f'dt_results{suffix}.csv'} ({len(df)} rows)")
    print(f"Saved {out_dir / f'dt_summary{suffix}.csv'} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
