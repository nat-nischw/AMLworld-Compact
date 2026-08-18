"""ICL-V against ICL-FS: did the verification step reduce over-prediction?

ICL-V is ICL-FS with a verification step inserted before the answer format, and
it was run as a pre-registered test of one claim: that the models' 98% recall
and 34% precision come from asserting typologies they have not checked, so
making them check should cut over-prediction without costing recall.

The three hypotheses, as registered before the run:

    H1  primary    dF1 at least +5 pp on HI-Small
    H2  mechanism  over-prediction rate falls by at least 10 pp
    H3  guard      under-prediction rate rises by no more than 5 pp

Both raw and HT-weighted detection metrics are reported. The raw ones are what
Table 3 quotes for LLMs, where the coreset's 1:2 design gives an interpretable
noise floor; the weighted ones are the population view, and are the ones to
compare against Doubt Triage.

Failure taxonomy, per case, unweighted: correct, over-prediction (flagged a
benign edge), under-prediction (missed an illicit one), typology error (found
it, named it wrong) and format failure (found it, named nothing). Format
failure is separated from typology error because it is a parsing problem, not a
reasoning one, and the verification step was expected to move only the second.

Reads
    The archived run directory: the coreset with its HT weights and typologies,
    and two prediction JSONs per cell, ICL-FS as control and ICL-V as
    intervention.

Writes
    ``results/audit/intervention_results.csv``, one row per (model, dataset,
    seed), and prints the hypothesis check per dataset.

ICL-V exists for two models at seed 42 (GPT-OSS-120B and Nemotron-3-Super-120B),
so most cells report as missing. The weights are the repaired HT weights, which
is what :func:`amlc.triage.doubt_triage.load_coreset_from_archive` returns
by default; the archived LI-Small vector summed to twice the population.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..triage.doubt_triage import (
    ht_weighted_prf,
    load_coreset_from_archive,
    load_llm_predictions,
    resolve_archive,
)

#: The prompting condition ICL-V is measured against.
CONTROL = "ICL-FS"
INTERVENTION = "ICL-V"

#: Pre-registered thresholds, in percentage points.
H1_MIN_DELTA_F1 = 5.0
H2_MAX_DELTA_OVER = -10.0
H3_MAX_DELTA_UNDER = 5.0


def raw_prf(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Unweighted precision, recall, F1 and the confusion counts."""
    preds, labels = np.asarray(preds).astype(int), np.asarray(labels).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def missing_typology(value) -> bool:
    """True when a flagged case carries no usable typology label.

    The typology comes from the shared extractor in
    :mod:`amlc.triage.doubt_triage`, which recovers the Unicode
    non-breaking hyphen GPT-OSS emits inside compound names and marks what it
    still cannot read as ``unknown``. The pre-release script read the raw field
    and tested only for ``None``, so every name the ASCII parser missed counted
    as a format failure. Those now count as whatever they are.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip() in ("", "unknown")


def failure_taxonomy(preds: np.ndarray, typologies, labels: np.ndarray,
                     gt_typologies) -> dict:
    """Share of cases in each failure mode, as percentages."""
    n = len(labels)
    counts = {"correct": 0, "over": 0, "under": 0, "typology": 0, "format": 0}
    for pred, typology, label, gt_typology in zip(preds, typologies, labels,
                                                  gt_typologies):
        if label == 1 and pred == 0:
            counts["under"] += 1
        elif label == 0 and pred == 1:
            counts["over"] += 1
        elif label == 1 and pred == 1:
            if missing_typology(typology):
                counts["format"] += 1
            elif typology == gt_typology:
                counts["correct"] += 1
            else:
                counts["typology"] += 1
        elif label == 0 and pred == 0:
            counts["correct"] += 1
    return {
        "correct_pct": 100 * counts["correct"] / n,
        "over_pred_pct": 100 * counts["over"] / n,
        "under_pred_pct": 100 * counts["under"] / n,
        "typology_err_pct": 100 * counts["typology"] / n,
        "format_fail_pct": 100 * counts["format"] / n,
    }


def evaluate_run(preds: np.ndarray, typologies, data: dict) -> dict:
    """Raw metrics, HT-weighted metrics and the taxonomy for one run."""
    raw = raw_prf(preds, data["labels"])
    wp, wr, wf1 = ht_weighted_prf(np.asarray(preds).astype(int),
                                  data["labels"], data["weights"])
    taxonomy = failure_taxonomy(preds, typologies, data["labels"],
                                data["gt_typo_str"])
    return {**raw, "ht_precision": wp, "ht_recall": wr, "ht_f1": wf1, **taxonomy}


def compare_one(archive: Path, data: dict, dataset: str, model: str,
                seed: int) -> dict:
    """One (model, dataset, seed) cell: intervention minus control."""
    control = load_llm_predictions(archive, dataset, model, CONTROL, seed, data["n"])
    intervention = load_llm_predictions(archive, dataset, model, INTERVENTION, seed,
                                        data["n"])
    if control is None or intervention is None:
        missing = CONTROL if control is None else INTERVENTION
        return {"model": model, "dataset": dataset, "seed": seed,
                "status": f"missing {missing} run"}

    c = evaluate_run(control[0], control[1], data)
    v = evaluate_run(intervention[0], intervention[1], data)

    row = {"model": model, "dataset": dataset, "seed": seed, "status": "ok"}
    for metric in ("f1", "precision", "recall", "ht_f1",
                   "over_pred_pct", "under_pred_pct", "format_fail_pct",
                   "typology_err_pct"):
        row[f"control_{metric}"] = c[metric]
        row[f"icl_v_{metric}"] = v[metric]
        row[f"delta_{metric}"] = v[metric] - c[metric]
    return row


def check_hypotheses(results: pd.DataFrame) -> pd.DataFrame:
    """Mean deltas per dataset against the three pre-registered thresholds."""
    ok = results[results.status == "ok"]
    rows = []
    for dataset, group in ok.groupby("dataset"):
        d_f1 = group.delta_f1.mean() * 100
        d_over = group.delta_over_pred_pct.mean()
        d_under = group.delta_under_pred_pct.mean()
        rows.append({
            "dataset": dataset, "n_cells": len(group),
            "delta_f1_pp": d_f1, "H1_pass": d_f1 >= H1_MIN_DELTA_F1,
            "delta_over_pred_pp": d_over, "H2_pass": d_over <= H2_MAX_DELTA_OVER,
            "delta_under_pred_pp": d_under, "H3_pass": d_under <= H3_MAX_DELTA_UNDER,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path, default=None,
                    help="archived experiment tree (default: $AMLC_ARCHIVE)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--models", nargs="+", default=list(config.LLM_MODELS),
                    choices=list(config.LLM_MODELS))
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS),
                    choices=list(config.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[config.SEEDS[0]])
    args = ap.parse_args()

    from ..paths import ensure, results

    archive = resolve_archive(args.archive)
    rows = []
    for dataset in args.datasets:
        data = load_coreset_from_archive(archive, dataset)
        for model in args.models:
            for seed in args.seeds:
                row = compare_one(archive, data, dataset, model, seed)
                rows.append(row)
                if row["status"] != "ok":
                    print(f"  {model:24s} | {dataset:9s} | seed {seed:4d} | "
                          f"{row['status']}")
                    continue
                print(f"  {model:24s} | {dataset:9s} | seed {seed:4d} | "
                      f"dF1={row['delta_f1'] * 100:+6.2f}pp | "
                      f"dover={row['delta_over_pred_pct']:+6.2f}pp | "
                      f"dunder={row['delta_under_pred_pct']:+6.2f}pp")

    frame = pd.DataFrame(rows)
    path = args.out or ensure(results() / "audit") / "intervention_results.csv"
    frame.to_csv(path, index=False)
    print(f"\nSaved {path} ({len(frame)} rows)")

    checks = check_hypotheses(frame)
    if checks.empty:
        print("\nNo complete cells: ICL-V predictions were not found.")
        return
    print("\nPre-registered hypotheses")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
