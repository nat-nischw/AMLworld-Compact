"""Score the LLM predictions under Horvitz-Thompson weighting.

Tables 2 and 14 report LLM detection on the coreset unweighted, against the
1:2 design ratio. That is the right frame for comparing one LLM to another,
because the predict-all-illicit floor sits at a legible 50 percent F1, but it
is the wrong frame for asking what an LLM would score on the full split. This
module answers the second question, for the LLM rows and for the predict-all
floor, using the same weights and the same estimator the supervised baselines
are scored with.

The arithmetic is not new: :func:`amlc.triage.doubt_triage.ht_weighted_prf`
does the weighting, and this module only assembles predictions and reports.
Recall is invariant to the weights on this coreset, because every illicit edge
is retained at weight one, so the whole effect lands on precision.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DATASETS, LLM_MODELS, SEEDS
from ..triage.doubt_triage import (
    ht_weighted_prf,
    load_coreset_from_archive,
    load_llm_predictions,
    resolve_archive,
)

__all__ = ["score_cell", "predict_all_floor", "score_dataset", "main"]


def _unweighted_prf(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Plain P/R/F1, i.e. every weight set to one."""
    return ht_weighted_prf(preds, labels, np.ones_like(labels, dtype=float))


def score_cell(
    archive: Path,
    data: dict,
    dataset: str,
    model: str,
    prompting: str,
    seeds=SEEDS,
) -> list[dict]:
    """Both framings of one (dataset, model, prompting) cell, per seed."""
    rows = []
    for seed in seeds:
        loaded = load_llm_predictions(archive, dataset, model, prompting, seed, data["n"])
        if loaded is None:
            continue
        llm_preds, _, _ = loaded
        p_u, r_u, f1_u = _unweighted_prf(llm_preds, data["labels"])
        p_w, r_w, f1_w = ht_weighted_prf(llm_preds, data["labels"], data["weights"])
        rows.append({
            "dataset": dataset,
            "model": model,
            "prompting": prompting,
            "seed": seed,
            "n_flagged": int(llm_preds.sum()),
            "subset_p": 100 * p_u,
            "subset_r": 100 * r_u,
            "subset_f1": 100 * f1_u,
            "ht_p": 100 * p_w,
            "ht_r": 100 * r_w,
            "ht_f1": 100 * f1_w,
        })
    return rows


def predict_all_floor(data: dict, dataset: str) -> dict:
    """The predict-all-illicit baseline under both framings.

    Under the 1:2 design ratio this is the noise floor the paper quotes for
    Tables 2 and 14. Under HT weighting it is the same rule scored against the
    full split, where precision falls to the population illicit rate.
    """
    preds = np.ones(data["n"], dtype=int)
    p_u, r_u, f1_u = _unweighted_prf(preds, data["labels"])
    p_w, r_w, f1_w = ht_weighted_prf(preds, data["labels"], data["weights"])
    return {
        "dataset": dataset,
        "model": "predict-all-illicit",
        "prompting": "—",
        "seed": -1,
        "n_flagged": data["n"],
        "subset_p": 100 * p_u,
        "subset_r": 100 * r_u,
        "subset_f1": 100 * f1_u,
        "ht_p": 100 * p_w,
        "ht_r": 100 * r_w,
        "ht_f1": 100 * f1_w,
    }


def score_dataset(
    dataset: str,
    archive: Optional[Path] = None,
    models=LLM_MODELS,
    promptings=("ICL-FS", "ICL-ZS"),
    verify: bool = True,
) -> list[dict]:
    """Every LLM cell on one dataset, plus the predict-all floor."""
    archive = resolve_archive(archive)
    data = load_coreset_from_archive(archive, dataset, verify=verify)
    rows = [predict_all_floor(data, dataset)]
    for model in models:
        for prompting in promptings:
            rows.extend(score_cell(archive, data, dataset, model, prompting))
    return rows


def main(argv=None):
    from .. import paths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, action="append")
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = args.out or (paths.results() / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in (args.dataset or list(DATASETS)):
        rows.extend(score_dataset(dataset, verify=args.verify))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "llm_ht_weighted.csv", index=False)

    # The span the paper quotes, over the 14 (model, prompting) cells, on the
    # seed-mean of each cell.
    cells = df[df.model != "predict-all-illicit"]
    per_cell = cells.groupby(["dataset", "model", "prompting"]).mean(numeric_only=True)
    summary = []
    for dataset, block in per_cell.groupby(level=0):
        floor = df[(df.dataset == dataset) & (df.model == "predict-all-illicit")].iloc[0]
        summary.append({
            "dataset": dataset,
            "n_cells": len(block),
            "subset_f1_lo": block.subset_f1.min(), "subset_f1_hi": block.subset_f1.max(),
            "ht_f1_lo": block.ht_f1.min(), "ht_f1_hi": block.ht_f1.max(),
            "ht_p_lo": block.ht_p.min(), "ht_p_hi": block.ht_p.max(),
            "ht_r_lo": block.ht_r.min(), "ht_r_hi": block.ht_r.max(),
            "floor_subset_f1": floor.subset_f1, "floor_subset_p": floor.subset_p,
            "floor_ht_f1": floor.ht_f1, "floor_ht_p": floor.ht_p,
        })
    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_dir / "llm_ht_weighted_summary.csv", index=False)
    print(sdf.to_string(index=False))
    return df, sdf
