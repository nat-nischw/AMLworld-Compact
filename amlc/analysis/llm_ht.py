"""Score saved predictions with HT weights and with unit weights.

HT P/R/F1 are plug-in ratios of weighted confusion counts that estimate
full-split performance; the ratios are not generally unbiased. Compact
(unweighted, named ``subset_*`` in the CSV) metrics describe the released
diagnostic cohort. All output metrics are percentages. Recall agrees between
the two because every illicit edge is retained at weight one.

Use ``--runs-dir`` to score fresh runner output against the released coreset
from Hugging Face or ``AMLC_CORESET_DIR``. Without it, use the original
``AMLC_ARCHIVE`` layout, including its full-test arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DATASETS, LLM_MODELS, PROMPTINGS, SEEDS
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
    strict: bool = False,
) -> list[dict]:
    """Both framings of one (dataset, model, prompting) cell, per seed."""
    rows = []
    for seed in seeds:
        source = {"runs_dir": archive} if strict else {"archive": archive}
        loaded = load_llm_predictions(
            dataset=dataset, model=model, prompting=prompting, seed=seed,
            n=data["n"], strict=strict, **source)
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

    Under the 1:2 design ratio its compact F1 is 50%. Under HT weighting its
    precision is the full-split illicit rate.
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
    *,
    runs_dir: Optional[Path] = None,
    seeds=SEEDS,
) -> list[dict]:
    """Requested cells and the predict-all baseline, with percentages.

    ``runs_dir`` needs only the runner's prediction tree; labels and weights
    come from :func:`amlc.hub.load_coreset`. Every requested file must contain
    exactly one binary verdict per coreset row. The legacy ``archive`` mode
    retains its original loading and missing-file behavior.
    """
    if runs_dir is not None:
        if archive is not None:
            raise ValueError("use either runs_dir or archive, not both")
        from ..hub import load_coreset

        archive = Path(runs_dir)
        data = load_coreset(dataset, with_ensemble_probs=False)
    else:
        archive = resolve_archive(archive)
        data = load_coreset_from_archive(archive, dataset, verify=verify)
    rows = [predict_all_floor(data, dataset)]
    for model in models:
        for prompting in promptings:
            rows.extend(score_cell(
                archive, data, dataset, model, prompting,
                seeds=seeds, strict=runs_dir is not None))
    return rows


def main(argv=None):
    from .. import paths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, action="append")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--runs-dir", type=Path,
                        help="fresh runner output; load released labels and weights")
    source.add_argument("--archive", type=Path,
                        help="original archive root; defaults to AMLC_ARCHIVE")
    ap.add_argument("--models", nargs="+", default=list(LLM_MODELS),
                    help="model directory names to score")
    ap.add_argument("--promptings", nargs="+", choices=PROMPTINGS,
                    default=["ICL-FS", "ICL-ZS"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip frozen-array verification in archive mode only")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = args.out or (paths.results() / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in (args.dataset or list(DATASETS)):
        rows.extend(score_dataset(
            dataset, archive=args.archive, models=args.models,
            promptings=args.promptings, verify=args.verify,
            runs_dir=args.runs_dir, seeds=args.seeds))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "llm_ht_weighted.csv", index=False)

    # Min/max over the requested (model, prompting) cells, after averaging
    # each cell's per-seed metrics. This is not a confidence interval.
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


if __name__ == "__main__":
    main()
