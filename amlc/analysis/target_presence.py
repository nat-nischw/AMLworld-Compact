"""Descriptive archived-output rates grouped by target presence in the input.

These are unweighted subset diagnostics. Present targets were still unmarked;
the grouping is observational and does not evaluate the corrected prompts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_presence(cases, case_ids, center_edge_ids):
    """Require a bijection and the same transaction identities before grouping."""
    cases = cases.copy()
    if cases.case_id.duplicated().any() or len(set(case_ids)) != len(case_ids):
        raise ValueError("case identities must be unique")
    if set(cases.case_id) != set(case_ids):
        raise ValueError("case identities do not match the prediction tensor")
    aligned = cases.set_index("case_id").loc[list(case_ids)]
    if not np.array_equal(aligned.target_edge_id.to_numpy(), center_edge_ids):
        raise ValueError("target edge identities do not match")
    values = aligned.original_target_present
    if values.dtype == object:
        values = values.astype(str).str.lower().map({"true": True, "false": False,
                                                    "1": True, "0": False})
    if values.isna().any() or not values.isin([True, False]).all():
        raise ValueError("presence must be a boolean for every case")
    return values.to_numpy(dtype=bool)


def summarize_presence(predictions, labels, present, models, promptings, run_ids):
    """Report each run, then average rates across fixed runs with shared cohorts."""
    predictions = np.asarray(predictions)
    labels, present = np.asarray(labels), np.asarray(present)
    if (predictions.ndim != 3 or labels.shape != predictions.shape[:1]
            or present.shape != labels.shape or not np.isin(labels, [0, 1]).all()
            or not np.isin(predictions, [0, 1]).all()
            or present.dtype != np.dtype(bool)):
        raise ValueError("aligned binary predictions/labels and boolean presence required")
    if predictions.shape[1:] != (len(models), len(run_ids)) or len(promptings) != len(models):
        raise ValueError("cell and run metadata do not match the tensor")
    rows = []
    for status, mask in [("present", present), ("absent", ~present)]:
        benign, illicit = mask & (labels == 0), mask & (labels == 1)
        for cell, (model, prompting) in enumerate(zip(models, promptings)):
            for run, run_id in enumerate(run_ids):
                fp = int(predictions[benign, cell, run].sum())
                tp = int(predictions[illicit, cell, run].sum())
                rows.append({
                    "model": str(model), "prompting": str(prompting), "target": status,
                    "run_id": int(run_id), "n_cases": int(mask.sum()),
                    "n_benign": int(benign.sum()), "n_illicit": int(illicit.sum()),
                    "false_positives": fp, "true_positives": tp,
                    "benign_flag_rate": fp / benign.sum() if benign.any() else np.nan,
                    "illicit_recall": tp / illicit.sum() if illicit.any() else np.nan,
                })
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "prompting", "target"], sort=False).agg(
        n_cases=("n_cases", "first"), n_benign=("n_benign", "first"),
        n_illicit=("n_illicit", "first"),
        benign_flag_rate=("benign_flag_rate", "mean"),
        benign_flag_run_sd=("benign_flag_rate", "std"),
        illicit_recall=("illicit_recall", "mean"),
        illicit_recall_run_sd=("illicit_recall", "std"),
    ).reset_index()
    return frame, summary
