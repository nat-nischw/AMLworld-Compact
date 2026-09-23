"""Scoring shared by the supervised baselines: threshold selection, HT-weighted P/R/F1.

Reads nothing and writes nothing. Two entry points.

:func:`find_optimal_threshold` sweeps the decision threshold and returns the one
that maximises minority-class F1. Every supervised stage calls it: the GBT
trainers tune on the validation split, GCPAL tunes per fine-tuning evaluation
and again on validation before it writes probabilities, the ensemble uses it per
member and per combination rule, and the Optuna objective for PNA used it before
that baseline was dropped. At 0.05-0.12 percent illicit a fixed 0.5 threshold
collapses recall, so the sweep is not a refinement, it is what makes the numbers
meaningful.

:func:`ht_weighted_prf` is the Horvitz-Thompson weighted counterpart used on the
coreset. It is **not** re-implemented here. There is one implementation, in
:mod:`amlc.triage.doubt_triage`, and this module imports it so that a
caller who wants both kinds of scoring has a single import and there is no
second copy to drift.

Prediction-result scoring lives elsewhere: ``compute_metrics`` and
``aggregate_metrics_across_seeds`` in :mod:`amlc.llm.runner` score
``PredictionResult`` objects against cases on the LLM evaluation path.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..triage.doubt_triage import ht_weighted_prf

__all__ = ["find_optimal_threshold", "ht_weighted_prf"]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_thresholds: int = 200,
    min_thresh: float = 0.01,
    max_thresh: float = 0.99,
) -> tuple[float, float]:
    """Sweep thresholds and return the ``(threshold, F1)`` that maximises F1.

    The grid is ``n_thresholds`` points evenly spaced on
    ``[min_thresh, max_thresh]``, and ties go to the lowest threshold because
    the sweep only replaces the incumbent on a strict improvement. Changing
    either changes which operating point a retrain lands on.

    Two things that look like outputs of this sweep are not. The
    ``best_threshold`` in ``data/tuned_params/`` is the constant 0.5, written
    deliberately because the Optuna search optimises F1 at a fixed threshold and
    leaves threshold selection to the training stage; see
    :mod:`amlc.baselines.ml.tuning`. And :data:`amlc.config.ML_THRESHOLDS` holds
    0.80 and 0.48, the paper's rounded operating points, which are not grid
    points here: the sweep returned 0.78809 and 0.48769, grid indices 158 and
    97. :mod:`amlc.coreset.ht_weights` explains why a reconstruction must use the
    unrounded values.

    When no threshold beats F1 = 0, the returned threshold is 0.5.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth binary labels, 1 for illicit.
    y_scores : ndarray
        Predicted probability of the illicit class.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    thresholds = np.linspace(min_thresh, max_thresh, n_thresholds)
    best_f1 = 0.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return float(best_thresh), float(best_f1)
