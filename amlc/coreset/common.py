"""Helpers shared by the two coreset constructions and by the ablation.

Both constructions start from the same material: the three supervised scorers'
probabilities on the full temporal test split, averaged into one ensemble
probability per edge, and a difficulty score derived from it. This module holds
that loading step, the metric primitives, the distribution summaries the size
sweeps report, and the Item Response Theory (IRT) eligibility check that the
Naive Coreset search uses as a constraint. It writes nothing.

Reads, when a stage regenerates from the original run directory. Every path is
built by :func:`amlc.archive.legacy_path`, because these filenames
were frozen before the paper's naming was settled:

    <archive>/test_probs/<dataset>/test_labels.npy
    <archive>/test_probs/<dataset>/test_typologies.npy
    <archive>/test_probs/<dataset>/<member>/seed_<seed>.npy
    <archive>/test_probs/<dataset>/<member>/seed_<seed>_typ.npy

Two threshold grids, deliberately
---------------------------------
The construction stages swept ``arange(0.05, 0.95, 0.01)`` for the ensemble's
operating threshold; the ablation and the per-model comparison swept
``linspace(0.01, 0.99, 200)``. The two grids do not contain the same points and
so do not always select the same threshold. Both produced published numbers, so
both are kept, named, and passed explicitly. Collapsing them into one grid would
silently move the operating point of one half of the results.

Token estimates
---------------
The size sweeps report a token budget per subset size, and the Naive Coreset's
562M-token figure in the paper's Naive-versus-HT table comes from that column.
The estimate is a single measurement multiplied by the subset size: the archived
pipeline serialised one synthetic 45-edge context graph in each format and
divided its character count by four. It never touched the real coreset, so the
per-format character counts are constants of the serialiser, recorded here as
:data:`MOCK_CASE_CHARS` and reproduced in every released summary JSON. Keeping
the number rather than the measurement means constructing a coreset does not
require the serialiser or a copy of AMLworld. Pass fresh ``chars_per_case`` to
:func:`token_stats` if the serialiser ever changes.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.stats import entropy as kl_entropy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .. import paths
from ..config import ENSEMBLE_MEMBERS, SEEDS
from ..archive import legacy_path, member_dir
from ..typology import TYPOLOGY_CLASSES
from .sampler import difficulty_from_probs

#: Typology integer -> name, in the encoding the arrays on disk were written
#: with. Owned by :mod:`amlc.typology`; never re-spell it here.
TYPOLOGY_NAMES: dict[int, str] = dict(enumerate(TYPOLOGY_CLASSES))

#: Worker threads for the probability files, 80% of the cores.
DEFAULT_WORKERS = max(1, int((os.cpu_count() or 4) * 0.8))

# ── IRT eligibility thresholds ───────────────────────────────────────────
# A subset is IRT-eligible when it holds enough illicit items, enough raters
# (model x seed combinations) and enough spread in difficulty to fit an item
# response model. Reference: https://aclanthology.org/2022.insights-1.14.pdf
IRT_MIN_ILLICIT = 50    # minimum illicit items for any IRT signal
IRT_GOOD_ILLICIT = 100  # minimum for stable parameter estimation
IRT_MIN_RATERS = 5      # minimum model x seed combinations
IRT_MIN_DIFF_STD = 0.10  # minimum std of the difficulty scores
IRT_BINS = 3            # difficulty bins: easy / medium / hard
IRT_MIN_PER_BIN = 5     # minimum illicit items per difficulty bin

#: Threshold grid used while constructing a coreset.
CONSTRUCTION_THRESHOLD_GRID = np.arange(0.05, 0.95, 0.01)

#: Threshold grid used by the ablation and the per-model comparison.
EVALUATION_THRESHOLD_GRID = np.linspace(0.01, 0.99, 200)

#: Edges in one context graph at k=2 hops, capped at 50 neighbours per hop.
AVG_EDGES_PER_CASE = 45

#: The cl100k_base rule of thumb the archived pipeline used.
CHARS_PER_TOKEN = 4

#: Serialised characters for one 45-edge context graph, per serialiser format.
#: Measured once on the released run; the same values appear under
#: ``serializer.chars_per_case`` in every released summary JSON.
MOCK_CASE_CHARS: dict[str, int] = {
    "edge_list": 3509,
    "adjacency": 2448,
    "structured": 5936,
    "json": 8068,
}


# ─────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────

def load_test_data(
    dataset: str,
    members: Sequence[str] = ENSEMBLE_MEMBERS,
    seeds: Sequence[int] = SEEDS,
    archive: Optional[Path] = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Load labels, typologies and every scorer's probabilities for one dataset.

    ``members`` takes the paper's ensemble-member names; the archived directory
    names are resolved by :mod:`amlc.archive`. Files that are missing
    or the wrong length are skipped, which is how the archived pipeline behaved
    when a seed had not finished.

    The probability files are read on a thread pool: numpy I/O releases the GIL,
    so the threads genuinely overlap.
    """
    archive = Path(archive) if archive is not None else paths.archive()
    labels = np.load(legacy_path(archive, "test_labels", dataset))
    typologies = np.load(legacy_path(archive, "test_typologies", dataset))
    n = len(labels)

    keys = [(m, s) for m in members for s in seeds]

    def _load_one(key):
        member, seed = key
        p = legacy_path(archive, "member_probs", dataset,
                        member=member_dir(member), seed=seed)
        if not p.exists():
            return key, None
        arr = np.load(p)
        return key, arr if len(arr) == n else None

    probs: dict[tuple[str, int], np.ndarray] = {}
    n_workers = min(workers, len(keys))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for key, arr in ex.map(_load_one, keys):
            if arr is not None:
                probs[key] = arr

    if not probs:
        raise RuntimeError(
            f"no scorer probabilities found for {dataset} under {archive}; "
            "check the members and seeds")
    print(f"  Loaded {len(probs)} rater(s) (member x seed) for {dataset}  "
          f"[{n_workers} threads]")

    return dict(labels=labels, typologies=typologies, probs=probs,
                n=n, n_raters=len(probs))


def load_member_typologies(
    dataset: str,
    members: Sequence[str] = ENSEMBLE_MEMBERS,
    seeds: Sequence[int] = SEEDS,
    archive: Optional[Path] = None,
) -> dict[tuple[str, int], np.ndarray]:
    """Per-scorer typology-head predictions on the full test split.

    Returns ``{(member, seed): array}`` for the files that exist. The integers
    follow the encoding in :mod:`amlc.typology`.
    """
    archive = Path(archive) if archive is not None else paths.archive()
    out: dict[tuple[str, int], np.ndarray] = {}
    for member in members:
        for seed in seeds:
            p = legacy_path(archive, "member_typology", dataset,
                            member=member_dir(member), seed=seed)
            if p.exists():
                out[(member, seed)] = np.load(p)
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Ensemble probability, threshold, difficulty
# ─────────────────────────────────────────────────────────────────────────

def ensemble_soft_avg(data: dict) -> np.ndarray:
    """Soft-average ensemble probability over every rater (member x seed)."""
    stacked = np.stack(list(data["probs"].values()), axis=0)
    return stacked.mean(axis=0)


def find_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray = CONSTRUCTION_THRESHOLD_GRID,
) -> tuple[float, float]:
    """Threshold maximising F1 over ``grid``. Returns ``(threshold, f1)``.

    See the module docstring on why two grids exist. Pass
    :data:`EVALUATION_THRESHOLD_GRID` from the ablation and the comparison.
    """
    best_f1, best_t = 0.0, 0.5
    for t in grid:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def compute_difficulty(data: dict) -> np.ndarray:
    """Per-edge difficulty, 1 at the decision boundary and 0 when confident.

    This is :func:`amlc.coreset.sampler.difficulty_from_probs` applied to
    the soft-average ensemble probability, which is what the archived pipeline
    computed inline.
    """
    return difficulty_from_probs(ensemble_soft_avg(data))


# ─────────────────────────────────────────────────────────────────────────
#  Item response theory
# ─────────────────────────────────────────────────────────────────────────

def build_response_matrix(data: dict, idx: np.ndarray) -> np.ndarray:
    """``R[i, j] = 1`` when rater j classifies item i correctly at p >= 0.5."""
    labels = data["labels"][idx]
    stacked = np.stack([arr[idx] for arr in data["probs"].values()], axis=0)
    preds = (stacked >= 0.5).astype(np.int8)
    return (preds == labels[np.newaxis, :]).astype(np.int8).T


def irt_difficulty_1pl(R: np.ndarray) -> np.ndarray:
    """1-PL item difficulty, ``beta_i = logit(error_rate_i)``. Higher is harder."""
    acc = np.clip(R.mean(axis=1), 0.01, 0.99)
    return np.log((1.0 - acc) / acc)


def irt_discrimination(R: np.ndarray) -> np.ndarray:
    """Point-biserial correlation of each rater with the mean total score."""
    total = R.mean(axis=1).astype(float)
    R_f = R.astype(float)
    combined = np.vstack([total, R_f.T])
    std = combined.std(axis=1)
    if std[0] == 0 or np.all(std[1:] == 0):
        return np.zeros(R.shape[1])
    corr = np.corrcoef(combined)
    disc = corr[0, 1:]
    disc[std[1:] == 0] = 0.0
    return disc


def check_irt(data: dict, difficulty: np.ndarray,
              illicit_idx: np.ndarray) -> dict:
    """IRT eligibility of a subset, judged on its illicit items only.

    Benign items are trivial for every rater and carry no information about
    item difficulty, so they are excluded. Returns a level of ``GOOD``, ``WEAK``
    or ``NO`` plus the reasons.

    The archived signature also took the subset index array and never used it;
    it is dropped here.
    """
    n_ill = len(illicit_idx)
    n_rat = data["n_raters"]
    diff_ill = difficulty[illicit_idx]
    diff_std = float(diff_ill.std()) if n_ill > 1 else 0.0

    issues = []
    if n_ill < IRT_MIN_ILLICIT:
        issues.append(f"too few illicit ({n_ill} < {IRT_MIN_ILLICIT})")
    if n_rat < IRT_MIN_RATERS:
        issues.append(f"too few raters ({n_rat} < {IRT_MIN_RATERS})")
    if diff_std < IRT_MIN_DIFF_STD:
        issues.append(f"low difficulty spread (sigma={diff_std:.3f} "
                      f"< {IRT_MIN_DIFF_STD})")

    if n_ill >= IRT_BINS:
        edges = np.percentile(diff_ill, np.linspace(0, 100, IRT_BINS + 1))
        bin_counts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            cnt = int(((diff_ill >= lo) & (diff_ill <= hi)).sum())
            bin_counts.append(cnt)
            if cnt < IRT_MIN_PER_BIN:
                issues.append(
                    f"difficulty bin has only {cnt} illicit < {IRT_MIN_PER_BIN}")
    else:
        bin_counts = [n_ill]

    level = "NO"
    if not issues:
        level = "GOOD" if n_ill >= IRT_GOOD_ILLICIT else "WEAK"

    return dict(level=level, issues=issues, n_illicit=n_ill, n_raters=n_rat,
                diff_std=diff_std, bin_counts=bin_counts)


# ─────────────────────────────────────────────────────────────────────────
#  Distribution fidelity
# ─────────────────────────────────────────────────────────────────────────

def typology_dist(typologies: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Typology frequency over illicit edges, Laplace-smoothed."""
    n_classes = len(TYPOLOGY_CLASSES)
    ill_typ = typologies[labels == 1]
    counts = np.zeros(n_classes)
    for t in ill_typ:
        if 0 <= t < n_classes:
            counts[t] += 1
    counts += 1e-9
    return counts / counts.sum()


def difficulty_dist(difficulty: np.ndarray, labels: np.ndarray,
                    n_bins: int = 10) -> np.ndarray:
    """Difficulty histogram over illicit edges, Laplace-smoothed."""
    diff_ill = difficulty[labels == 1]
    hist, _ = np.histogram(diff_ill, bins=n_bins, range=(0, 1))
    hist = hist.astype(float) + 1e-9
    return hist / hist.sum()


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D(p || q)."""
    return float(kl_entropy(p, q))


# ─────────────────────────────────────────────────────────────────────────
#  Token budget
# ─────────────────────────────────────────────────────────────────────────

def token_stats(chars_per_case: Optional[Mapping[str, int]] = None,
                avg_edges: int = AVG_EDGES_PER_CASE) -> dict:
    """Characters and tokens for one context graph, per serialiser format.

    ``chars_per_case`` defaults to :data:`MOCK_CASE_CHARS`, the measurement the
    released numbers were computed from.
    """
    chars_per_case = MOCK_CASE_CHARS if chars_per_case is None else chars_per_case
    return {
        fmt: dict(n_chars=int(n_chars),
                  n_tokens=max(1, int(n_chars) // CHARS_PER_TOKEN),
                  chars_per_edge=int(n_chars) // avg_edges)
        for fmt, n_chars in chars_per_case.items()
    }


def estimate_context_size(n_cases: int, tok_stats: dict,
                          fmt: str = "edge_list") -> dict:
    """LLM load at one subset size: context graphs, edges and tokens."""
    s = tok_stats[fmt]
    return dict(
        n_context_graphs=n_cases,
        est_edges_total=n_cases * AVG_EDGES_PER_CASE,
        est_tokens_total=n_cases * s["n_tokens"],
        est_tokens_per_case=s["n_tokens"],
        serializer_fmt=fmt,
        all_fmt_tokens={f: n_cases * v["n_tokens"] for f, v in tok_stats.items()},
    )


# ─────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────

def weighted_metrics(labels: np.ndarray, preds: np.ndarray,
                     weights: np.ndarray) -> dict:
    """Horvitz-Thompson weighted precision, recall and F1.

    Each edge carries the inverse of its inclusion probability, so the weighted
    confusion counts estimate the full-population ones. Illicit edges are a
    census and weigh 1; benign edges weigh ``N_stratum / n_from_stratum``.
    """
    w_tp = float(weights[(labels == 1) & (preds == 1)].sum())
    w_fp = float(weights[(labels == 0) & (preds == 1)].sum())
    w_fn = float(weights[(labels == 1) & (preds == 0)].sum())
    w_tn = float(weights[(labels == 0) & (preds == 0)].sum())

    p = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0.0
    r = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return dict(precision=p, recall=r, f1=f1, tp=w_tp, fp=w_fp, fn=w_fn, tn=w_tn)


def unweighted_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Standard precision, recall and F1, for the naive comparison column."""
    return dict(
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        f1=float(f1_score(labels, preds, zero_division=0)),
    )


def typology_metrics(gt_typ: np.ndarray, preds: np.ndarray,
                     typ_preds: np.ndarray) -> dict:
    """Typology macro-F1 and accuracy over predicted-illicit edges with a label."""
    eval_mask = (preds == 1) & (gt_typ >= 0)
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        return dict(typ_macro_f1=0.0, typ_accuracy=0.0, typ_n_eval=0)
    return dict(
        typ_macro_f1=float(f1_score(gt_typ[eval_mask], typ_preds[eval_mask],
                                    average="macro", zero_division=0)),
        typ_accuracy=float(accuracy_score(gt_typ[eval_mask],
                                          typ_preds[eval_mask])),
        typ_n_eval=n_eval,
    )


def ensemble_typology_vote(typ_preds_list: Sequence[np.ndarray],
                           n: int) -> np.ndarray:
    """Majority-vote typology across the scorers' typology heads.

    Ties go to the class the first rater predicted, which is what
    ``Counter.most_common`` does and therefore what the published numbers used.
    """
    stack = np.stack(typ_preds_list)
    result = np.zeros(n, dtype=int)
    for i in range(n):
        counter = Counter(stack[:, i].tolist())
        result[i] = counter.most_common(1)[0][0]
    return result
