"""Horvitz-Thompson weight recomputation for an already-drawn HT-Coreset.

Given a realised coreset (the subset index array) the HT weights are fully
determined by the stratification, with no randomness left:

    difficulty(e)  = 1 - 2 * |p_ens(e) - 0.5|
    hard-negative  = benign edge with p_ens >= 0.5 * tau
    easy terciles  = the remaining benign edges split at the 33rd and 67th
                     percentiles of difficulty
    w(e)           = N_s / n_s   for the stratum s containing e
    w(e)           = 1           for every illicit edge (Stage 1 census)

so ``sum(w) == n_test_full`` by construction.

Why this module exists
----------------------
the pre-release generator's ``select_benign_stratified`` computes each stratum's weight
at the moment that stratum is drawn, then appends a "spare fill" block when the
per-stratum budgets round down to fewer edges than requested::

    still_need = n_select - len(selected)
    if still_need > 0:
        spare   = ben_pool[~used_mask[ben_pool]]
        extra   = rng.choice(spare, min(still_need, len(spare)), replace=False)
        w_extra = len(spare) / len(extra)          # <-- the defect

At that point every benign edge is *already* represented: the hard-negative
block carries mass ``N_hard`` and the three tercile blocks carry ``N_easy``
between them, summing to ``N_benign``. Treating the leftover pool as a fresh
unrepresented stratum adds a second copy of almost the entire benign
population on top.

Observed on the released draws:

===========  ========  ==========================  ===================
dataset      n_fill    sum(w) as archived          correct sum(w)
===========  ========  ==========================  ===================
HI-Small     0         1,015,669  (already right)  1,015,669
LI-Small     1         2,767,353  (1.998x)         1,384,810
===========  ========  ==========================  ===================

For LI-Small the single fill edge (position 646, test-split edge 442087,
benign) received w = 1,382,543 where the next-largest benign weight is 1,136.

The repair does not redraw anything. Each fill edge already belongs to one of
the four strata; recomputing ``N_s / n_s`` over the realised sample simply
counts it in its own stratum, which both fixes the total and leaves every
other weight essentially unchanged. On LI-Small only the 415 edges of the
first easy tercile move, and 414 of them by 0.24% (1,103.00 -> 1,100.36).

Verification hook: on HI-Small, which has no fill block, ``recompute_weights``
reproduces the archived weight vector bit for bit. That is the regression test
that the reconstruction matches the generator.

Draw names are the paper's (``ht-coreset``, ``ablation-redraw``); every
pre-release filename lives in :mod:`amlc.archive`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Fraction of the decision threshold above which a benign edge counts as a hard
# negative and is retained as a census.
from ..config import HARD_NEG_THRESHOLD_FRAC

# Nominal ensemble decision thresholds, as reported in the paper. These are NOT
# safe to assume when reconstructing a stratification: each draw was built with
# the threshold its own run computed, and the ablation re-draw used a different
# one (0.78809 / 0.48769 rather than the paper's rounded 0.80 / 0.48). Prefer
# :func:`infer_hard_neg_cut`.
from ..config import CONSTRUCTION_THRESHOLDS as NOMINAL_THRESHOLDS

# A draw may carry a handful of spare-fill edges. More anomalous weights than
# this inside one stratum means the stratification itself is wrong.
MAX_EXPECTED_FILL = 8


def difficulty_from_probs(ens_probs: np.ndarray) -> np.ndarray:
    """Per-edge difficulty: 1 at the decision boundary, 0 when confident.

    Peaks at p = 0.5 regardless of the operating threshold tau.
    """
    return 1.0 - np.abs(ens_probs - 0.5) * 2.0


def infer_hard_neg_cut(
    subset_idx: np.ndarray,
    labels: np.ndarray,
    ens_probs: np.ndarray,
    archived_weights: np.ndarray,
) -> float:
    """Recover the hard-negative probability cut used to build a given draw.

    The hard-negative block is a census: every benign edge above the cut is
    retained, at weight exactly 1. So the retained benign edges with w == 1 are
    the whole stratum, and the cut is pinned to the open interval between the
    highest-probability benign edge left out of the coreset and the
    lowest-probability one inside it. Returns the midpoint of that interval,
    which reproduces the partition exactly.

    Use this instead of assuming ``0.5 * NOMINAL_THRESHOLDS[dataset]``: the
    released draw happens to match the nominal value, the ablation re-draw does
    not, and getting it wrong silently reassigns most of the weights.
    """
    sel_labels = labels[subset_idx]
    census = subset_idx[(sel_labels == 0) & (archived_weights == 1.0)]
    if len(census) == 0:
        raise ValueError("no benign edge carries weight 1; cannot locate the cut")
    benign = np.where(labels == 0)[0]
    outside = np.setdiff1d(benign, census, assume_unique=False)

    lo = float(ens_probs[outside].max())
    hi = float(ens_probs[census].min())
    if not lo < hi:
        raise ValueError(
            f"the retained weight-1 benign edges are not separated by probability "
            f"from the rest (max outside {lo}, min inside {hi}); this draw's "
            "hard-negative block is not a clean census"
        )
    return (lo + hi) / 2.0


def assign_strata(
    labels: np.ndarray, ens_probs: np.ndarray, threshold: float,
    hard_neg_cut: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Partition the full test split into the four construction strata.

    Returns a dict of stratum name -> global edge indices. Illicit edges are
    not included: they are a deterministic census at weight 1.
    """
    difficulty = difficulty_from_probs(ens_probs)
    ben_pool = np.where(labels == 0)[0]

    cut = hard_neg_cut if hard_neg_cut is not None else threshold * HARD_NEG_THRESHOLD_FRAC
    hard_mask = ens_probs[ben_pool] >= cut
    hard_pool = ben_pool[hard_mask]
    easy_pool = ben_pool[~hard_mask]

    diff_easy = difficulty[easy_pool]
    q33, q67 = np.percentile(diff_easy, [33, 67])

    return {
        "hard_neg": hard_pool,
        "easy_t1": easy_pool[diff_easy < q33],
        "easy_t2": easy_pool[(diff_easy >= q33) & (diff_easy < q67)],
        "easy_t3": easy_pool[diff_easy >= q67],
    }


def recompute_weights(
    subset_idx: np.ndarray,
    labels: np.ndarray,
    ens_probs: np.ndarray,
    threshold: Optional[float] = None,
    hard_neg_cut: Optional[float] = None,
    archived_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Recompute HT weights for a realised coreset draw.

    Pass ``archived_weights`` whenever they are available: the cut is then
    inferred from them rather than assumed, and the result is checked against
    them (see :func:`check_stratification`). Returns ``(weights, report)``.
    """
    if hard_neg_cut is None and archived_weights is not None:
        hard_neg_cut = infer_hard_neg_cut(subset_idx, labels, ens_probs, archived_weights)
    if hard_neg_cut is None and threshold is None:
        raise ValueError("supply archived_weights, hard_neg_cut, or threshold")

    strata = assign_strata(labels, ens_probs, threshold or 0.0, hard_neg_cut)
    sel_labels = labels[subset_idx]
    weights = np.ones(len(subset_idx), dtype=float)

    report = {"n_illicit": int((sel_labels == 1).sum()), "strata": {}}
    covered = np.zeros(len(subset_idx), dtype=bool)
    covered[sel_labels == 1] = True

    for name, pool in strata.items():
        in_stratum = (sel_labels == 0) & np.isin(subset_idx, pool)
        n_s = int(in_stratum.sum())
        if n_s == 0:
            continue
        w_s = len(pool) / n_s
        weights[in_stratum] = w_s
        covered |= in_stratum
        report["strata"][name] = {
            "N": int(len(pool)), "n_selected": n_s, "weight": float(w_s),
        }

    if not covered.all():
        raise ValueError(
            f"{int((~covered).sum())} selected edges fall outside every stratum; "
            "the ensemble probabilities or threshold do not match the ones used "
            "to build this subset."
        )

    report["hard_neg_cut"] = float(hard_neg_cut) if hard_neg_cut is not None else None
    report["sum_weights"] = float(weights.sum())
    report["n_test_full"] = int(len(labels))
    # Note this is NOT a correctness test: weights are assigned as N_s/n_s over
    # a partition of the benign population, so the total is n_test_full for ANY
    # partition, including a wrong one. Kept as a sanity value only. The real
    # check is check_stratification().
    report["sum_matches_population"] = bool(
        np.isclose(weights.sum(), len(labels), rtol=0, atol=1e-6)
    )

    if archived_weights is not None:
        report["stratification_check"] = check_stratification(
            subset_idx, labels, weights, archived_weights, strata)

    return weights, report


def check_stratification(
    subset_idx: np.ndarray,
    labels: np.ndarray,
    recomputed: np.ndarray,
    archived: np.ndarray,
    strata: dict,
) -> dict:
    """Verify the reconstructed strata against the archived weight vector.

    Two independent tests.

    **Census membership.** The hard-negative block is a census, so an edge
    carries archived weight exactly 1 if and only if it is in that block. The
    reconstructed ``hard_neg`` stratum must therefore equal the set of benign
    coreset edges with archived weight 1, exactly.

    **Weight homogeneity.** Under the correct stratification each stratum's
    archived weights take one value, plus at most a couple of spare-fill
    outliers, and a fill outlier is always *larger* than the modal value
    because it was assigned the whole leftover pool's mass. An anomaly at or
    below the modal value is a misassigned edge, never a fill edge, so it is an
    error at any count.

    **Pool consistency.** The decisive test. The generator set each stratum's
    weight to ``N_s / n_s``, so the archived modal weight must equal the
    reconstructed pool size divided by the count of non-fill members. A cut
    that is merely close moves ``N_s`` by a few edges and shows up here at the
    sixth significant figure, even when the census membership happens to
    survive because the extra population edges were never sampled.

    Together these catch a wrong hard-negative cut, which ``sum(w) ==
    n_test_full`` cannot: that total holds for any partition whatsoever.
    """
    sel_labels = labels[subset_idx]
    benign = sel_labels == 0

    census_expected = benign & (archived == 1.0)
    census_got = benign & np.isin(subset_idx, strata["hard_neg"])
    if not np.array_equal(census_expected, census_got):
        raise ValueError(
            f"the reconstructed hard-negative census holds {int(census_got.sum())} "
            f"edges but {int(census_expected.sum())} benign edges carry archived "
            f"weight 1 ({int((census_expected ^ census_got).sum())} disagree). The "
            "hard-negative cut does not match the one this draw was built with; "
            "pass archived_weights so it is inferred rather than assumed."
        )

    out = {"strata": {}, "n_fill_detected": 0}
    for name, pool in strata.items():
        members = benign & np.isin(subset_idx, pool)
        if not members.any():
            continue
        vals, counts = np.unique(archived[members], return_counts=True)
        modal = float(vals[int(np.argmax(counts))])
        anom = archived[members] != modal
        low = int((archived[members][anom] <= modal).sum())
        high = int((archived[members][anom] > modal).sum())
        n_nonfill = int((archived[members] == modal).sum())
        implied = len(pool) / n_nonfill
        out["strata"][name] = {
            "n_members": int(members.sum()),
            "pool_size": int(len(pool)),
            "distinct_archived_weights": int(len(vals)),
            "modal_archived_weight": modal,
            "implied_weight": float(implied),
            "n_fill_like": high,
            "n_misassigned": low,
        }
        out["n_fill_detected"] += high
        if not np.isclose(modal, implied, rtol=1e-9, atol=0):
            raise ValueError(
                f"stratum {name!r} has archived weight {modal:.6f} but the "
                f"reconstructed pool implies {implied:.6f} "
                f"({len(pool)} population edges / {n_nonfill} non-fill members). "
                "The reconstructed stratification does not match the one this "
                "draw was built with; pass archived_weights so the "
                "hard-negative cut is inferred rather than assumed."
            )
        if low:
            raise ValueError(
                f"stratum {name!r} holds {low} archived weights at or below its "
                f"modal value {modal:.4f}. A spare-fill edge is always heavier "
                "than its stratum, so these are misassigned edges: the "
                "reconstructed stratification is wrong."
            )
        if high > MAX_EXPECTED_FILL:
            raise ValueError(
                f"stratum {name!r} holds {high} archived weights above its modal "
                f"value {modal:.4f}, more than a spare-fill block "
                f"({MAX_EXPECTED_FILL}) could explain."
            )
    out["archived_sum"] = float(archived.sum())
    out["recomputed_sum"] = float(recomputed.sum())
    out["reproduces_archived_exactly"] = bool(np.array_equal(archived, recomputed))
    return out


def repair_archived_weights(
    archive: Path,
    dataset: str,
    draw: str = "ht-coreset",
    threshold: Optional[float] = None,
) -> dict:
    """Recompute the archived weights for one draw and report what changes.

    ``draw`` takes a paper name; :mod:`amlc.archive` resolves it to
    the filenames used inside the archived run directory. Reads only; the
    caller decides whether to write the result. Uses only the frozen historical
    construction scores and thresholds, independently of the evaluation ensemble.
    """
    from ..archive import legacy_path

    archive = Path(archive)
    subset_idx = np.load(legacy_path(archive, "subset", dataset, draw=draw))
    saved = np.load(legacy_path(archive, "weights", dataset, draw=draw))
    ens_probs = np.load(legacy_path(archive, "construction_probs", dataset))
    labels = np.load(legacy_path(archive, "test_labels", dataset))
    thr = NOMINAL_THRESHOLDS[dataset] if threshold is None else threshold

    fixed, report = recompute_weights(subset_idx, labels, ens_probs,
                                      threshold=thr, archived_weights=saved)

    changed = ~np.isclose(saved, fixed, rtol=1e-9, atol=1e-9)
    report.update({
        "dataset": dataset,
        "draw": draw,
        "threshold": thr,
        "n_edges": int(len(subset_idx)),
        "sum_saved": float(saved.sum()),
        "sum_fixed": float(fixed.sum()),
        "n_changed": int(changed.sum()),
        "max_abs_change": float(np.abs(saved - fixed).max()) if changed.any() else 0.0,
        "identical": bool(not changed.any()),
        "changed_positions": np.flatnonzero(changed).tolist()[:20],
    })
    return {"saved": saved, "fixed": fixed, "subset_idx": subset_idx, "report": report}
