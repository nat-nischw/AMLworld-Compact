"""HT-Coreset benign sampler, with the spare-fill weighting fixed.

Stage 1 of the protocol retains every illicit edge at weight 1. This module is
Stage 2: sample benign edges from four strata and attach Horvitz-Thompson
weights.

    hard-negative   benign with p_ens >= 0.5 * tau, retained as a census
    easy tercile 1  the rest, split at the 33rd and 67th percentiles
    easy tercile 2  of difficulty = 1 - 2 * |p_ens - 0.5|
    easy tercile 3

Weights are assigned once, at the end, as ``N_s / n_s`` over the realised
sample. That is the whole fix.

The defect
----------
The pre-release ``select_benign_stratified`` computed each stratum's weight at
the moment that stratum was drawn, then appended a spare-fill block when the
integer budgets rounded down::

    still_need = n_select - len(selected)
    if still_need > 0:
        spare   = ben_pool[~used_mask[ben_pool]]
        extra   = rng.choice(spare, min(still_need, len(spare)), replace=False)
        w_extra = len(spare) / len(extra)          # <-- adds a second copy

By that point the four blocks already carried the entire benign population, so
the fill added almost all of it again. LI-Small summed to 2,767,353 against a
population of 1,384,810, with one edge at 1,382,543 where the next largest was
1,136. HI-Small happened to need no fill and was correct. The identical bug sat
in ``eval_baselines.py`` (which builds the ablation re-draw) and in the Elliptic
builder, whose coreset summed to 1.805x its population.

Assigning weights from realised membership makes the fill harmless: an extra
edge simply raises its own stratum's ``n_s``. :mod:`amlc.coreset.ht_weights`
applies the same rule to coresets that were already drawn.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import HARD_NEG_RATIO as DEFAULT_HARD_NEG_RATIO
from ..config import HARD_NEG_THRESHOLD_FRAC


def difficulty_from_probs(ens_probs: np.ndarray) -> np.ndarray:
    """1 at the decision boundary, 0 when the ensemble is confident."""
    return 1.0 - np.abs(ens_probs - 0.5) * 2.0


def sample_benign_stratified(
    rng: np.random.Generator,
    ben_pool: np.ndarray,
    ens_probs: np.ndarray,
    threshold: float,
    n_select: int,
    hard_neg_ratio: float = DEFAULT_HARD_NEG_RATIO,
    difficulty: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Draw ``n_select`` benign edges and weight them.

    Returns ``(indices, weights, breakdown)``. ``sum(weights)`` equals
    ``len(ben_pool)`` exactly, whatever the fill block does.
    """
    if difficulty is None:
        difficulty = difficulty_from_probs(ens_probs)
    n_ben_total = len(ben_pool)
    if n_select >= n_ben_total:
        return (ben_pool, np.ones(n_ben_total, dtype=float),
                {"hard_neg": n_ben_total, "easy": [], "n_fill": 0})

    cut = threshold * HARD_NEG_THRESHOLD_FRAC
    hard_pool = ben_pool[ens_probs[ben_pool] >= cut]
    easy_pool = ben_pool[ens_probs[ben_pool] < cut]

    n_hard_budget = min(int(n_select * hard_neg_ratio), len(hard_pool))
    n_easy_budget = min(n_select - n_hard_budget, len(easy_pool))
    if n_easy_budget < (n_select - n_hard_budget) and len(hard_pool) > n_hard_budget:
        n_hard_budget = min(n_select - n_easy_budget, len(hard_pool))

    # (population pool, drawn members). Weighting happens after every draw.
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    if n_hard_budget > 0 and len(hard_pool) > 0:
        blocks.append((hard_pool, rng.choice(hard_pool, n_hard_budget, replace=False)))

    if n_easy_budget > 0 and len(easy_pool) > 0:
        diff_easy = difficulty[easy_pool]
        q33, q67 = np.percentile(diff_easy, [33, 67])
        tertiles = [diff_easy < q33,
                    (diff_easy >= q33) & (diff_easy < q67),
                    diff_easy >= q67]
        n_per = max(1, n_easy_budget // 3)
        for mask in tertiles:
            pool = easy_pool[mask]
            if len(pool) == 0:
                continue
            n_take = min(n_per, len(pool))
            blocks.append((pool, rng.choice(pool, n_take, replace=False)))

    if not blocks:
        return (np.array([], dtype=int), np.array([], dtype=float),
                {"hard_neg": 0, "easy": [0, 0, 0], "n_fill": 0})

    n_fill = 0
    n_drawn = sum(len(c) for _, c in blocks)
    if n_drawn < n_select:
        used = np.zeros(len(ens_probs), dtype=bool)
        for _, chosen in blocks:
            used[chosen] = True
        spare = ben_pool[~used[ben_pool]]
        if len(spare):
            extra = rng.choice(spare, min(n_select - n_drawn, len(spare)),
                               replace=False)
            n_fill = len(extra)
            # Fold each extra edge into the stratum it belongs to, so the
            # weighting below counts it there.
            blocks = [
                (pool, np.concatenate([chosen, extra[np.isin(extra, pool)]]))
                for pool, chosen in blocks
            ]

    selected = np.concatenate([c for _, c in blocks])
    weights = np.concatenate(
        [np.full(len(c), len(pool) / len(c)) for pool, c in blocks])

    breakdown = {
        "hard_neg": int(len(blocks[0][1])) if blocks else 0,
        "easy": [int(len(c)) for _, c in blocks[1:]],
        "n_fill": int(n_fill),
        "sum_weights": float(weights.sum()),
        "n_benign_population": int(n_ben_total),
    }
    if not np.isclose(weights.sum(), n_ben_total, rtol=0, atol=1e-6):
        raise AssertionError(
            f"benign weights sum to {weights.sum():.1f}, expected {n_ben_total}. "
            "Weights must be assigned from realised stratum membership."
        )
    return selected, weights, breakdown
