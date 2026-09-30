"""Fixed-design held-out GFP-tree sampling diagnostics from cached predictions.

Construction accepts one family only. Evaluation thresholds belong to the other
family and do not enter construction. All illicit rows are a census; independent
simple random samples within benign strata have known positive inclusion rates.
The deterministic residual allocation differs from the released random fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from .sampling_uncertainty import hypergeom_count_interval

SAMPLERS = ("uniform_ht", "difficulty_ht", "hard_terciles_ht")
METRICS = ("precision", "recall", "f1")


@dataclass(frozen=True)
class Stratum:
    name: str
    indices: np.ndarray
    sample_size: int

    @property
    def population_size(self) -> int:
        return len(self.indices)

    @property
    def inclusion_probability(self) -> float:
        return self.sample_size / self.population_size


def _probabilities_and_thresholds(probabilities, validation_thresholds):
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(validation_thresholds, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[1] == 0
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("probabilities must be a finite n-by-k matrix in [0,1]")
    if (thresholds.shape != (probabilities.shape[1],)
            or not np.isfinite(thresholds).all()
            or ((thresholds < 0) | (thresholds > 1)).any()):
        raise ValueError("one saved validation threshold in [0,1] is required per seed")
    return probabilities, thresholds


def constructor_pool(probabilities, validation_thresholds):
    """Pool only constructor seeds; there is no test-label threshold search."""
    probabilities, thresholds = _probabilities_and_thresholds(
        probabilities, validation_thresholds)
    return probabilities.mean(axis=1), float(thresholds.mean())


def threshold_predictions(probabilities, validation_thresholds):
    """Each held-out seed keeps its own saved validation threshold."""
    probabilities, thresholds = _probabilities_and_thresholds(
        probabilities, validation_thresholds)
    return probabilities >= thresholds


def _terciles(indices, difficulty, prefix):
    if not len(indices):
        return []
    values = difficulty[indices]
    q33, q67 = np.percentile(values, [33, 67])
    masks = (values < q33, (values >= q33) & (values < q67), values >= q67)
    return [(f"{prefix}_{i + 1}", indices[mask])
            for i, mask in enumerate(masks) if mask.any()]


def _residual_allocation(sizes, initial, budget):
    """Largest remainder proportional to residual population; stable tie order."""
    sizes = np.asarray(sizes, dtype=np.int64)
    allocation = np.asarray(initial, dtype=np.int64).copy()
    if ((allocation <= 0).any() or (allocation > sizes).any()
            or allocation.sum() > budget or budget > sizes.sum()):
        raise ValueError("budget must permit positive inclusion in every nonempty stratum")
    residual = int(budget - allocation.sum())
    if residual:
        capacity = sizes - allocation
        # Integer arithmetic avoids a rounding-dependent tie or capacity breach.
        numerators = residual * capacity
        total_capacity = int(capacity.sum())
        extra = numerators // total_capacity
        remainders = numerators % total_capacity
        remaining = residual - int(extra.sum())
        order = np.argsort(-remainders, kind="stable")
        extra[order[:remaining]] += 1
        allocation += extra
    assert allocation.sum() == budget and np.all((allocation > 0) & (allocation <= sizes))
    return allocation


def make_design(labels, constructor_scores, constructor_tau, benign_budget, sampler):
    """Define all benign strata and fixed allocations before examining evaluation errors."""
    labels = np.asarray(labels)
    scores = np.asarray(constructor_scores, dtype=float)
    if (labels.ndim != 1 or not np.isin(labels, [0, 1]).all()
            or scores.shape != labels.shape or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise ValueError("binary labels and a matching finite constructor score vector required")
    if not np.isfinite(constructor_tau) or not 0 <= constructor_tau <= 1:
        raise ValueError("constructor tau must be in [0,1]")
    benign = np.flatnonzero(labels == 0)
    if (isinstance(benign_budget, bool) or not isinstance(benign_budget, (int, np.integer))
            or not 0 < benign_budget <= len(benign)):
        raise ValueError("benign budget must be a positive integer no larger than its population")
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler: {sampler}")
    if sampler == "uniform_ht":
        return [Stratum("benign", benign, int(benign_budget))]
    difficulty = 1 - 2 * np.abs(scores - .5)
    if sampler == "difficulty_ht":
        pools = _terciles(benign, difficulty, "difficulty")
        initial = [min(len(pool), max(1, benign_budget // 3)) for _, pool in pools]
    else:
        hard = benign[scores[benign] >= constructor_tau / 2]
        easy = benign[scores[benign] < constructor_tau / 2]
        easy_pools = _terciles(easy, difficulty, "easy")
        hard_n = min(len(hard), max(1, int(.3 * benign_budget)),
                     benign_budget - len(easy_pools)) if len(hard) else 0
        pools = ([("hard", hard)] if len(hard) else []) + easy_pools
        easy_n = max(1, (benign_budget - hard_n) // 3)
        initial = ([hard_n] if len(hard) else []) + [
            min(len(pool), easy_n) for _, pool in easy_pools]
    allocation = _residual_allocation([len(pool) for _, pool in pools], initial, benign_budget)
    return [Stratum(name, pool, int(n)) for (name, pool), n in zip(pools, allocation)]


def draw_counts(design, predictions, rng):
    """Sample rows, preserving the shared draw and covariance across held-out seeds."""
    counts = []
    for stratum in design:
        rows = (stratum.indices if stratum.sample_size == stratum.population_size else
                rng.choice(stratum.indices, stratum.sample_size, replace=False))
        counts.append(predictions[rows].sum(axis=0))
    return np.asarray(counts, dtype=np.int64)


@cache
def cached_count_interval(N, n, x, alpha):
    return hypergeom_count_interval(int(N), int(n), int(x), float(alpha))


def metrics_from_counts(tp, illicit_count, fp):
    tp, fp = np.asarray(tp, dtype=float), np.asarray(fp, dtype=float)
    if illicit_count <= 0:
        raise ValueError("the illicit census must be nonempty")
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp != 0)
    recall = tp / illicit_count
    f1 = 2 * tp / (tp + illicit_count + fp)
    return np.column_stack((precision, recall, f1))


def evaluate_counts(design, sampled_counts, tp, illicit_count, alpha=.05):
    """Return per-seed P/R/F1 point estimates and simultaneous count envelopes.

    Bonferroni allocation is across k seeds times all noncensus benign strata,
    as in sampling_uncertainty.summarize_cell. It is not across configurations.
    """
    counts = np.asarray(sampled_counts)
    tp = np.asarray(tp)
    if (counts.shape != (len(design), len(tp)) or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 0).any()):
        raise ValueError("one integer observed count per stratum and evaluation seed required")
    k = len(tp)
    n_components = k * sum(s.sample_size < s.population_size for s in design)
    component_alpha = alpha / n_components if n_components else None
    fp, lower, upper = np.zeros(k), np.zeros(k), np.zeros(k)
    for stratum, observed in zip(design, counts):
        N, n = stratum.population_size, stratum.sample_size
        if (observed > n).any():
            raise ValueError("observed count exceeds the stratum sample size")
        fp += N / n * observed
        if n == N:
            lower += observed
            upper += observed
        else:
            bounds = np.asarray([cached_count_interval(N, n, int(x), component_alpha)
                                 for x in observed])
            lower += bounds[:, 0]
            upper += bounds[:, 1]
    return {
        "point": metrics_from_counts(tp, illicit_count, fp),
        "lower": metrics_from_counts(tp, illicit_count, upper),
        "upper": metrics_from_counts(tp, illicit_count, lower),
        "fp": fp, "fp_lower": lower, "fp_upper": upper,
        "n_components": n_components, "component_alpha": component_alpha,
    }


def summarize_draws(points, lower, upper, truth):
    """Summaries over draws; inputs may have trailing seed/metric dimensions."""
    points, lower, upper, truth = map(np.asarray, (points, lower, upper, truth))
    errors = points - truth
    covered = (lower <= truth + 1e-12) & (upper >= truth - 1e-12)
    coverage = covered.mean(axis=0)
    return {
        "truth": truth, "mean_estimate": points.mean(axis=0),
        "signed_bias": errors.mean(axis=0), "mae": np.abs(errors).mean(axis=0),
        "rmse": np.sqrt(np.square(errors).mean(axis=0)),
        "sd": points.std(axis=0, ddof=1), "coverage95": coverage,
        "coverage_mcse": np.sqrt(coverage * (1 - coverage) / len(points)),
        "mean_width95": (upper - lower).mean(axis=0),
    }
