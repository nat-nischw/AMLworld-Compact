"""Conditional design-based uncertainty for fixed coreset predictions.

The design is independent simple random sampling without replacement within
strata, conditional on the realized stratum sample sizes. All illicit edges
must be in census strata. Predictions, inference seeds, and per-edge contexts
are fixed: these calculations do not estimate variation over future inference
seeds, changing contexts, or future graphs. Metrics use fractions, not percent.

The linearized SE of a mean over seeds includes their shared-sample covariance.
Confidence envelopes instead invert finite-population hypergeometric counts
with Bonferroni allocation across all random-stratum/seed components. Their
coverage is at least 95% for each reported cell or contrast separately, and
does not rely on independence between seeds or on nonzero sample variance.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
from scipy.stats import hypergeom

__all__ = [
    "hypergeom_count_interval", "stratified_covariance", "summarize_cell",
    "summarize_contrast",
]

_METRICS = ("p", "f1", "lift")


def _strata(stratum_ids, population_sizes: Mapping[str, int], n_rows: int):
    ids = np.asarray(stratum_ids)
    if ids.ndim != 1 or ids.size != n_rows:
        raise ValueError("stratum_ids must have one entry per sample row")
    if not n_rows or set(ids.tolist()) != set(population_sizes):
        raise ValueError("every population stratum must occur in the sample")
    strata = []
    for name, size in population_sizes.items():
        if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
            raise ValueError("population sizes must be positive integers")
        rows = np.flatnonzero(ids == name)
        n = len(rows)
        if n > size:
            raise ValueError("stratum sample exceeds its population size")
        if n < size and n < 2:
            raise ValueError("sampled strata need at least two rows for variance estimation")
        strata.append((str(name), int(size), rows))
    return strata


def stratified_covariance(values, stratum_ids, population_sizes) -> np.ndarray:
    """Estimate covariance of HT totals for n-by-k fixed sample values.

    Census strata contribute zero. For each sampled stratum the contribution
    is N_h**2 * (1 - n_h/N_h) * sample_covariance / n_h, with ddof=1.
    Stratum identity, not equality of weights, determines the grouping.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("values must be a finite n-by-k matrix with k >= 1")
    covariance = np.zeros((values.shape[1], values.shape[1]))
    for _, size, rows in _strata(stratum_ids, population_sizes, len(values)):
        n = len(rows)
        if n == size:
            continue
        centered = values[rows] - values[rows].mean(axis=0)
        sample_covariance = centered.T @ centered / (n - 1)
        covariance += size**2 * (1 - n / size) * sample_covariance / n
    return covariance


def hypergeom_count_interval(N: int, n: int, x: int, alpha: float = .05) -> tuple[int, int]:
    """Equal-tailed exact confidence interval for the population success count.

    If X ~ Hypergeom(N, M, n), retain candidate M when both P_M(X >= x) and
    P_M(X <= x) are at least alpha/2. Monotonicity permits integer binary search.
    Includes all-zero/all-one samples and census samples; n=0 returns [0,N].
    Discreteness makes coverage conservative rather than exactly 1-alpha.
    """
    for value in (N, n, x):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("N, n, and x must be integers")
    if not 0 <= x <= n <= N or not 0 < alpha < 1:
        raise ValueError("require 0 <= x <= n <= N and 0 < alpha < 1")
    if n == N:
        return int(x), int(x)
    if n == 0:
        return 0, int(N)
    tail = alpha / 2
    # At least x successes and n-x failures are observed in the population.
    support_lo, support_hi = int(x), int(N - n + x)
    lo, hi = support_lo, support_hi
    while lo < hi:
        mid = (lo + hi) // 2
        if hypergeom.sf(x - 1, N, mid, n) >= tail:
            hi = mid
        else:
            lo = mid + 1
    lower = lo
    lo, hi = support_lo, support_hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hypergeom.cdf(x, N, mid, n) >= tail:
            lo = mid
        else:
            hi = mid - 1
    return lower, lo


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(
        numerator, denominator, out=np.zeros(numerator.shape, dtype=float),
        where=denominator != 0,
    )


def _prepare(predictions, labels, stratum_ids, population_sizes):
    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels)
    if (predictions.ndim != 2 or not predictions.shape[1]
            or not np.isin(predictions, [0, 1]).all()):
        raise ValueError("predictions must be a binary n-by-k matrix with k >= 1")
    if labels.shape != (len(predictions),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be a binary vector with one entry per sample row")
    strata = _strata(stratum_ids, population_sizes, len(predictions))
    weights = np.empty(len(labels), dtype=float)
    for _, size, rows in strata:
        if size != len(rows) and labels[rows].any():
            raise ValueError("all illicit edges must belong to census strata")
        weights[rows] = size / len(rows)
    illicit = int(labels.sum())
    if not illicit:
        raise ValueError("precision lift requires a positive illicit population")
    total = sum(size for _, size, _ in strata)
    return predictions, labels, strata, weights, illicit, total


def _point_metrics(predictions, labels, weights, illicit, total):
    tp = (predictions * labels[:, None]).sum(axis=0)
    fn = illicit - tp
    benign_predictions = predictions * (1 - labels[:, None])
    fp = (benign_predictions * weights[:, None]).sum(axis=0)
    p = _divide(tp, tp + fp)
    f1 = _divide(2 * tp, 2 * tp + fn + fp)
    prevalence = illicit / total
    metrics = {"tp": tp, "fn": fn, "fp": fp, "tn": total - illicit - fp,
               "p": p, "r": tp / illicit, "f1": f1, "lift": p / prevalence}
    gradients = {"p": -_divide(tp, (tp + fp)**2),
                 "f1": -_divide(2 * tp, (2 * tp + fn + fp)**2)}
    gradients["lift"] = gradients["p"] / prevalence
    return metrics, gradients, benign_predictions


def _count_envelope(predictions, labels, strata, metrics, prevalence, alpha, comparisons):
    k = predictions.shape[1]
    n_random = sum(size > len(rows) for _, size, rows in strata)
    n_components = comparisons * k * n_random
    per_component_alpha = alpha / n_components if n_components else None
    lower, upper = np.zeros(k), np.zeros(k)
    for _, size, rows in strata:
        counts = (predictions[rows] * (1 - labels[rows, None])).sum(axis=0).astype(int)
        if size == len(rows):
            lower += counts
            upper += counts
        else:
            for seed, count in enumerate(counts):
                lo, hi = hypergeom_count_interval(size, len(rows), count, per_component_alpha)
                lower[seed] += lo
                upper[seed] += hi
    tp, fn = metrics["tp"], metrics["fn"]
    intervals = {
        "fp": np.column_stack((lower, upper)),
        "p": np.column_stack((_divide(tp, tp + upper), _divide(tp, tp + lower))),
        "f1": np.column_stack((_divide(2 * tp, 2 * tp + fn + upper),
                               _divide(2 * tp, 2 * tp + fn + lower))),
    }
    intervals["lift"] = intervals["p"] / prevalence
    return intervals, n_components, per_component_alpha


def _quadratic_se(gradient, covariance):
    # Rounding can produce a tiny negative value for exact-zero contrasts.
    return float(np.sqrt(max(0., gradient @ covariance @ gradient)))


def summarize_cell(predictions, labels, stratum_ids, population_sizes) -> dict:
    """Summarize a seed-mean metric with shared-sample SE and a 95% envelope.

    ``seed_sd`` is descriptive variation across the supplied inference seeds
    (sample SD, ddof=1). It is separate from ``sampling_se``, which conditions
    on all supplied prediction vectors. Five columns are used in the paper;
    the function also permits other positive numbers of columns for validation.

    Even an observed all-one prediction matrix retains envelope uncertainty.
    In contrast, ``predict_all`` is a known rule on the entire population and
    has exactly known metrics, so its subtraction adds no sampling uncertainty.
    """
    predictions, labels, strata, weights, illicit, total = _prepare(
        predictions, labels, stratum_ids, population_sizes)
    k = predictions.shape[1]
    prevalence = illicit / total
    metrics, gradients, benign_predictions = _point_metrics(
        predictions, labels, weights, illicit, total)
    covariance = stratified_covariance(benign_predictions, stratum_ids, population_sizes)
    intervals, n_components, component_alpha = _count_envelope(
        predictions, labels, strata, metrics, prevalence, .05, comparisons=1)
    means = {name: float(values.mean()) for name, values in metrics.items()}
    seed_sd = {name: float(values.std(ddof=1)) if k > 1 else 0.
               for name, values in metrics.items()}
    sampling_se = {name: _quadratic_se(gradient / k, covariance)
                   for name, gradient in gradients.items()}
    sampling_se.update(fp=_quadratic_se(np.ones(k) / k, covariance), r=0.)
    envelope = {name: intervals[name].mean(axis=0).tolist() for name in _METRICS}
    per_seed = []
    for seed in range(k):
        row = {name: float(values[seed]) for name, values in metrics.items()}
        row["fp_sampling_se"] = float(np.sqrt(max(0., covariance[seed, seed])))
        for name, gradient in gradients.items():
            row[name + "_sampling_se"] = float(abs(gradient[seed]) * row["fp_sampling_se"])
        for name, bounds in intervals.items():
            row[name + "_lower"], row[name + "_upper"] = bounds[seed].tolist()
        per_seed.append(row)
    predict_all = {"p": prevalence, "r": 1., "f1": 2 * illicit / (total + illicit),
                   "lift": 1.}
    differences = {name: means[name] - predict_all[name] for name in _METRICS}
    difference_envelope = {
        name: [endpoint - predict_all[name] for endpoint in envelope[name]]
        for name in _METRICS
    }
    observed_strata = []
    for name, size, rows in strata:
        if size == len(rows):
            continue
        counts = benign_predictions[rows].sum(axis=0).astype(int)
        observed_strata.append({
            "stratum": name, "population_size": size, "sample_size": len(rows),
            "positive_counts": counts.tolist(),
            "all_positive_seed_count": int((counts == len(rows)).sum()),
            "all_zero_seed_count": int((counts == 0).sum()),
        })
    return {
        "n_seeds": k, "population_size": total, "illicit_population": illicit,
        "prevalence": prevalence,
        "random_strata": [name for name, size, rows in strata if size > len(rows)],
        "mean": means, "seed_sd": seed_sd, "seed_sd_ddof": 1,
        "sampling_se": sampling_se, "envelope95": envelope, "per_seed": per_seed,
        "predict_all": predict_all,
        "versus_predict_all": {"difference": differences, "envelope95": difference_envelope},
        "fp_covariance": covariance.tolist(), "alpha": .05,
        "n_simultaneous_count_intervals": n_components,
        "component_alpha": component_alpha,
        "observed_random_strata": observed_strata,
    }


def summarize_contrast(predictions_a, predictions_b, labels, stratum_ids, population_sizes) -> dict:
    """Paired difference (A minus B) of seed-mean metrics on the same sample.

    The SE includes cross-model/prompt and cross-seed covariance. The envelope
    allocates alpha across *both* cells' component counts, then subtracts
    simultaneous endpoints. It is conservative, including for identical cells.
    Each contrast has its own >=95% coverage; this is not familywise coverage
    across all model comparisons in a results table.
    """
    a, labels_a, strata, weights, illicit, total = _prepare(
        predictions_a, labels, stratum_ids, population_sizes)
    b, _, _, _, _, _ = _prepare(predictions_b, labels, stratum_ids, population_sizes)
    if a.shape != b.shape:
        raise ValueError("paired cells must have identical row and seed dimensions")
    k = a.shape[1]
    metrics_a, gradient_a, benign_a = _point_metrics(a, labels_a, weights, illicit, total)
    metrics_b, gradient_b, benign_b = _point_metrics(b, labels_a, weights, illicit, total)
    intervals_a, n_components, component_alpha = _count_envelope(
        a, labels_a, strata, metrics_a, illicit / total, .05, comparisons=2)
    intervals_b, _, _ = _count_envelope(
        b, labels_a, strata, metrics_b, illicit / total, .05, comparisons=2)
    differences, se, envelopes = {}, {}, {}
    for name in _METRICS:
        differences[name] = float(metrics_a[name].mean() - metrics_b[name].mean())
        # This scalar projection is algebraically the joint-covariance
        # quadratic form. Subtract before summation so identical paired
        # predictions cancel exactly instead of leaving roundoff variance.
        linearized = (benign_a * gradient_a[name] - benign_b * gradient_b[name]).mean(axis=1)
        variance = stratified_covariance(linearized[:, None], stratum_ids, population_sizes)[0, 0]
        se[name] = float(np.sqrt(max(0., variance)))
        lo_a, hi_a = intervals_a[name].mean(axis=0)
        lo_b, hi_b = intervals_b[name].mean(axis=0)
        envelopes[name] = [float(lo_a - hi_b), float(hi_a - lo_b)]
    return {"difference": differences, "sampling_se": se, "envelope95": envelopes,
            "alpha": .05, "n_seeds": k, "n_simultaneous_count_intervals": n_components,
            "component_alpha": component_alpha}
