"""Finite-population checks independent of the AML cached prediction files."""

from itertools import combinations, product

import numpy as np
import pytest
from scipy.stats import hypergeom

from amlc.analysis.sampling_uncertainty import (
    hypergeom_count_interval,
    stratified_covariance,
    summarize_cell,
    summarize_contrast,
)


def test_hypergeom_endpoints_match_exhaustive_inversion_and_cover():
    alpha = .1
    for N in range(1, 13):
        for n in range(N + 1):
            intervals = [hypergeom_count_interval(N, n, x, alpha) for x in range(n + 1)]
            for x, interval in enumerate(intervals):
                accepted = [M for M in range(N + 1)
                            if hypergeom.sf(x - 1, N, M, n) >= alpha / 2
                            and hypergeom.cdf(x, N, M, n) >= alpha / 2]
                assert interval == (min(accepted), max(accepted))
            for M in range(N + 1):
                covered = sum(hypergeom.pmf(x, N, M, n)
                              for x, (lo, hi) in enumerate(intervals) if lo <= M <= hi)
                assert covered >= 1 - alpha - 1e-12


def test_hypergeom_boundary_samples_are_not_censuses():
    assert hypergeom_count_interval(100, 10, 0)[1] > 0
    assert hypergeom_count_interval(100, 10, 10)[0] < 100
    assert hypergeom_count_interval(100, 100, 64) == (64, 64)
    assert hypergeom_count_interval(100, 0, 0) == (0, 100)


@pytest.mark.parametrize("args", [(10, 11, 1), (10, 5, 6), (10, 5, -1), (10., 5, 2)])
def test_hypergeom_rejects_invalid_counts(args):
    with pytest.raises((TypeError, ValueError)):
        hypergeom_count_interval(*args)


def test_covariance_matches_enumeration_with_census_and_equal_weight_strata():
    # A and B have the same N/n weight, but very different values and means.
    a = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=float)
    b = np.array([[9, 3], [10, 1], [12, 3], [15, 1]], dtype=float)
    census = np.array([[100, -100]], dtype=float)
    totals, estimated_covariances = [], []
    for ia, ib in product(combinations(range(4), 2), repeat=2):
        sample = np.vstack((a[list(ia)], b[list(ib)], census))
        totals.append(2 * sample[:4].sum(axis=0) + sample[4])
        estimated_covariances.append(stratified_covariance(
            sample, np.array(["a", "a", "b", "b", "c"]), {"a": 4, "b": 4, "c": 1}))
    truth = np.cov(np.array(totals), rowvar=False, ddof=0)
    np.testing.assert_allclose(np.mean(estimated_covariances, axis=0), truth, atol=1e-12)
    assert truth[0, 1] != 0


def test_conditional_residual_fill_is_uniform_and_covariance_is_unbiased():
    # Draw one from A and two from B, then fill one uniformly from residuals.
    # Condition on the realized allocation A=2,B=2, as for the LI allocation.
    a = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=float)
    b = np.array([[0, 2], [2, 0], [1, 1]], dtype=float)
    final_samples = {}
    for initial_a in range(4):
        for initial_b in combinations(range(3), 2):
            for extra_a in set(range(4)) - {initial_a}:
                key = (tuple(sorted((initial_a, extra_a))), initial_b)
                # All initial samples and each residual edge have equal probability.
                final_samples[key] = final_samples.get(key, 0) + 1
    assert len(final_samples) == 6 * 3
    assert set(final_samples.values()) == {2}
    totals, estimates = [], []
    for ia, ib in final_samples:
        sample = np.vstack((a[list(ia)], b[list(ib)]))
        totals.append(2 * sample[:2].sum(axis=0) + 1.5 * sample[2:].sum(axis=0))
        estimates.append(stratified_covariance(
            sample, np.array(["a", "a", "b", "b"]), {"a": 4, "b": 3}))
    np.testing.assert_allclose(np.mean(estimates, axis=0),
                               np.cov(np.array(totals), rowvar=False, ddof=0), atol=1e-12)


def _example():
    labels = np.array([1, 1, 0, 0, 0, 0, 0])
    strata = np.array(["illicit"] * 2 + ["hard"] + ["a"] * 2 + ["b"] * 2)
    populations = {"illicit": 2, "hard": 1, "a": 8, "b": 6}
    predictions = np.array([1, 0, 1, 0, 1, 1, 0])[:, None]
    return predictions, labels, strata, populations


def test_repeated_seeds_retain_single_seed_se_and_metric_mean():
    predictions, labels, strata, sizes = _example()
    one = summarize_cell(predictions, labels, strata, sizes)
    five = summarize_cell(np.repeat(predictions, 5, axis=1), labels, strata, sizes)
    for metric in ("p", "f1", "lift", "fp"):
        assert five["sampling_se"][metric] > 0
        assert five["sampling_se"][metric] == pytest.approx(one["sampling_se"][metric])
        assert five["mean"][metric] == pytest.approx(one["mean"][metric])
        assert five["seed_sd"][metric] == pytest.approx(0.)
    assert five["n_simultaneous_count_intervals"] == 10
    assert five["component_alpha"] == .005


def test_mean_metrics_are_not_metrics_of_averaged_predictions():
    first, labels, strata, sizes = _example()
    second = first.copy()
    second[1] = 1  # TP changes from 1 to 2.
    second[4] = 0  # Weighted FP changes from 8 to 4.
    summary = summarize_cell(np.column_stack((first, second)), labels, strata, sizes)
    assert summary["mean"]["p"] == pytest.approx((1 / 9 + 2 / 6) / 2)
    assert summary["mean"]["f1"] == pytest.approx((2 / 11 + 4 / 8) / 2)
    assert summary["mean"]["p"] != pytest.approx(1.5 / (1.5 + 6))


def test_paired_identical_cells_have_zero_difference_and_linearized_se():
    predictions, labels, strata, sizes = _example()
    predictions = np.repeat(predictions, 5, axis=1)
    paired = summarize_contrast(predictions, predictions, labels, strata, sizes)
    for metric in ("p", "f1", "lift"):
        assert paired["difference"][metric] == 0
        assert paired["sampling_se"][metric] == pytest.approx(0., abs=1e-12)
        lo, hi = paired["envelope95"][metric]
        assert lo < 0 < hi
    assert paired["n_simultaneous_count_intervals"] == 20


def test_paired_delta_se_matches_direct_scalar_linearization():
    a, labels, strata, sizes = _example()
    a = np.repeat(a, 5, axis=1)
    b = a.copy()
    b[0, 0] = 0
    b[3, 1:3] = 1
    b[5, 3:] = 0
    paired = summarize_contrast(a, b, labels, strata, sizes)
    sa, sb = summarize_cell(a, labels, strata, sizes), summarize_cell(b, labels, strata, sizes)
    for metric in ("p", "f1", "lift"):
        def gradient(row, metric=metric):
            if metric == "f1":
                return -2 * row["tp"] / (2 * row["tp"] + row["fn"] + row["fp"])**2
            value = -row["tp"] / (row["tp"] + row["fp"])**2
            return value / sa["prevalence"] if metric == "lift" else value
        ga = np.array([gradient(row) for row in sa["per_seed"]])
        gb = np.array([gradient(row) for row in sb["per_seed"]])
        scalar = ((a * ga - b * gb) * (1 - labels[:, None])).mean(axis=1)
        expected_variance = 0.
        for name in ("a", "b"):
            vals = scalar[strata == name]
            N, n = sizes[name], len(vals)
            expected_variance += N**2 * (1 - n / N) * np.var(vals, ddof=1) / n
        assert paired["sampling_se"][metric] == pytest.approx(np.sqrt(expected_variance))
        assert paired["difference"][metric] == pytest.approx(
            sa["mean"][metric] - sb["mean"][metric])


def test_all_observed_positive_has_zero_se_but_nonzero_envelope():
    _, labels, strata, sizes = _example()
    summary = summarize_cell(np.ones((len(labels), 5)), labels, strata, sizes)
    assert summary["mean"]["lift"] == pytest.approx(1.)
    assert summary["sampling_se"]["f1"] == 0
    assert summary["envelope95"]["lift"][0] == pytest.approx(1.)
    assert summary["envelope95"]["lift"][1] > 1
    assert summary["versus_predict_all"]["difference"]["f1"] == pytest.approx(0.)


def test_census_has_point_envelopes_and_no_sampling_uncertainty():
    predictions, labels, strata, _ = _example()
    sizes = {str(name): int(sum(strata == name)) for name in np.unique(strata)}
    summary = summarize_cell(predictions, labels, strata, sizes)
    assert summary["n_simultaneous_count_intervals"] == 0
    assert summary["component_alpha"] is None
    for metric in ("p", "f1", "lift"):
        assert summary["sampling_se"][metric] == 0
        assert summary["envelope95"][metric] == [summary["mean"][metric]] * 2


def test_non_census_illicit_rejected():
    predictions, labels, strata, sizes = _example()
    sizes["illicit"] = 3
    with pytest.raises(ValueError, match="illicit edges"):
        summarize_cell(predictions, labels, strata, sizes)
