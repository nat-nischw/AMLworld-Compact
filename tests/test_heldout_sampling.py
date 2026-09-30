"""Independent small-population checks of the held-out sampling experiment."""

from itertools import combinations, product

import numpy as np
import pytest

from amlc.analysis.heldout_sampling import (
    SAMPLERS,
    Stratum,
    cached_count_interval,
    constructor_pool,
    draw_counts,
    evaluate_counts,
    make_design,
    metrics_from_counts,
    summarize_draws,
    threshold_predictions,
)
from amlc.analysis.sampling_uncertainty import summarize_cell


def test_constructor_uses_only_own_seeds_and_validation_thresholds():
    constructor = np.array([[.1, .2], [.2, .4], [.5, .7], [.9, .8]])
    scores, tau = constructor_pool(constructor, [.4, .8])
    np.testing.assert_allclose(scores, [.15, .3, .6, .85])
    assert tau == pytest.approx(.6)
    labels = np.array([1, 0, 0, 0])
    design = make_design(labels, scores, tau, 3, "hard_terciles_ht")
    assert design[0].name == "hard"
    np.testing.assert_array_equal(design[0].indices, [1, 2, 3])
    # The evaluator's own thresholds change its answers, not the constructor tau.
    heldout = np.array([[.5, .5], [.8, .8], [.1, .9], [.6, .6]])
    np.testing.assert_array_equal(threshold_predictions(heldout, [.2, .8]),
                                  [[1, 0], [1, 1], [0, 1], [1, 0]])
    np.testing.assert_array_equal(threshold_predictions(heldout, [.7, .4]),
                                  [[0, 1], [1, 1], [0, 1], [0, 1]])
    assert tau == pytest.approx(.6)


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_exact_budget_positive_inclusion_disjoint_strata_and_weight_total(sampler):
    labels = np.array([1, 1] + [0] * 20)
    # Repeated values exercise empty terciles and deterministic tie handling.
    scores = np.array([.9, .4] + [.01] * 6 + [.1] * 6 + [.2] * 3 + [.8] * 5)
    for budget in (4, 8, 16, 20):
        design = make_design(labels, scores, .6, budget, sampler)
        assert sum(s.sample_size for s in design) == budget
        assert all(0 < s.inclusion_probability <= 1 for s in design)
        np.testing.assert_array_equal(np.sort(np.concatenate([s.indices for s in design])),
                                      np.arange(2, 22))
        assert sum(s.sample_size / s.inclusion_probability for s in design) == pytest.approx(20)


def test_hard_population_larger_than_budget_is_sampled_with_positive_probability():
    labels = np.array([1] + [0] * 30)
    scores = np.r_[.9, np.linspace(.01, .29, 15), np.linspace(.3, .9, 15)]
    design = make_design(labels, scores, .6, 8, "hard_terciles_ht")
    hard = design[0]
    assert hard.name == "hard" and hard.population_size == 15
    assert 0 < hard.sample_size < hard.population_size
    assert sum(s.sample_size for s in design) == 8
    with pytest.raises(ValueError, match="positive inclusion"):
        make_design(labels, scores, .6, 2, "hard_terciles_ht")


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_all_equal_scores_and_census_boundaries(sampler):
    labels = np.array([1] + [0] * 8)
    for score in (0., .8):
        design = make_design(labels, np.full(9, score), .5, 3, sampler)
        assert sum(s.sample_size for s in design) == 3
        assert sum(s.population_size for s in design) == 8
        census = make_design(labels, np.full(9, score), .5, 8, sampler)
        predictions = np.tile(np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])[:, None], (1, 5))
        counts = draw_counts(census, predictions, np.random.default_rng(7))
        result = evaluate_counts(census, counts, np.ones(5), 1)
        np.testing.assert_array_equal(result["point"], result["lower"])
        np.testing.assert_array_equal(result["point"], result["upper"])
        assert result["n_components"] == 0


def test_exhaustive_design_ht_counts_unbiased_and_envelopes_cover_fixed_seed_means():
    # Exhaust all 36 equally likely draws from two independent SRS strata.
    predictions = np.array([
        [1, 1, 1, 0, 1], [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0], [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0], [0, 0, 1, 1, 1],
        [1, 1, 0, 1, 1], [0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0], [0, 0, 1, 0, 1],
    ])
    design = [Stratum("a", np.arange(2, 6), 2), Stratum("b", np.arange(6, 10), 2)]
    tp = predictions[:2].sum(axis=0)
    truth_fp = predictions[2:].sum(axis=0)
    truth = metrics_from_counts(tp, 2, truth_fp)
    fp, points, lower, upper, inclusions = [], [], [], [], []
    for selected in product(combinations(range(2, 6), 2), combinations(range(6, 10), 2)):
        observed = np.array([predictions[list(rows)].sum(axis=0) for rows in selected])
        result = evaluate_counts(design, observed, tp, 2)
        assert result["n_components"] == 10 and result["component_alpha"] == .005
        fp.append(result["fp"])
        points.append(result["point"])
        lower.append(result["lower"])
        upper.append(result["upper"])
        inclusions.append(np.isin(np.arange(2, 10), np.concatenate(selected)))
    np.testing.assert_allclose(np.mean(fp, axis=0), truth_fp)
    np.testing.assert_allclose(np.mean(inclusions, axis=0), .5)
    stats = summarize_draws(np.mean(points, axis=1), np.mean(lower, axis=1),
                            np.mean(upper, axis=1), truth.mean(axis=0))
    assert np.all(stats["coverage95"] >= .95)
    assert stats["signed_bias"][1] == pytest.approx(0)
    assert stats["sd"][1] == pytest.approx(0)
    assert stats["mean_width95"][1] == 0
    assert stats["signed_bias"][0] > 0  # Ratio metrics need not be unbiased.


def test_envelopes_match_existing_sampling_uncertainty_and_cache_inversions():
    labels = np.array([1, 1, 0, 0, 0, 0, 0])
    predictions = np.array([[1, 1], [0, 1], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
    design = [Stratum("hard", np.arange(1), 1),
              Stratum("a", np.arange(8), 2), Stratum("b", np.arange(6), 2)]
    observed = np.array([predictions[2], predictions[3:5].sum(axis=0),
                         predictions[5:].sum(axis=0)])
    cached_count_interval.cache_clear()
    result = evaluate_counts(design, observed, predictions[:2].sum(axis=0), 2)
    misses = cached_count_interval.cache_info().misses
    evaluate_counts(design, observed, predictions[:2].sum(axis=0), 2)
    assert cached_count_interval.cache_info().misses == misses
    assert cached_count_interval.cache_info().hits >= 4
    reference = summarize_cell(predictions, labels,
                               np.array(["illicit", "illicit", "hard", "a", "a", "b", "b"]),
                               {"illicit": 2, "hard": 1, "a": 8, "b": 6})
    for index, name in ((0, "p"), (2, "f1")):
        assert result["point"][:, index].mean() == pytest.approx(reference["mean"][name])
        np.testing.assert_allclose([result["lower"][:, index].mean(),
                                    result["upper"][:, index].mean()],
                                   reference["envelope95"][name])


@pytest.mark.parametrize("probabilities,thresholds", [
    ([[.1, .2]], [.5]), ([[.1, float("nan")]], [.5, .5]),
    ([[.1, .2]], [.5, 1.1]), ([[1.1]], [.5]),
])
def test_invalid_probability_or_validation_threshold_rejected(probabilities, thresholds):
    with pytest.raises(ValueError):
        constructor_pool(probabilities, thresholds)
