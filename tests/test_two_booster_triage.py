"""Regression checks for evaluating a frozen construction draw with new scorers."""

import numpy as np
import pytest

from amlc.triage import doubt_triage as triage
from amlc.archive import legacy_path, member_dir
from amlc.config import ENSEMBLE_MEMBERS


def _save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)


def test_evaluation_scores_do_not_redefine_construction_weights(tmp_path, monkeypatch):
    archive = tmp_path / "outputs"
    dataset = "HI-Small"
    labels = np.array([1] + [0] * 10)
    construction = np.array([0.9, 0.7, 0.01, 0.02, 0.03, 0.04,
                             0.05, 0.06, 0.07, 0.08, 0.09])
    indices = np.array([0, 1, 2, 5, 8])
    expected_weights = np.array([1., 1., 3., 3., 3.])
    for kind, values in (("test_labels", labels),
                         ("test_typologies", np.array([0] + [-1] * 10)),
                         ("construction_probs", construction)):
        _save(legacy_path(archive, kind, dataset), values)
    _save(legacy_path(archive, "subset", dataset, draw="ht-coreset"), indices)
    _save(legacy_path(archive, "weights", dataset, draw="ht-coreset"), expected_weights)
    # Evaluation now calls all sampled benign cases illicit. Their design
    # weights must still represent the original three strata, not be reset.
    evaluation = np.full(len(labels), 0.95)
    monkeypatch.setattr("amlc.baselines.ml.ensemble.load_evaluation_probabilities",
                        lambda archive, dataset: evaluation)
    data = triage.load_coreset_from_archive(archive, dataset, verify=False)
    np.testing.assert_array_equal(data["ml_probs"], evaluation[indices])
    np.testing.assert_array_equal(data["weights"], expected_weights)
    assert data["weights"].sum() == len(labels)


def test_typology_ties_use_frozen_canonical_encoding(tmp_path):
    archive = tmp_path / "outputs"
    heads = ([7, 0, 4], [1, 2, 4])
    for member, values in zip(ENSEMBLE_MEMBERS, heads):
        _save(legacy_path(archive, "member_typology", "HI-Small",
                          member=member_dir(member), seed=42), values)
    result = triage.load_ensemble_typology(archive, "HI-Small")
    assert result.tolist() == ["fan-in", "fan-out", "gather-scatter"]


def test_missing_typology_member_cannot_change_vote(tmp_path):
    archive = tmp_path / "outputs"
    _save(legacy_path(archive, "member_typology", "HI-Small",
                      member=member_dir(ENSEMBLE_MEMBERS[0]), seed=42), [0, 1])
    with pytest.raises(FileNotFoundError, match="missing evaluation typology"):
        triage.load_ensemble_typology(archive, "HI-Small")


def test_dt_routing_preserves_llm_labels():
    dt = triage.DoubtTriage(np.array([0.1, 0.9]), np.array([1., 1.]), 0.8)
    preds, routing = dt.predict(np.array([1, 0]))
    assert preds.tolist() == [1, 1]
    assert routing.tolist() == ["llm_flipped", "ml_illicit"]
