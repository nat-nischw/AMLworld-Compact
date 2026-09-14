"""The evaluation model can change without changing the sampling design."""

import json

import numpy as np
import pytest

from amlc import config, hub
from amlc.archive import legacy_path, member_dir
from amlc.baselines.ml.ensemble import (
    _score_typology, load_evaluation_probabilities, score_evaluation_ensemble,
)


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "outputs"
    labels = legacy_path(root, "test_labels", "HI-Small")
    labels.parent.mkdir(parents=True)
    np.save(labels, [0, 1, 0, 1])
    for i, member in enumerate(config.ENSEMBLE_MEMBERS):
        for seed in (42, 123):
            path = legacy_path(root, "member_probs", "HI-Small",
                               member=member_dir(member), seed=seed)
            path.parent.mkdir(exist_ok=True)
            np.save(path, np.array([0.1, 0.9, 0.2, 0.7]) + i * 0.02 + (seed == 123) * 0.04)
    frozen = legacy_path(root, "construction_probs", "HI-Small")
    frozen.parent.mkdir(parents=True)
    np.save(frozen, [0.0, 0.0, 1.0, 1.0])
    return root


def test_evaluation_excludes_frozen_construction_and_gcpal(archive):
    before = legacy_path(archive, "construction_probs", "HI-Small").read_bytes()
    got = load_evaluation_probabilities(archive, "HI-Small", seeds=[42, 123])
    np.testing.assert_allclose(got, [0.13, 0.93, 0.23, 0.73])
    # No GCPAL file is needed, and the old construction array stays untouched.
    assert legacy_path(archive, "ensemble_probs", "HI-Small").read_bytes() == before
    np.testing.assert_allclose(
        load_evaluation_probabilities(archive, "HI-Small", seeds=[42]),
        [0.11, 0.91, 0.21, 0.71])


def test_missing_seed_never_changes_ensemble_silently(archive):
    path = legacy_path(archive, "member_probs", "HI-Small",
                       member=member_dir(config.ENSEMBLE_MEMBERS[0]), seed=123)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete evaluation ensemble"):
        load_evaluation_probabilities(archive, "HI-Small", seeds=[42, 123])


@pytest.mark.parametrize("bad", [[0.1, 0.2], [[0.1, 0.2, 0.3, 0.4]],
                                  [0.1, np.nan, 0.2, 0.3],
                                  [0.1, 1.1, 0.2, 0.3]])
def test_invalid_member_probabilities_fail(archive, bad):
    path = legacy_path(archive, "member_probs", "HI-Small",
                       member=member_dir(config.ENSEMBLE_MEMBERS[0]), seed=42)
    np.save(path, bad)
    with pytest.raises(ValueError):
        load_evaluation_probabilities(archive, "HI-Small", seeds=[42])


@pytest.mark.parametrize("seeds", [[], [42, 42]])
def test_invalid_seed_selection_fails(archive, seeds):
    with pytest.raises(ValueError, match="seeds"):
        load_evaluation_probabilities(archive, "HI-Small", seeds=seeds)


def test_primary_scoring_keeps_inherited_threshold(archive):
    rows = score_evaluation_ensemble(archive, "HI-Small", seeds=[42, 123])
    assert len(rows) == 2
    for row in rows:
        assert row["threshold"] == 0.80
        # At 0.80 only one of two illicit items is detected. Retuning would
        # reach perfect F1, so this catches an accidental oracle search.
        assert row["detection_f1"].split("±")[0] == "66.7"
        assert "GCPAL" not in row["ensemble_members"]


def test_typology_ties_choose_lowest_canonical_index():
    got = _score_typology(np.array([0, 1]), np.ones(2, dtype=int),
                          [np.array([0, 2]), np.array([2, 1])])
    assert got["typ_f1"] == 1.0


def test_hub_rejects_stale_three_member_evaluation(tmp_path, monkeypatch):
    d = tmp_path / "HI-Small"
    d.mkdir()
    for filename, array in [("ht_subset_indices.npy", [0, 1]),
                             ("ht_weights.npy", [1, 1]),
                             ("labels.npy", [0, 1]),
                             ("typologies.npy", [-1, 0]),
                             ("ensemble_probs_coreset.npy", [0.1, 0.9])]:
        np.save(d / filename, array)
    monkeypatch.setattr(config, "CORESET_DIR", str(tmp_path))
    monkeypatch.setitem(config.N_TEST_FULL, "HI-Small", 2)
    # Existing target-only consumers do not need baseline metadata.
    assert hub.load_coreset("HI-Small", with_ensemble_probs=False)["n"] == 2
    with pytest.raises(FileNotFoundError, match="scoring_metadata"):
        hub.load_coreset("HI-Small")
    metadata = {"evaluation": {"members": list(config.CONSTRUCTION_MEMBERS),
                               "seeds": list(config.SEEDS), "threshold": 0.80}}
    (d / "scoring_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        hub.load_coreset("HI-Small")
    metadata["evaluation"]["members"] = list(config.ENSEMBLE_MEMBERS)
    (d / "scoring_metadata.json").write_text(json.dumps(metadata))
    np.testing.assert_array_equal(hub.load_coreset("HI-Small")["ml_probs"], [0.1, 0.9])


def test_historical_typology_ties_preserve_member_order():
    # Three distinct votes: the historical Counter rule picks the first
    # member's 2; the primary rule picks canonical class 0.
    truth = np.array([2])
    predictions = np.array([1])
    heads = [np.array([2]), np.array([0]), np.array([1])]
    historical = _score_typology(truth, predictions, heads, tie_rule="first_member")
    primary = _score_typology(truth, predictions, heads)
    assert historical["typ_acc"] == 1.0
    assert primary["typ_acc"] == 0.0
    # A true plurality always wins, even if another class was seen first.
    heads = [np.array([0]), np.array([2]), np.array([2])]
    for tie_rule in ("lowest", "first_member"):
        assert _score_typology(truth, predictions, heads, tie_rule)["typ_acc"] == 1.0
