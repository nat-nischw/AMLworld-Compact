import numpy as np
import pandas as pd
import pytest

from amlc.analysis.target_presence import align_presence, summarize_presence


def test_presence_join_reorders_by_case_and_checks_transaction_identity():
    cases = pd.DataFrame({"case_id": ["b", "a"], "target_edge_id": ["e2", "e1"],
                          "original_target_present": [False, True]})
    assert align_presence(cases, ["a", "b"], ["e1", "e2"]).tolist() == [True, False]
    with pytest.raises(ValueError, match="edge identities"):
        align_presence(cases, ["a", "b"], ["e2", "e1"])
    with pytest.raises(ValueError, match="unique"):
        align_presence(pd.concat([cases, cases.iloc[:1]]), ["a", "b"], ["e1", "e2"])


def test_class_conditioned_denominators_do_not_confuse_prevalence_with_error():
    # Present cohort is 2 benign + 1 illicit; absent cohort is 1 benign + 2 illicit.
    labels = np.array([0, 0, 1, 0, 1, 1])
    present = np.array([True, True, True, False, False, False])
    predictions = np.array([[1, 0], [0, 1], [1, 1], [1, 1], [1, 1], [0, 0]])[:, None, :]
    runs, summary = summarize_presence(predictions, labels, present, ["m"], ["FS"], [1, 2])
    values = summary.set_index("target")
    assert values.loc["present", "benign_flag_rate"] == .5
    assert values.loc["absent", "benign_flag_rate"] == 1.
    assert values.loc["present", "illicit_recall"] == 1.
    assert values.loc["absent", "illicit_recall"] == .5
    assert len(runs) == 4


def test_empty_class_has_undefined_rate_not_perfect_performance():
    _, summary = summarize_presence(np.ones((2, 1, 2)), np.array([0, 1]),
                                    np.array([True, False]), ["m"], ["FS"], [1, 2])
    values = summary.set_index("target")
    assert np.isnan(values.loc["present", "illicit_recall"])
    assert np.isnan(values.loc["absent", "benign_flag_rate"])
