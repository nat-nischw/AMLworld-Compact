"""Offline checks of audit eligibility, lexical vacuity, and released counts."""

from pathlib import Path

import pandas as pd
import pytest

from amlc.analysis.audit_sensitivity import (
    JUDGES,
    PREFIX_CHARS,
    as_bool,
    attach_labels,
    lexical_features,
    read_annotations,
    summarize,
)
from amlc.audit.sample_traces import annotate_trace


def test_original_scoring_is_preserved_and_benign_vacuity_is_visible():
    text = "The graph has acct_one as its central hub. fan-out fanout."
    suspicious = lexical_features(text, True)
    benign = lexical_features(text, False)
    original = annotate_trace(text, {"illicit": True}, 0, "")
    assert {s: suspicious[s] for s in ("parse", "recall", "match")} == {
        s: int(original[s]) for s in ("parse", "recall", "match")}
    assert suspicious["recall"] == 1  # Two surface forms of one class.
    assert suspicious["canonical_recall"] == 0
    assert suspicious["canonical_typology_count"] == 1
    assert suspicious["match"] == suspicious["marker_ge2"] == 0
    assert benign["match"] == benign["benign_match_bypass"] == 1
    assert benign["match_without_two_markers"] == 1
    assert benign["marker_ge2"] == 0
    assert lexical_features("", False)["match"] == 0


def test_truncation_holds_prediction_fixed_and_can_drop_markers():
    prefix = "The central hub acct_a has a fan-out pattern and a cycle. "
    trace = prefix + "x" * (PREFIX_CHARS - len(prefix)) + " monthly timestamp evidence."
    full, truncated = lexical_features(trace, True), lexical_features(trace[:PREFIX_CHARS], True)
    assert full["marker_ge2"] == full["match"] == 1
    assert truncated["marker_ge2"] == truncated["match"] == 0
    assert full["parse"] == truncated["parse"] == 1
    assert full["recall"] == truncated["recall"] == 1
    # Negation is deliberately not interpreted as evidence or semantic truth.
    denied = lexical_features("There is no cycle and no temporal regularity.", True)
    assert denied["marker_ge2"] == 1


def test_reference_absent_is_neither_a_typed_success_nor_a_typed_error():
    annotations = pd.DataFrame([
        {"model": "m", "case_id": "a", "gt_label": 0, "gt_typology": "",
         "pred_illicit": False, "pred_typology": "", "outcome": "correct_legit", "conclude": True},
        {"model": "m", "case_id": "b", "gt_label": 1, "gt_typology": "cycle",
         "pred_illicit": True, "pred_typology": "cycle", "outcome": "correct_illicit", "conclude": True},
        {"model": "m", "case_id": "c", "gt_label": 1, "gt_typology": "",
         "pred_illicit": True, "pred_typology": "cycle", "outcome": "typology_error", "conclude": False},
        {"model": "m", "case_id": "d", "gt_label": 1, "gt_typology": "cycle",
         "pred_illicit": False, "pred_typology": "", "outcome": "under_prediction", "conclude": False},
    ])
    features = annotations[["model", "case_id"]].assign(trace_chars=20)
    joined = attach_labels(annotations, features)
    assert joined.binary_correct.tolist() == [True, True, True, False]
    assert joined.reference_evaluable.tolist() == [True, True, False, True]
    assert joined.typed_eligible.tolist() == [False, True, False, True]
    assert joined.typed_correct.sum() == 1
    assert joined.loc[2, "error_class"] == "reference_absent_flagged"
    assert joined.loc[2, "outcome"] == "typology_error"  # Frozen label stays intact.
    with pytest.raises(ValueError, match="keys differ"):
        attach_labels(annotations, features.iloc[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        attach_labels(annotations, pd.concat([features, features.iloc[:1]]))


def test_boolean_csv_values_are_not_python_string_truthiness():
    assert as_bool("False") is False
    assert as_bool("0") is False
    assert as_bool("True") is True
    with pytest.raises(ValueError):
        as_bool("unknown")


def test_released_features_recompute_report_without_raw_archive():
    root = Path(__file__).resolve().parents[1]
    out = root / "results/analysis/audit_sensitivity"
    annotations = read_annotations(root / "results/audit/trace_4step_annotations_n1000.csv")
    features = pd.read_csv(out / "inputs/trace_features.csv", keep_default_na=False)
    frame = attach_labels(annotations, features)
    judges = {name: pd.read_csv(root / f"results/audit/trace_4step_{name}_n1000.csv",
                                keep_default_na=False) for name in JUDGES}
    tables = summarize(frame, judges)
    assert len(frame) == 1000
    assert int((~frame.reference_evaluable).sum()) == 240
    assert int(frame.typed_eligible.sum()) == 456
    assert int(frame.typed_correct.sum()) == 183
    for step in ("parse", "recall", "match"):
        assert frame[step].eq(frame[f"full_{step}"].astype(bool)).all()
    conjunctions = tables["conjunctions"]
    overall = conjunctions[(conjunctions.model == "ALL") & (conjunctions.group_kind == "all")
                           & (conjunctions.window == "full")
                           & (conjunctions.rule == "original_PRM")].set_index("correctness")
    assert overall.loc["strict_archived", ["count", "denominator"]].tolist() == [538, 1000]
    assert overall.loc["exclude_illicit_reference_absent", ["count", "denominator"]].tolist() == [421, 760]
    rates = tables["rates"]
    conditional = rates[(rates.model == "ALL") & (rates.group_kind == "binary_outcome")]
    for group, metric, count, denominator in (
        ("incorrect", "full_nonvacuous_PRM", 298, 530),
        ("incorrect", "prefix8000_nonvacuous_PRM", 295, 530),
        ("correct", "full_nonvacuous_PRM", 397, 470),
        ("correct", "prefix8000_nonvacuous_PRM", 388, 470),
    ):
        row = conditional[(conditional.group == group) & (conditional.metric == metric)].iloc[0]
        assert (row["count"], row.denominator) == (count, denominator)
    for name, table in tables.items():
        saved = pd.read_csv(out / f"{name}.csv")
        pd.testing.assert_frame_equal(table.reset_index(drop=True), saved,
                                      check_dtype=False, check_exact=False)
    # Partitioning by model or original outcome must preserve each total.
    full = tables["truncation"]
    overall_parse = full[(full.model == "ALL") & (full.group_kind == "all")
                        & (full.metric == "parse")].iloc[0]
    models_parse = full[(full.model != "ALL") & (full.group_kind == "all")
                       & (full.metric == "parse")]
    assert models_parse.denominator.sum() == overall_parse.denominator
    assert models_parse.full_count.sum() == overall_parse.full_count
    discordance = tables["judge_discordance"]
    assert discordance[["both_pass", "both_fail", "parser_pass_judge_fail",
                        "parser_fail_judge_pass"]].sum(axis=1).eq(discordance.denominator).all()
    judge_totals = discordance[(discordance.model == "ALL") & (discordance.group_kind == "all")
                              & (discordance.parser_definition == "strict_archived")]
    assert set(judge_totals[judge_totals.judge_score_scope == "all_rows"].denominator) == {1000}
    assert set(judge_totals[judge_totals.judge_score_scope == "scored_only"].denominator) == {852}
