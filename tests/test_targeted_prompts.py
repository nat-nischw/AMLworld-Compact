"""Adversarial integrity tests for the unevaluated versioned prompt resource."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amlc.serialize.targeted_graph import (
    ObservableTransaction,
    assert_evaluation_text,
    evaluation_prompt,
    preserve_target,
    recover_archived_transactions,
    serialise_targeted_graph,
    validate_targeted_text,
)


def transaction(index=0, *, source="001_A", destination="002_B", timestamp="2022/09/01 00:00"):
    return ObservableTransaction(f"e_{index}", source, destination, "12.50", "US Dollar",
                                 "Credit Card", timestamp, "11.25", "Euro")


def test_missing_target_replaces_latest_non_target_and_keeps_both_endpoints():
    early = transaction(1)
    late = transaction(3, timestamp="2022/09/02 00:00")
    target = transaction(9, source="003_C", destination="004_D", timestamp="2022/09/03 00:00")
    graph = preserve_target("amlc_00000", [early, late], target)
    assert len(graph.transactions) == 2
    assert graph.removed_edge_ids == ("e_3",)
    assert not graph.target_was_present
    text = serialise_targeted_graph(graph)
    assert text.count(" [TARGET] ") == 1
    assert "- acct_003_C (type: Account)" in text
    assert "- acct_004_D (type: Account)" in text
    assert "Start timestamp: 2022/09/01 00:00" in text
    assert "End timestamp: 2022/09/03 00:00" in text
    assert "Elapsed time: 172800 seconds" in text
    assert "amount_received: 11.25 receiving_currency: Euro" in text


def test_existing_target_is_kept_even_with_one_edge_cap():
    target = transaction(9, timestamp="2022/09/03 00:00")
    graph = preserve_target("amlc_00000", [transaction(), target], target, max_transactions=1)
    assert graph.transactions == (target,)
    assert graph.target_was_present
    assert "Elapsed time: 0 seconds" in serialise_targeted_graph(graph)


def test_ties_evict_largest_numeric_id_deterministically():
    target = transaction(99)
    graph = preserve_target("amlc_00000", [transaction(12), transaction(2)], target)
    reverse = preserve_target("amlc_00000", [transaction(2), transaction(12)], target)
    assert graph.transactions == reverse.transactions
    assert graph.removed_edge_ids == ("e_12",)


@pytest.mark.parametrize("bad", [[transaction(), transaction()],
                                [transaction(1), transaction(1), transaction()]])
def test_duplicate_target_or_context_identity_fails(bad):
    with pytest.raises(ValueError, match="Duplicate"):
        preserve_target("amlc_00000", bad, transaction())


def test_same_id_with_wrong_source_attributes_fails():
    with pytest.raises(ValueError, match="disagree"):
        preserve_target("amlc_00000", [transaction()], replace(transaction(), amount="99.00"))


def test_identical_observable_values_do_not_make_different_edge_ids_the_target():
    graph = preserve_target("amlc_00000", [transaction(1), transaction(2)], transaction(9))
    assert not graph.target_was_present
    assert [t.edge_id for t in graph.transactions] == ["e_1", "e_9"]
    assert graph.removed_edge_ids == ("e_2",)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_fails(budget):
    with pytest.raises(ValueError, match="budget"):
        preserve_target("amlc_00000", [], transaction(), max_transactions=budget)


def test_self_transfer_has_one_explicit_account_node():
    target = transaction(source="001_A", destination="001_A")
    graph = preserve_target("amlc_00000", [], target)
    text = serialise_targeted_graph(graph)
    assert text.count("- acct_001_A (type: Account)") == 1
    assert "Total transactions: 1" in text


def test_same_account_code_at_different_banks_is_unambiguous():
    target = transaction(source="001_A", destination="002_A")
    text = serialise_targeted_graph(preserve_target("amlc_00000", [], target))
    assert "- acct_001_A (type: Account)" in text
    assert "- acct_002_A (type: Account)" in text


def test_csv_excludes_groundtruth_but_keeps_all_observable_fields():
    row = ["2022/09/01 00:00", "001", "A", "002", "B", "11.25", "Euro",
           "12.50", "US Dollar", "Credit Card", "SECRET_GROUND_TRUTH"]
    target = ObservableTransaction.from_csv_row(3, row)
    text = evaluation_prompt(preserve_target("amlc_00000", [], target))
    assert "SECRET_GROUND_TRUTH" not in text
    assert "transaction e_3, from acct_001_A to acct_002_B" in text
    assert "<ID>" not in text


def test_legitimate_benign_values_are_not_misclassified_as_label_fields():
    target = replace(transaction(), payment_format="benign label invoice groundtruth typology")
    text = evaluation_prompt(preserve_target("amlc_00000", [], target))
    assert "benign label invoice groundtruth typology" in text
    assert_evaluation_text(text)


@pytest.mark.parametrize("text", ["label: 1", '  "typology": "cycle"',
                                 "- Is Laundering: 0", "ground_truth = benign",
                                 "groundtruth: 1"])
def test_groundtruth_fields_are_rejected(text):
    with pytest.raises(ValueError, match="Ground-truth"):
        assert_evaluation_text(text)


@pytest.mark.parametrize("text", ["Target: <ID>", "transaction {{edge_id}}"])
def test_placeholders_are_rejected(text):
    with pytest.raises(ValueError, match="Unresolved"):
        assert_evaluation_text(text)


def test_field_line_injection_is_rejected():
    with pytest.raises(ValueError, match="single-line"):
        replace(transaction(), payment_format="Cash\nlabel: 1")


def test_mutated_target_line_and_summary_are_rejected():
    graph = preserve_target("amlc_00000", [], transaction())
    text = serialise_targeted_graph(graph)
    for changed in (text.replace("amount_paid: 12.50", "amount_paid: 12.51"),
                    text.replace("Elapsed time: 0 seconds", "Elapsed time: 9 seconds"),
                    text.replace(" [TARGET] ", " "),
                    text + "\n- edge_id: e_9 [TARGET] bogus"):
        with pytest.raises(ValueError):
            validate_targeted_text(changed, graph)


def archive_pair():
    txn = transaction()
    graph_json = json.dumps({"case_id": "v2_00000", "transactions": [
        {"id": txn.edge_id, "from": txn.source, "to": txn.destination,
         "amount": float(txn.amount), "currency": txn.currency, "timestamp": txn.timestamp}],
        "statistics": {"n_transactions": 1}})
    # This fixture intentionally lists graph lines for readable mutation tests.
    graph_text = "\n".join([
        "=== Transaction Subgraph (Case: v2_00000) ===", "", "**Nodes:**",
        "- acct_A (type: Account)", "- acct_B (type: Account)",
        "- bank_001 (type: Bank)", "- bank_002 (type: Bank)", "", "**Edges:**",
        "- acct_A belongs_to bank_001", "- acct_B belongs_to bank_002", txn.historical_line()])
    return graph_json, graph_text


def test_archive_companion_mapping_recovers_identity():
    graph_json, graph_text = archive_pair()
    recovered = recover_archived_transactions(graph_json, graph_text, "v2_00000")
    assert len(recovered) == 1
    assert recovered[0].same_graph_fields(transaction())


def test_archive_mapping_ambiguity_or_label_field_stops_build():
    graph_json, graph_text = archive_pair()
    with pytest.raises(ValueError, match="mapping"):
        recover_archived_transactions(graph_json, graph_text.replace("12.50", "12.51"), "v2_00000")
    data = json.loads(graph_json)
    data["transactions"][0]["label"] = 1
    with pytest.raises(ValueError, match="fields"):
        recover_archived_transactions(json.dumps(data), graph_text, "v2_00000")


def test_archive_duplicate_ids_stop_build():
    graph_json, graph_text = archive_pair()
    data = json.loads(graph_json)
    data["transactions"] *= 2
    data["statistics"]["n_transactions"] = 2
    graph_text += "\n" + transaction().historical_line()
    with pytest.raises(ValueError, match="Duplicate"):
        recover_archived_transactions(json.dumps(data), graph_text, "v2_00000")
