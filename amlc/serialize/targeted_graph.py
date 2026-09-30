"""Versioned, target-preserving graph text; never used by historical scores.

The archived renderer stays frozen. This renderer accepts only observable
transaction attributes, gives every edge an ID, preserves the focal edge under
the transaction budget, and measures elapsed time. Source CSV labels are never
copied into a transaction or prompt. A bank/account pair identifies an account.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache

VERSION = "targeted-prompts-v1"
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
_EDGE = re.compile(r"e_(0|[1-9][0-9]*)\Z")
_CASE = re.compile(r"(?:amlc|v2)_[0-9]{5,}\Z")
_FORBIDDEN_KEY = re.compile(
    r'''(?im)^\s*(?:[-*]\s*)?["']?(?:label|labels|illicit|is_illicit|'''
    r'''is[_ ]?laundering|typology|typology_id|ground[_ ]?truth)["']?\s*[:=]'''
)
_PLACEHOLDER = re.compile(r"<\s*ID\s*>|\{\{[^{}]*\}\}", re.IGNORECASE)


@lru_cache(maxsize=32768)
def _timestamp(value: str) -> datetime:
    # AMLworld provides no timezone; assigning UTC would invent source metadata.
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def assert_evaluation_text(text: str) -> None:
    """Reject label *fields* and unresolved identifiers, not benign values.

    This is a secondary check. The primary protection is the transaction field
    allowlist below and the strict archive-to-observable-field parser.
    """
    if _FORBIDDEN_KEY.search(text):
        raise ValueError("Ground-truth field in evaluation text")
    if _PLACEHOLDER.search(text):
        raise ValueError("Unresolved identifier in evaluation text")


def _scalar(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError(f"Invalid {name}: expected a nonempty single-line string")
    if _PLACEHOLDER.search(value):
        raise ValueError(f"Unresolved identifier in {name}")


def _amount(value: str) -> None:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Invalid transaction amount") from exc
    if not number.is_finite():
        raise ValueError("Nonfinite transaction amount")


@dataclass(frozen=True)
class ObservableTransaction:
    """An edge with no binary label, typology, or other outcome fields.

    ``amount``/``currency`` are the paid amount/currency used by the archived
    graph. The receipt fields are required for the source-verified target.
    """

    edge_id: str
    source: str
    destination: str
    amount: str
    currency: str
    payment_format: str
    timestamp: str
    amount_received: str | None = None
    receiving_currency: str | None = None

    def __post_init__(self) -> None:
        for name in ("edge_id", "source", "destination", "amount", "currency",
                     "payment_format", "timestamp"):
            _scalar(getattr(self, name), name)
        if not _EDGE.fullmatch(self.edge_id):
            raise ValueError("Invalid edge ID")
        for node in (self.source, self.destination):
            if not re.fullmatch(r"[A-Za-z0-9]+_[A-Za-z0-9]+", node):
                raise ValueError("Account must be an explicit bank_account pair")
        _amount(self.amount)
        _timestamp(self.timestamp)
        if (self.amount_received is None) != (self.receiving_currency is None):
            raise ValueError("Receipt amount and currency must be supplied together")
        if self.amount_received is not None:
            _scalar(self.amount_received, "amount_received")
            _scalar(self.receiving_currency, "receiving_currency")
            _amount(self.amount_received)

    @classmethod
    def from_csv_row(cls, row_index: int, row: Sequence[str]) -> ObservableTransaction:
        """Read positional AMLworld columns, excluding column 10 (the label)."""
        if len(row) != 11:
            raise ValueError("Expected the eleven-column AMLworld transaction schema")
        return cls(
            edge_id=f"e_{row_index}", source=f"{row[1].strip()}_{row[2].strip()}",
            destination=f"{row[3].strip()}_{row[4].strip()}", amount=row[7].strip(),
            currency=row[8].strip(), payment_format=row[9].strip(),
            timestamp=row[0].strip(), amount_received=row[5].strip(),
            receiving_currency=row[6].strip(),
        )

    def historical_line(self) -> str:
        return (f"- acct_{self.source.split('_', 1)[1]} transfers_to "
                f"acct_{self.destination.split('_', 1)[1]} "
                f"amount: {float(self.amount):.2f} {self.currency} "
                f"via: {self.payment_format} timestamp: {self.timestamp}")

    def same_graph_fields(self, other: ObservableTransaction) -> bool:
        return (self.edge_id == other.edge_id and self.source == other.source
                and self.destination == other.destination
                and Decimal(self.amount) == Decimal(other.amount)
                and self.currency == other.currency
                and self.payment_format == other.payment_format
                and self.timestamp == other.timestamp)


def recover_archived_transactions(
    archived_json_text: str, archived_typed_text: str, archived_case_id: str,
) -> list[ObservableTransaction]:
    """Prove each JSON edge maps to the corresponding archived transfer line.

    JSON preserves edge IDs and bank-qualified accounts; typed text preserves
    payment format. Both archives have the same stable timestamp ordering.
    Any discrepancy or repeated edge identity stops the build.
    """
    data = json.loads(archived_json_text)
    if set(data) != {"case_id", "transactions", "statistics"}:
        raise ValueError("Unexpected archived JSON graph schema")
    if data["case_id"] != archived_case_id or not _CASE.fullmatch(archived_case_id):
        raise ValueError("Archived case ID mismatch")
    lines = archived_typed_text.splitlines()
    if not lines or lines[0] != f"=== Transaction Subgraph (Case: {archived_case_id}) ===":
        raise ValueError("Archived typed-graph case ID mismatch")
    transfer_lines = [line for line in lines if " transfers_to " in line]
    if len(transfer_lines) != len(data["transactions"]):
        raise ValueError("Archived companion transaction counts disagree")
    recovered = []
    expected_keys = {"id", "from", "to", "amount", "currency", "timestamp"}
    for item, line in zip(data["transactions"], transfer_lines):
        if set(item) != expected_keys:
            raise ValueError("Unexpected archived transaction fields")
        payment = re.search(r" via: (.*?) timestamp: ", line)
        if not payment:
            raise ValueError("Missing archived payment format")
        txn = ObservableTransaction(
            edge_id=item["id"], source=item["from"], destination=item["to"],
            amount=str(item["amount"]), currency=item["currency"],
            timestamp=item["timestamp"], payment_format=payment.group(1),
        )
        if txn.historical_line() != line:
            raise ValueError(f"Ambiguous archive mapping at {txn.edge_id}")
        recovered.append(txn)
    if len({t.edge_id for t in recovered}) != len(recovered):
        raise ValueError("Duplicate archived transaction identity")
    accounts = {node for t in recovered for node in (t.source, t.destination)}
    expected_memberships = Counter(
        f"- acct_{node.split('_', 1)[1]} belongs_to bank_{node.split('_', 1)[0]}"
        for node in accounts
    )
    if expected_memberships != Counter(line for line in lines if " belongs_to " in line):
        raise ValueError("Archived endpoint memberships do not agree")
    expected_nodes = Counter(f"- acct_{node.split('_', 1)[1]} (type: Account)"
                             for node in accounts)
    expected_nodes.update(f"- bank_{bank} (type: Bank)"
                          for bank in {n.split('_', 1)[0] for n in accounts})
    if expected_nodes != Counter(line for line in lines if " (type: " in line):
        raise ValueError("Archived node sets do not agree")
    if data["statistics"]["n_transactions"] != len(recovered):
        raise ValueError("Archived statistics transaction count mismatch")
    return recovered


@dataclass(frozen=True)
class TargetedGraph:
    case_id: str
    target: ObservableTransaction
    transactions: tuple[ObservableTransaction, ...]
    target_was_present: bool
    removed_edge_ids: tuple[str, ...]
    original_transaction_count: int


def preserve_target(case_id: str, transactions: Sequence[ObservableTransaction],
                    target: ObservableTransaction,
                    max_transactions: int | None = None) -> TargetedGraph:
    """Keep the target and earliest context edges within an explicit edge budget.

    With no explicit budget, preserve the archived transaction count (or one
    for an empty context). A missing target replaces the latest context edge,
    breaking timestamp ties by numeric edge ID. Existing targets are verified
    against source fields, never trusted on their ID alone. Duplicate edge IDs
    fail instead of being silently collapsed.
    """
    if not _CASE.fullmatch(case_id):
        raise ValueError("Invalid case ID")
    if target.amount_received is None:
        raise ValueError("Target must include all observable source CSV fields")
    ids = [t.edge_id for t in transactions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate transaction identity, including possible target")
    present = target.edge_id in ids
    if present and not transactions[ids.index(target.edge_id)].same_graph_fields(target):
        raise ValueError("Archived target attributes disagree with original CSV")
    budget = max(1, len(transactions)) if max_transactions is None else max_transactions
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("Transaction budget must reserve at least one target edge")
    key = lambda t: (t.timestamp, int(t.edge_id[2:]))
    candidates = sorted((t for t in transactions if t.edge_id != target.edge_id), key=key)
    kept = candidates[:budget - 1]
    removed = tuple(t.edge_id for t in candidates[budget - 1:])
    result = tuple(sorted([*kept, target], key=key))
    return TargetedGraph(case_id, target, result, present, removed, len(transactions))


def _edge_line(txn: ObservableTransaction, is_target: bool) -> str:
    marker = " [TARGET]" if is_target else ""
    line = (f"- edge_id: {txn.edge_id}{marker} acct_{txn.source} transfers_to "
            f"acct_{txn.destination} amount_paid: {txn.amount} {txn.currency} "
            f"via: {txn.payment_format} timestamp: {txn.timestamp}")
    if is_target:
        line += (f" amount_received: {txn.amount_received} "
                 f"receiving_currency: {txn.receiving_currency}")
    return line


def serialise_targeted_graph(graph: TargetedGraph) -> str:
    """Render target identity once as an edge marker and include both endpoints."""
    txns = graph.transactions
    if sum(t.edge_id == graph.target.edge_id for t in txns) != 1:
        raise ValueError("Target must occur exactly once")
    if len({t.edge_id for t in txns}) != len(txns):
        raise ValueError("Duplicate transaction identity")
    if next(t for t in txns if t.edge_id == graph.target.edge_id) != graph.target:
        raise ValueError("Target attributes were changed after source verification")
    accounts = sorted({n for t in txns for n in (t.source, t.destination)})
    banks = sorted({n.split('_', 1)[0] for n in accounts})
    lines = [f"=== Transaction Subgraph (Case: {graph.case_id}) ===", "",
             f"Serialization version: {VERSION}",
             f"Target transaction ID: {graph.target.edge_id}",
             f"Target source account: acct_{graph.target.source}",
             f"Target destination account: acct_{graph.target.destination}", "",
             "**Nodes:**"]
    lines.extend(f"- acct_{n} (type: Account)" for n in accounts)
    lines.extend(f"- bank_{bank} (type: Bank)" for bank in banks)
    lines.extend(["", "**Edges:**"])
    lines.extend(f"- acct_{n} belongs_to bank_{n.split('_', 1)[0]}" for n in accounts)
    lines.extend(_edge_line(t, t.edge_id == graph.target.edge_id) for t in txns)
    out_degree = Counter(t.source for t in txns)
    in_degree = Counter(t.destination for t in txns)
    volume = sum((Decimal(t.amount) for t in txns), Decimal(0))
    start = min(t.timestamp for t in txns)
    end = max(t.timestamp for t in txns)
    elapsed = int((_timestamp(end) - _timestamp(start)).total_seconds())
    max_out = min(out_degree, key=lambda n: (-out_degree[n], n))
    max_in = min(in_degree, key=lambda n: (-in_degree[n], n))
    lines.extend(["", "SUMMARY STATISTICS:",
                  f"  Total transactions: {len(txns)}", f"  Unique accounts: {len(accounts)}",
                  f"  Total paid volume (mixed currencies): {volume:,.2f}",
                  f"  Avg paid amount (mixed currencies): {volume / len(txns):,.2f}",
                  f"  Start timestamp: {start}", f"  End timestamp: {end}",
                  f"  Elapsed time: {elapsed} seconds",
                  f"  Max out-degree: {out_degree[max_out]} (node: acct_{max_out})",
                  f"  Max in-degree: {in_degree[max_in]} (node: acct_{max_in})"])
    text = "\n".join(lines)
    validate_targeted_text(text, graph)
    return text


def validate_targeted_text(text: str, graph: TargetedGraph) -> None:
    """Check emitted target identity, attributes, endpoints, count and time span."""
    assert_evaluation_text(text)
    lines = text.splitlines()
    if sum(" [TARGET] " in line for line in lines) != 1:
        raise ValueError("Expected exactly one target edge marker")
    if lines.count(_edge_line(graph.target, True)) != 1:
        raise ValueError("Target line does not match verified source attributes")
    for node in (graph.target.source, graph.target.destination):
        if lines.count(f"- acct_{node} (type: Account)") != 1:
            raise ValueError("Target endpoint missing or duplicated")
        if f"- acct_{node} belongs_to bank_{node.split('_', 1)[0]}" not in lines:
            raise ValueError("Target bank membership missing")
    if sum(line.startswith("- edge_id: ") for line in lines) != len(graph.transactions):
        raise ValueError("Serialized transaction count mismatch")
    start = min(t.timestamp for t in graph.transactions)
    end = max(t.timestamp for t in graph.transactions)
    seconds = int((_timestamp(end) - _timestamp(start)).total_seconds())
    for required in (f"Target transaction ID: {graph.target.edge_id}",
                     f"Target source account: acct_{graph.target.source}",
                     f"Target destination account: acct_{graph.target.destination}",
                     f"  Start timestamp: {start}", f"  End timestamp: {end}",
                     f"  Elapsed time: {seconds} seconds"):
        if lines.count(required) != 1:
            raise ValueError("Target declaration or elapsed-time summary mismatch")


def evaluation_prompt(graph: TargetedGraph, graph_text: str | None = None) -> str:
    """A complete new zero-shot input; historical runners do not select it."""
    graph_text = serialise_targeted_graph(graph) if graph_text is None else graph_text
    validate_targeted_text(graph_text, graph)
    prompt = (
        "Assess the specified transaction for possible money laundering using the "
        "provided transaction graph. Focus your decision on transaction "
        f"{graph.target.edge_id}, from acct_{graph.target.source} to "
        f"acct_{graph.target.destination}, marked [TARGET]. Other transfers are "
        "context. Return a JSON object with suspicious_probability (a number "
        "between 0 and 1) and rationale (a brief explanation).\n\n" + graph_text
    )
    assert_evaluation_text(prompt)
    return prompt
