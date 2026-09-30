"""Turn a k-hop context graph into the text an LLM is shown.

The paper evaluates exactly one serialisation, the **typed graph** of Pirmorad
et al. (2025): accounts and banks as typed nodes, ``transfers_to`` and
``belongs_to`` as typed edges, with amount, currency, payment method and
timestamp carried on the transfer::

    === Transaction Subgraph (Case: amlc_00417) ===

    **Nodes:**
    - acct_80BB1E9D0 (type: Account)
    - bank_0024424 (type: Bank)

    **Edges:**
    - acct_80BB1E9D0 belongs_to bank_0024424
    - acct_80BB1E9D0 transfers_to acct_80F19DAE0 amount: 1200.00 US Dollar
      via: Cheque timestamp: 2022/09/01 00:20

    SUMMARY STATISTICS:
      ...

:data:`TYPED_GRAPH` is that format and is the default everywhere in this
package. Four other renderings survive behind the ``fmt`` argument -- an edge
list, an XML-ish structured form, JSON, and an adjacency list. **No number in
the paper was produced with any of them.** They were built for a
format-comparison study that was cut, and they are kept only because the
archived run directory holds a ``cases_<format>.jsonl`` for each one and a
reader may want to regenerate those files. Anything reported, and anything the
released evaluation table carries, is the typed graph.

The pre-release serialiser is ``graph_serializer.py``; the typed-graph branch
here is a faithful copy of its ``_serialize_paper_format``, down to the two
spaces and the ``:`` placement, because the released prompts are byte-compared
against the executed ones. This frozen format does not mark the target
transaction, and its ``Time span`` field counts distinct timestamp strings
rather than elapsed time. The source context can omit the target after capping;
this serialiser does not restore it.

Input is any object shaped like the pre-release ``Case``: a ``case_id`` and a
sequence of transactions. The package does not vendor the AMLworld loader that
produces them, so the shape is stated here as a Protocol rather than imported.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional, Protocol, Sequence, runtime_checkable

#: The typed graph of Pirmorad et al. (2025). The only format the paper reports.
TYPED_GRAPH = "typed-graph"

#: Every renderer, primary first. The four after the typed graph back no
#: reported number; see the module docstring.
FORMATS = (TYPED_GRAPH, "edge-list", "structured", "json", "adjacency")


@runtime_checkable
class Transaction(Protocol):
    """One directed transfer inside a context graph."""

    edge_id: str
    source: str
    target: str
    amount: float
    timestamp: str
    currency: str
    payment_format: Optional[str]


@runtime_checkable
class Case(Protocol):
    """A case identifier and extracted context, which may omit its focal edge."""

    case_id: str
    transactions: Sequence[Transaction]


def approx_tokens(text: str) -> int:
    """Token estimate at four characters per token.

    Crude, and deliberately so: it is what the pre-release pipeline recorded in
    ``n_tokens_approx`` and in the token totals quoted for the coreset, so the
    released files stay comparable with the archived ones. Use a real tokeniser
    for anything that has to be exact.
    """
    return max(1, len(text) // 4)


class GraphSerialiser:
    """Render cases in one format. Stateless apart from the format choice."""

    def __init__(self, fmt: str = TYPED_GRAPH):
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        self.fmt = fmt

    def serialise(self, case: Case, include_stats: bool = True) -> str:
        if self.fmt == TYPED_GRAPH:
            return self._typed_graph(case, include_stats)
        if self.fmt == "edge-list":
            return self._edge_list(case, include_stats)
        if self.fmt == "structured":
            return self._structured(case, include_stats)
        if self.fmt == "json":
            return self._json(case, include_stats)
        return self._adjacency(case, include_stats)

    # ── the format the paper reports ────────────────────────────────────

    @staticmethod
    def _split_node_id(node_id: str) -> tuple[str, str]:
        """Split a merged ``bankCode_accountCode`` node id into its parts."""
        parts = node_id.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", node_id

    def _typed_graph(self, case: Case, include_stats: bool) -> str:
        """Typed nodes and typed edges, after Pirmorad et al. (2025)."""
        accounts = {}   # full node id -> (bank code, account code)
        banks = set()

        for txn in case.transactions:
            for nid in (txn.source, txn.target):
                if nid not in accounts:
                    bank, acct = self._split_node_id(nid)
                    accounts[nid] = (bank, acct)
                    banks.add(bank)

        lines = [f"=== Transaction Subgraph (Case: {case.case_id}) ===\n"]

        lines.append("**Nodes:**")
        for nid in sorted(accounts):
            _, acct = accounts[nid]
            lines.append(f"- acct_{acct} (type: Account)")
        for bank in sorted(banks):
            lines.append(f"- bank_{bank} (type: Bank)")

        lines.append("\n**Edges:**")

        # One belongs_to per account, deduplicated on (account, bank) rather
        # than on the node id, so the same account at the same bank reached
        # through two node ids is stated once.
        seen = set()
        for nid in sorted(accounts):
            bank, acct = accounts[nid]
            key = (acct, bank)
            if key not in seen:
                lines.append(f"- acct_{acct} belongs_to bank_{bank}")
                seen.add(key)

        for txn in sorted(case.transactions, key=lambda t: t.timestamp):
            _, src_acct = accounts[txn.source]
            _, dst_acct = accounts[txn.target]
            pay_fmt = getattr(txn, "payment_format", None) or "Unknown"
            lines.append(
                f"- acct_{src_acct} transfers_to acct_{dst_acct} "
                f"amount: {txn.amount:.2f} {txn.currency} via: {pay_fmt} "
                f"timestamp: {txn.timestamp}"
            )

        if include_stats:
            lines.append("\n" + self._stats_text(case))

        return "\n".join(lines)

    # ── the four unreported formats ─────────────────────────────────────

    def _edge_list(self, case: Case, include_stats: bool) -> str:
        lines = [f"=== Transaction Subgraph (Case: {case.case_id}) ===\n"]
        lines.append("TRANSACTIONS (sorted by time):")
        lines.append("-" * 60)
        lines.append(f"{'ID':<10} {'From':<12} {'To':<12} {'Amount':>12} "
                     f"{'Currency':<6} {'Time':>8}")
        lines.append("-" * 60)
        for txn in sorted(case.transactions, key=lambda t: t.timestamp):
            lines.append(
                f"{txn.edge_id:<10} {txn.source:<12} {txn.target:<12} "
                f"{txn.amount:>12,.2f} {txn.currency:<6} {txn.timestamp:>8}"
            )
        if include_stats:
            lines.append("\n" + self._stats_text(case))
        return "\n".join(lines)

    def _structured(self, case: Case, include_stats: bool) -> str:
        lines = [f"<case id=\"{case.case_id}\">"]

        nodes = set()
        for txn in case.transactions:
            nodes.add(txn.source)
            nodes.add(txn.target)

        lines.append("  <nodes>")
        for node in sorted(nodes):
            lines.append(f"    <node id=\"{node}\"/>")
        lines.append("  </nodes>")

        lines.append("  <transactions>")
        for txn in sorted(case.transactions, key=lambda t: t.timestamp):
            lines.append(
                f"    <txn id=\"{txn.edge_id}\" "
                f"from=\"{txn.source}\" to=\"{txn.target}\" "
                f"amount=\"{txn.amount:.2f}\" currency=\"{txn.currency}\" "
                f"time=\"{txn.timestamp}\"/>"
            )
        lines.append("  </transactions>")

        if include_stats:
            lines.append("  <statistics>")
            for k, v in self._stats(case).items():
                lines.append(f"    <{k}>{v}</{k}>")
            lines.append("  </statistics>")

        lines.append("</case>")
        return "\n".join(lines)

    def _json(self, case: Case, include_stats: bool) -> str:
        data = {"case_id": case.case_id, "transactions": []}
        for txn in sorted(case.transactions, key=lambda t: t.timestamp):
            data["transactions"].append({
                "id": txn.edge_id,
                "from": txn.source,
                "to": txn.target,
                "amount": txn.amount,
                "currency": txn.currency,
                "timestamp": txn.timestamp,
            })
        if include_stats:
            data["statistics"] = self._stats(case)
        return json.dumps(data, indent=2)

    def _adjacency(self, case: Case, include_stats: bool) -> str:
        lines = [f"Case: {case.case_id}",
                 "\nAdjacency List (node -> [outgoing transactions]):\n"]
        adj = defaultdict(list)
        for txn in case.transactions:
            adj[txn.source].append(txn)
        for node in sorted(adj.keys()):
            lines.append(f"{node}:")
            for txn in sorted(adj[node], key=lambda t: t.timestamp):
                lines.append(f"  -> {txn.target} [{txn.edge_id}]: "
                             f"{txn.amount:.2f} {txn.currency}")
        if include_stats:
            lines.append("\n" + self._stats_text(case))
        return "\n".join(lines)

    # ── summary statistics, appended to every format ────────────────────

    def _stats_text(self, case: Case) -> str:
        s = self._stats(case)
        return "\n".join([
            "SUMMARY STATISTICS:",
            f"  Total transactions: {s['n_transactions']}",
            f"  Unique accounts: {s['n_accounts']}",
            f"  Total volume: {s['total_volume']:,.2f}",
            f"  Avg transaction: {s['avg_amount']:,.2f}",
            f"  Time span: {s['time_span']} units",
            f"  Max out-degree: {s['max_out_degree']} (node: {s['max_out_node']})",
            f"  Max in-degree: {s['max_in_degree']} (node: {s['max_in_node']})",
        ])

    @staticmethod
    def _stats(case: Case) -> dict:
        txns = case.transactions

        nodes = set()
        out_degree: dict[str, int] = defaultdict(int)
        in_degree: dict[str, int] = defaultdict(int)
        for txn in txns:
            nodes.add(txn.source)
            nodes.add(txn.target)
            out_degree[txn.source] += 1
            in_degree[txn.target] += 1

        amounts = [t.amount for t in txns]
        timestamps = [t.timestamp for t in txns]

        max_out_node = max(out_degree, key=out_degree.get) if out_degree else None
        max_in_node = max(in_degree, key=in_degree.get) if in_degree else None

        # AMLworld timestamps are strings in the CSV and integers in the
        # synthetic fixtures, so the span is a subtraction where that works and
        # a count of distinct stamps where it does not.
        time_span = 0
        if timestamps:
            try:
                time_span = max(timestamps) - min(timestamps)
            except TypeError:
                time_span = len(set(timestamps))

        return {
            "n_transactions": len(txns),
            "n_accounts": len(nodes),
            "total_volume": sum(amounts),
            "avg_amount": sum(amounts) / len(amounts) if amounts else 0,
            "min_amount": min(amounts) if amounts else 0,
            "max_amount": max(amounts) if amounts else 0,
            "time_span": time_span,
            "max_out_degree": max(out_degree.values()) if out_degree else 0,
            "max_out_node": max_out_node,
            "max_in_degree": max(in_degree.values()) if in_degree else 0,
            "max_in_node": max_in_node,
        }


