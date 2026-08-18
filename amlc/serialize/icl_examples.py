"""Build the in-context demonstration pool: 8 suspicious plus 4 non-suspicious.

ICL-FS and ICL-V show the model twelve worked examples before the case under
test: one illicit context graph per AMLworld typology, and four benign ones.
Each carries a one-sentence explanation naming the pattern. This module draws
that pool and writes it to ``data/icl_examples/<dataset>/icl_examples.json``,
which :func:`amlc.llm.prompts.load_icl_examples` reads and the templates
in ``prompts/`` render.

**Every demonstration comes from the training split.** The pool is fixed once
and then appears in the prompt of every test case, so a single test-split edge
in it would leak the evaluation set into all 3,753 (HI-Small) or 2,268
(LI-Small) prompts. :func:`generate` asserts the split of each sampled edge and
refuses to write if any of them is not ``train``.

The demonstrations are rendered as typed graphs, the same serialiser the test
cases use, but with a tighter neighbourhood: k=1 and 15 neighbours per hop
against the k=2 and 50 of a test case. Twelve full-size context graphs would
cost more prompt budget than the case being judged. The pool as shipped is
about 30K tokens on HI-Small and 22K on LI-Small, which is what pushes the
largest cases over the context window of the 128K models; see the truncation
handling in :mod:`amlc.llm.clients`.

The graph source is supplied by the caller, exactly as in
:mod:`amlc.serialize.build_coreset_prompts`, because the package does not
vendor the AMLworld loader.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Protocol, Sequence

from .. import config, paths
from ..config import DATASETS
from ..typology import TYPOLOGY_CLASSES
from .build_coreset_prompts import AMLworldGraph, resolve_source
from .typed_graph import TYPED_GRAPH, GraphSerialiser, approx_tokens

#: Order the demonstrations appear in, which is the order the pre-release
#: generator iterated its typology list and therefore the order frozen into the
#: released pool and into every executed prompt. It is NOT the integer encoding
#: order of :mod:`amlc.typology`, and swapping one for the other would
#: reorder the few-shot block without changing anything else.
DEMONSTRATION_ORDER = (
    "fan-out", "fan-in", "scatter-gather", "gather-scatter",
    "cycle", "random", "bipartite", "stack",
)

if set(DEMONSTRATION_ORDER) != set(TYPOLOGY_CLASSES):
    raise ImportError(
        "DEMONSTRATION_ORDER must be a permutation of "
        "amlc.typology.TYPOLOGY_CLASSES"
    )

#: The pool is drawn once, under the first experiment seed. The five-seed
#: protocol varies LLM sampling, not the demonstrations: every seed of every
#: model saw this one pool.
DEFAULT_SEED = config.SEEDS[0]

#: Compact settings for a demonstration, against K_HOP / MAX_NEIGHBOURS_PER_HOP
#: for a test case.
ICL_K_HOP = 1
ICL_MAX_NEIGHBOURS = 15

#: Explanation shown under each suspicious demonstration, one per typology,
#: phrased after Pirmorad et al. (2025).
SUSPICIOUS_EXPLANATIONS = {
    "fan-out": (
        "This pattern resembles fan-out behavior, where a single account "
        "disperses funds to many accounts within a short window, often to "
        "obscure the origin."
    ),
    "fan-in": (
        "Multiple accounts consolidating into one destination, consistent "
        "with the collection phase of money laundering (fan-in)."
    ),
    "scatter-gather": (
        "Fan-out from source to intermediaries, then fan-in to destination "
        "indicates a layering structure (scatter-gather)."
    ),
    "gather-scatter": (
        "This reflects a gather-scatter pattern, where funds from multiple "
        "sources converge briefly before dispersing again, which is a common "
        "layering tactic."
    ),
    "cycle": (
        "This shows a cycle pattern where funds return to the origin after "
        "passing through intermediaries, typical of round-tripping schemes."
    ),
    "random": (
        "Structured transactions just below reporting thresholds indicate "
        "intentional structuring/smurfing (random)."
    ),
    "bipartite": (
        "A highly connected bipartite structure, consistent with mule "
        "networks or coordinated burst activity."
    ),
    "stack": (
        "Multiple sequential layers with decreasing amounts suggest "
        "commission-based laundering chain (stack)."
    ),
}

#: Explanations for the four benign demonstrations, in draw order.
NON_SUSPICIOUS_EXPLANATIONS = [
    (
        "These transfers occur across a regular schedule, between known "
        "parties with consistent values. No indicators of laundering behavior."
    ),
    (
        "Regular business payments to known vendors with varying amounts "
        "consistent with normal operations."
    ),
    (
        "Disbursements from registered investment fund to verified clients, "
        "consistent with legitimate returns."
    ),
    (
        "Large bilateral transfers between banks are normal for settlement "
        "and liquidity management."
    ),
]

N_NON_SUSPICIOUS = len(NON_SUSPICIOUS_EXPLANATIONS)


class TrainingSplit(AMLworldGraph, Protocol):
    """What drawing a demonstration pool needs, on top of :class:`AMLworldGraph`."""

    def train_illicit_edges(self) -> Sequence[str]:
        """Edge ids of every illicit edge in the training split."""

    def train_benign_edges(self) -> Sequence[str]:
        """Edge ids of every benign edge in the training split."""

    def typology_of(self, edge_id: str) -> Optional[str]:
        """Typology name of an illicit edge, or None if it carries no pattern."""

    def split_of(self, edge_id: str) -> str:
        """``train``, ``val`` or ``test``."""


def draw_edges(source: TrainingSplit, seed: int = DEFAULT_SEED
               ) -> tuple[list[tuple[str, str]], list[str]]:
    """Pick one illicit edge per typology and four benign ones, from train only.

    Returns ``([(edge_id, typology), ...], [edge_id, ...])``. Sampling uses
    :class:`random.Random` seeded per call, so a rerun with the same seed and
    the same training split reproduces the pool exactly.
    """
    rng = random.Random(seed)

    pools: dict[str, list[str]] = defaultdict(list)
    for eid in source.train_illicit_edges():
        typ = source.typology_of(eid)
        if typ:
            pools[typ].append(eid)

    suspicious = []
    for typ in DEMONSTRATION_ORDER:
        pool = pools.get(typ, [])
        if not pool:
            print(f"  WARNING: no training edge for typology {typ!r}, skipping it")
            continue
        suspicious.append((rng.choice(pool), typ))

    benign_pool = list(source.train_benign_edges())
    rng.shuffle(benign_pool)
    benign = benign_pool[:N_NON_SUSPICIOUS]

    for eid in [e for e, _ in suspicious] + benign:
        split = source.split_of(eid)
        if split != "train":
            raise AssertionError(
                f"leakage: demonstration edge {eid} is in the {split!r} split. "
                "A demonstration outside train appears in every test prompt."
            )
    return suspicious, benign


def generate(dataset: str, source: TrainingSplit, seed: int = DEFAULT_SEED,
             out_path: Optional[Path] = None, write: bool = True) -> dict:
    """Draw, render and (unless ``write`` is off) save one dataset's pool."""
    print(f"\n{dataset}: drawing the demonstration pool (seed {seed})")
    suspicious, benign = draw_edges(source, seed)
    serialiser = GraphSerialiser(TYPED_GRAPH)

    suspicious_examples = []
    for eid, typ in suspicious:
        case = source.build_case(eid, f"icl_sus_{typ}", 1)
        if case is None:
            print(f"  WARNING: no case for {eid} ({typ}), dropping it")
            continue
        text = serialiser.serialise(case, include_stats=True)
        suspicious_examples.append({
            "typology": typ,
            "edge_id": eid,
            "case_id": case.case_id,
            "serialized_text": text,
            "explanation": SUSPICIOUS_EXPLANATIONS[typ],
            "n_transactions": len(case.transactions),
            "n_tokens_approx": approx_tokens(text),
        })
        print(f"  {typ:<15s} {eid}: {len(case.transactions)} transactions, "
              f"~{approx_tokens(text):,} tokens")

    non_suspicious_examples = []
    for i, eid in enumerate(benign):
        case = source.build_case(eid, f"icl_legit_{i}", 0)
        if case is None:
            print(f"  WARNING: no case for {eid}, dropping it")
            continue
        text = serialiser.serialise(case, include_stats=True)
        non_suspicious_examples.append({
            "edge_id": eid,
            "case_id": case.case_id,
            "serialized_text": text,
            "explanation": NON_SUSPICIOUS_EXPLANATIONS[i],
            "n_transactions": len(case.transactions),
            "n_tokens_approx": approx_tokens(text),
        })
        print(f"  legit_{i:<9d} {eid}: {len(case.transactions)} transactions, "
              f"~{approx_tokens(text):,} tokens")

    pool = {
        "dataset": dataset,
        "seed": seed,
        "k_hop": ICL_K_HOP,
        "max_neighbours_per_hop": ICL_MAX_NEIGHBOURS,
        "format": TYPED_GRAPH,
        "n_suspicious": len(suspicious_examples),
        "n_non_suspicious": len(non_suspicious_examples),
        "total_tokens_approx": sum(
            e["n_tokens_approx"]
            for e in suspicious_examples + non_suspicious_examples),
        "suspicious_examples": suspicious_examples,
        "non_suspicious_examples": non_suspicious_examples,
    }
    print(f"  pool: {pool['n_suspicious']} suspicious + "
          f"{pool['n_non_suspicious']} non-suspicious, "
          f"~{pool['total_tokens_approx']:,} tokens of prompt overhead per case")

    if write:
        out_path = out_path or paths.icl_examples(dataset)
        paths.ensure(out_path.parent)
        out_path.write_text(json.dumps(pool, indent=2), encoding="utf-8")
        print(f"  wrote  {out_path}")
    return pool


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Draw the in-context demonstration pool from the training split.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--loader", default="amlc.data.loader:graph_source", metavar="MODULE:FACTORY",
                    help="dotted path to the AMLworld graph-source factory, "
                         "called as factory(dataset, k_hop=, max_neighbours=). Defaults to the AMLworld loader this package ships")
    ap.add_argument("--dry-run", action="store_true",
                    help="render an ICL-FS prompt with the drawn pool and print "
                         "it, instead of writing the pool")
    args = ap.parse_args(argv)

    for dataset in args.datasets:
        source = resolve_source(args.loader, dataset, ICL_K_HOP, ICL_MAX_NEIGHBOURS)
        pool = generate(dataset, source, seed=args.seed, write=not args.dry_run)
        if args.dry_run:
            from ..llm.prompts import render
            prompt = render("ICL-FS", "<<the test case would go here>>", pool)
            print(prompt)
            print(f"\n  prompt length ~{approx_tokens(prompt):,} tokens")


if __name__ == "__main__":
    sys.exit(main())
