"""Turn the HT-Coreset into the prompts the LLMs were shown.

The coreset is a vector of positions into the full temporal test split. This
stage resolves each position to its AMLworld edge, wraps that edge in its k-hop
context graph, renders the graph as a typed graph
(:mod:`amlc.serialize.typed_graph`) and writes one JSON line per case.
Everything downstream -- the released evaluation table, the prompts on Hugging
Face, the LLM runner -- consumes what this stage writes.

Reads
    the archived run directory, through :func:`amlc.archive.legacy_path`:
    the coreset indices, the archived HT weights and the full-split labels;
    plus AMLworld itself, through a caller-supplied graph source.

Writes, into ``--out`` (default ``paths.coreset(dataset)``)
    ``cases_<format>.jsonl``  one record per case: identifier, centre edge,
                              label, typology, the rendered text, its token
                              estimate, the coreset position and the HT weight
    ``labels.npy``            0/1 illicit, in coreset row order
    ``ht_weights.npy``        Horvitz-Thompson weights, repaired
    ``case_index.csv``        the same fields as a join table
    ``metadata.json``         sizes, token statistics, typology counts

Two corrections against the pre-release ``serialize_v2_subset.py``
------------------------------------------------------------------

**Case identifiers.** The emitted ``case_id`` now comes from
:func:`amlc.case_ids.case_id`, so a regenerated prompt carries
``amlc_00417`` and matches the released artefacts. The pre-release built the
string inline in the pre-release vocabulary.

**Overwriting is explicit.** The pre-release guarded the arrays with a bare
existence check::

    if not labels_path.exists():
        np.save(labels_path, labels)
        np.save(weights_path, weights)

which reads as "do not redo work" but behaves as "keep whatever is on disk".
The weight vector was the casualty: a re-run with a repaired or re-drawn weight
vector wrote fresh prompts quoting fresh weights beside an ``.npy`` still
holding the old ones, and nothing said so. The guard is now ``--force``.
Without it an existing array is left alone and the run says which file it
skipped; with it every output is rewritten from this run.

**Weights come from the repair, never from the archive.** The archived
LI-Small weight vector double-counts the benign population (see
:mod:`amlc.coreset.ht_weights`), so the weight quoted in a prompt record
is the recomputed one. This is the rule
``scripts/build_hf_dataset.py`` follows, and it is why the array here is
called ``ht_weights.npy`` rather than the archive's ``weights.npy``: the two
names should never denote different vectors.

The AMLworld graph source
-------------------------

The graph is injected rather than imported, so a clone that only consumes the
released prompts never loads AMLworld. Pass an object satisfying
:class:`AMLworldGraph`, or on the command line a ``module:factory`` dotted path
resolved by :func:`resolve_source`. It defaults to
``amlc.data.loader:graph_source``, which opens a variant with this package's own
loader; the argument used to be required with no shipped factory to give it.

The GFP feature tensors the pre-release wrote alongside the prompts are not
produced here. No prompt depends on them, they duplicate the supervised
pipeline's own feature extraction, and the released copy is the one that
pipeline wrote.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from .. import config, paths
from ..case_ids import case_id
from ..coreset.ht_weights import repair_archived_weights
from ..archive import legacy_path
from ..config import CORESET_DRAWS, DATASETS
from .typed_graph import FORMATS, TYPED_GRAPH, Case, GraphSerialiser, approx_tokens

# Temporal split, as the supervised pipeline cuts it. Recorded in the metadata
# so a reader can tell which edges were eligible; nothing here re-derives it.
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2


class AMLworldGraph(Protocol):
    """What this stage needs from a loaded AMLworld variant.

    :func:`amlc.data.loader.graph_source` returns one. The interface is stated
    here rather than imported so that this stage can be driven by any loader,
    including one reading a format AMLworld does not use.
    """

    def test_split_start(self) -> int:
        """Row index of the first edge in the temporal test split."""

    def n_edges(self) -> int:
        """Number of edges in the variant, across all three splits."""

    def build_case(self, edge_id: str, case_id: str, label: int) -> Optional[Case]:
        """The k-hop context graph around one edge, or None if it cannot be built.

        Beyond what :class:`~amlc.serialize.typed_graph.Case` needs for
        rendering, the returned object carries ``center_edge_id``, ``label`` and
        ``typology``; those three go into the record and the join table.
        """


# Set in the parent before the pool forks; the children inherit the loaded
# graph instead of each rebuilding it, which is the only reason case building
# fits in memory.
_SOURCE: Optional[AMLworldGraph] = None


def _build_case_worker(args):
    edge_id, cid, label = args
    return _SOURCE.build_case(edge_id, cid, label)


def resolve_source(spec: str, dataset: str, k_hop: int,
                   max_neighbours: int) -> AMLworldGraph:
    """Import a graph-source factory from a ``module:factory`` dotted path.

    The factory is called as ``factory(dataset, k_hop=..., max_neighbours=...)``
    and must return an :class:`AMLworldGraph`.
    """
    if ":" not in spec:
        raise ValueError(
            f"--loader takes module:factory, got {spec!r}. It names the callable "
            "that opens an AMLworld variant and returns a graph source; see "
            "AMLworldGraph in this module for the interface."
        )
    module_name, attr = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attr)
    return factory(dataset, k_hop=k_hop, max_neighbours=max_neighbours)


def load_draw(archive: Path, dataset: str, draw: str = "ht-coreset") -> dict:
    """Coreset positions, repaired HT weights and labels for one draw."""
    repair = repair_archived_weights(archive, dataset, draw)
    subset_idx = repair["subset_idx"]
    labels = np.load(legacy_path(archive, "test_labels", dataset))
    return {
        "subset_idx": subset_idx,
        "weights": repair["fixed"],
        "archived_weights": repair["saved"],
        "labels": labels[subset_idx],
        "n_test_full": int(len(labels)),
        "repair": repair["report"],
    }


def edge_ids_for(subset_idx: np.ndarray, test_split_start: int) -> list[str]:
    """Coreset position -> AMLworld edge id.

    Position ``j`` is the ``j``-th edge of the temporal test split, which is row
    ``test_split_start + j`` of the transaction file, and edge ids are that row
    number.
    """
    return [f"e_{test_split_start + int(j)}" for j in subset_idx]


def build_cases(source: AMLworldGraph, edge_ids: Sequence[str],
                labels: np.ndarray, workers: int = 1) -> list[Optional[Case]]:
    """Build one context graph per coreset row, in coreset row order."""
    global _SOURCE

    args = [(eid, case_id(i), int(lbl))
            for i, (eid, lbl) in enumerate(zip(edge_ids, labels))]

    # The pool costs a fork of the loaded graph, which only pays off on a real
    # coreset; below that the sequential path is both faster and easier to trace.
    if workers > 1 and len(args) > 100:
        _SOURCE = source
        try:
            with mp.Pool(workers) as pool:
                cases = list(pool.imap(_build_case_worker, args, chunksize=64))
        finally:
            _SOURCE = None
        return cases

    return [source.build_case(eid, cid, lbl) for eid, cid, lbl in args]


def serialise_cases(cases: Sequence[Case],
                    formats: Sequence[str] = (TYPED_GRAPH,)) -> dict:
    """Render every case in every requested format."""
    out = {}
    for fmt in formats:
        serialiser = GraphSerialiser(fmt)
        rendered = []
        for case in cases:
            text = serialiser.serialise(case, include_stats=True)
            rendered.append({"text": text, "n_tokens": approx_tokens(text)})
        out[fmt] = rendered
    return out


def _write_guarded(path: Path, write, force: bool) -> bool:
    """Write unless the file exists and ``force`` is off. Returns whether it wrote."""
    if path.exists() and not force:
        print(f"  kept   {path.name} (already exists; pass --force to rewrite)")
        return False
    write(path)
    print(f"  wrote  {path.name}")
    return True


def write_outputs(out_dir: Path, cases: Sequence[Case], serialised: dict,
                  subset_idx: np.ndarray, weights: np.ndarray,
                  labels: np.ndarray, dataset: str, n_test_full: int,
                  k_hop: int, max_neighbours: int, draw: str,
                  force: bool = False) -> dict:
    """Write the prompt files and the arrays that go with them."""
    paths.ensure(out_dir)

    format_stats = {}
    for fmt, rendered in serialised.items():
        jsonl = out_dir / f"cases_{fmt.replace('-', '_')}.jsonl"
        counts = []
        with jsonl.open("w", encoding="utf-8") as f:
            for i, (case, ser) in enumerate(zip(cases, rendered)):
                f.write(json.dumps({
                    "case_id": case.case_id,
                    "center_edge_id": case.center_edge_id,
                    "label": case.label,
                    "typology": case.typology,
                    "n_transactions": len(case.transactions),
                    "format": fmt,
                    "text": ser["text"],
                    "n_tokens_approx": ser["n_tokens"],
                    "subset_index": int(subset_idx[i]),
                    "weight": float(weights[i]),
                }, ensure_ascii=False) + "\n")
                counts.append(ser["n_tokens"])
        format_stats[fmt] = {
            "n_cases": len(cases),
            "avg_tokens": int(np.mean(counts)),
            "total_tokens": int(np.sum(counts)),
            "min_tokens": int(np.min(counts)),
            "max_tokens": int(np.max(counts)),
            "median_tokens": int(np.median(counts)),
        }
        print(f"  wrote  {jsonl.name} "
              f"(avg {format_stats[fmt]['avg_tokens']:,} tokens, "
              f"total {format_stats[fmt]['total_tokens']:,})")

    _write_guarded(out_dir / "labels.npy", lambda p: np.save(p, labels), force)
    _write_guarded(out_dir / "ht_weights.npy", lambda p: np.save(p, weights), force)

    def _write_index(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "center_edge_id", "label", "typology",
                        "n_transactions", "subset_index", "weight"])
            for i, case in enumerate(cases):
                w.writerow([case.case_id, case.center_edge_id, case.label,
                            case.typology or "", len(case.transactions),
                            int(subset_idx[i]), f"{weights[i]:.4f}"])

    _write_guarded(out_dir / "case_index.csv", _write_index, force)

    typologies: dict[str, int] = {}
    for case in cases:
        if case.typology:
            typologies[case.typology] = typologies.get(case.typology, 0) + 1
    txn_counts = [len(c.transactions) for c in cases]

    metadata = {
        "dataset": dataset,
        "draw": draw,
        "n_cases": len(cases),
        "n_illicit": int(labels.sum()),
        "n_legit": int(len(labels) - labels.sum()),
        "n_test_full": n_test_full,
        "reduction_pct": round((1 - len(cases) / n_test_full) * 100, 2),
        "k_hop": k_hop,
        "max_neighbours_per_hop": max_neighbours,
        "formats": format_stats,
        "temporal_split": {
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": round(1 - TRAIN_RATIO - VAL_RATIO, 2),
        },
        "typology_distribution": typologies,
        "transaction_stats": {
            "avg": round(float(np.mean(txn_counts)), 1),
            "min": int(np.min(txn_counts)),
            "max": int(np.max(txn_counts)),
            "median": int(np.median(txn_counts)),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False))
    print("  wrote  metadata.json")
    return metadata


def build(dataset: str, source: AMLworldGraph, archive: Path,
          out_dir: Optional[Path] = None,
          formats: Sequence[str] = (TYPED_GRAPH,),
          draw: str = "ht-coreset", workers: int = 1,
          k_hop: int = config.K_HOP,
          max_neighbours: int = config.MAX_NEIGHBOURS_PER_HOP,
          force: bool = False) -> dict:
    """Serialise one dataset's coreset. Returns the metadata that was written."""
    t0 = time.time()
    out_dir = out_dir or paths.coreset(dataset)
    print(f"\n{dataset}: serialising the {draw} draw")

    drawn = load_draw(archive, dataset, draw)
    subset_idx, weights, labels = drawn["subset_idx"], drawn["weights"], drawn["labels"]
    print(f"  {len(subset_idx):,} rows, {int(labels.sum()):,} illicit, "
          f"weights sum to {weights.sum():,.0f}")

    # A source pointed at the wrong variant, or split differently, would build
    # context graphs around the wrong edges and say nothing about it.
    n_test = source.n_edges() - source.test_split_start()
    if n_test != drawn["n_test_full"]:
        raise ValueError(
            f"{dataset}: the graph source holds {n_test:,} test edges but the "
            f"archived label vector holds {drawn['n_test_full']:,}. The source "
            "is a different variant or a different temporal split."
        )
    if n_test != config.N_TEST_FULL[dataset]:
        raise ValueError(
            f"{dataset}: the graph source holds {n_test:,} test edges, the paper "
            f"reports {config.N_TEST_FULL[dataset]:,}."
        )

    edge_ids = edge_ids_for(subset_idx, source.test_split_start())
    built = build_cases(source, edge_ids, labels, workers=workers)

    # An edge whose context graph cannot be built drops out, and every array
    # has to drop the same row or the coreset stops lining up with its weights.
    keep = [i for i, c in enumerate(built) if c is not None]
    if len(keep) != len(built):
        print(f"  WARNING: {len(built) - len(keep)} edges built no case and are dropped")
        subset_idx, weights, labels = subset_idx[keep], weights[keep], labels[keep]
    cases = [built[i] for i in keep]
    print(f"  built  {len(cases):,} cases")

    serialised = serialise_cases(cases, formats)
    metadata = write_outputs(out_dir, cases, serialised, subset_idx, weights,
                             labels, dataset, drawn["n_test_full"], k_hop,
                             max_neighbours, draw, force=force)
    print(f"  done in {time.time() - t0:.1f}s -> {out_dir}")
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Serialise the HT-Coreset into typed-graph prompts.")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--draw", default="ht-coreset", choices=CORESET_DRAWS)
    ap.add_argument("--loader", default="amlc.data.loader:graph_source", metavar="MODULE:FACTORY",
                    help="dotted path to the AMLworld graph-source factory, "
                         "called as factory(dataset, k_hop=, max_neighbours=). Defaults to the AMLworld loader this package ships")
    ap.add_argument("--formats", nargs="+", default=[TYPED_GRAPH], choices=FORMATS,
                    help="default: the typed graph alone, the only format the "
                         "paper reports")
    ap.add_argument("--archive", type=Path, default=None,
                    help="the archived run directory's outputs/; defaults to "
                         "$AMLC_ARCHIVE")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to data/coreset/<dataset>")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes used to build context graphs")
    ap.add_argument("--k-hop", type=int, default=config.K_HOP)
    ap.add_argument("--max-neighbours", type=int, default=config.MAX_NEIGHBOURS_PER_HOP)
    ap.add_argument("--force", action="store_true",
                    help="rewrite labels.npy, ht_weights.npy and case_index.csv "
                         "instead of keeping the copies already on disk")
    args = ap.parse_args(argv)

    archive = args.archive or paths.archive()
    for dataset in args.datasets:
        source = resolve_source(args.loader, dataset, args.k_hop, args.max_neighbours)
        build(dataset, source, archive,
              out_dir=args.out / dataset if args.out else None,
              formats=args.formats, draw=args.draw, workers=args.workers,
              k_hop=args.k_hop, max_neighbours=args.max_neighbours,
              force=args.force)


if __name__ == "__main__":
    sys.exit(main())
