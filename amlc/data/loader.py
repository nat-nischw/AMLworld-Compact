"""AMLworld transactions to context graphs, under the temporal split.

Reads the two files AMLworld ships per variant, ``<dataset>_Trans.csv`` and
``<dataset>_Patterns.txt``, both located by
:mod:`amlc.data.download_amlworld`. Writes nothing. Everything
downstream, the supervised baselines, the HT-Coreset construction and the
prompt serialisation, takes the in-memory graph and the :class:`Case` objects
from here.

What a case is
--------------
One case is one focal edge plus the context graph around it: a k-hop BFS from
the focal edge's source account, capped at
:data:`amlc.config.MAX_NEIGHBOURS_PER_HOP` neighbours per hop so a hub
account cannot pull in half the graph. The paper's setting is k=2. Every edge
inside the extracted subgraph becomes a :class:`Transaction`, and the flattened
subgraph is what the LLM eventually sees in the prompt.

The BFS runs over the FULL transaction graph, including edges dated after the
focal edge. That is deliberate for test cases and is the AMLworld protocol: a
reviewer looking at a flagged transaction has the account's whole history
available, and the label being predicted is the focal edge's own. Training and
validation cases are different, and :meth:`AMLworldDataset.create_train_cases`
and :meth:`~AMLworldDataset.create_val_cases` swap in the truncated graph from
:meth:`~AMLworldDataset.get_temporal_graph` so no training case can see a
future edge.

The split
---------
60/20/20 by row order, computed in :mod:`amlc.data.splits` because the
Snap ML feature stage needs the same two indices. Every edge carries its split
in the graph and in ``_edge_split``.

Dropped from the pre-release loader
-----------------------------------
``load_pgts`` and the pre-sampled-case branch it fed. The purged-group
time-series split was built and never used: it produced no number in the paper
and its output directory is empty. Keeping it would mean keeping a second
labelling of every edge, a second set of split lists, and a "purge" case that
half the accessors here had to test for.

``load_synthetic`` and the ``load_dataset`` convenience wrapper. The synthetic
random graph existed to smoke-test the pipeline without the 5 GB CSV; it
generates cases with no relation to AMLworld and nothing in the paper is
computed from it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import paths
from ..case_ids import is_case_id, to_canonical
from ..config import DATASETS, K_HOP, MAX_NEIGHBOURS_PER_HOP, SEEDS
from ..typology import TYPOLOGY_CLASSES
from . import splits
from .download_amlworld import patterns_txt, trans_csv

#: Case counts for the on-the-fly stratified sampler, carried over from the
#: pre-release ``ExperimentConfig``. The paper's evaluation set is the
#: HT-Coreset, not this sampler, so these only matter to a caller that asks for
#: a fresh stratified draw.
N_SAMPLED_ILLICIT = 50
N_SAMPLED_LEGIT = 450
N_TRAIN_ILLICIT = 200
N_TRAIN_LEGIT = 1800

#: The file the serialisation stage writes inside a coreset directory. The
#: directory itself is reached with ``archive.legacy_path(archive,
#: "serialised", dataset)`` when reading from the archived run.
CORESET_CASES_FILENAME = "cases_edge_list.jsonl"

#: Below this many cases the multiprocessing pool costs more than it saves.
MIN_CASES_FOR_POOL = 500

# Module-level reference for the multiprocessing workers. fork lets a worker
# read the parent's loaded graph without pickling it, which matters: the graph
# is several GB.
_ds_ref: Optional["AMLworldDataset"] = None


def _build_case_worker(args: Tuple[str, str, int]) -> Optional["Case"]:
    """Picklable worker for parallel case creation, using fork-inherited state."""
    eid, case_id, label = args
    return _ds_ref._build_case(eid, case_id, label)


@dataclass
class Transaction:
    """A single transaction edge."""
    edge_id: str
    source: str
    target: str
    amount: float
    timestamp: str
    currency: str
    is_illicit: bool
    typology: Optional[str] = None
    payment_format: Optional[str] = None


@dataclass
class Case:
    """A focal edge and its context graph, with the label being predicted."""
    case_id: str
    center_edge_id: str
    transactions: List[Transaction]
    label: int  # 0 = legitimate, 1 = illicit
    typology: Optional[str] = None
    illicit_edges: Set[str] = None
    # Set when the case is read back from a serialised coreset, where the
    # prompt text is already built and the transaction list is not needed.
    _serialized_text: Optional[str] = None

    def __post_init__(self):
        if self.illicit_edges is None:
            self.illicit_edges = {
                t.edge_id for t in self.transactions if t.is_illicit
            }


#: ``Patterns.txt`` writes typology names in upper case. These are the names
#: the rest of the package uses.
_PATTERN_NAME_MAP = {
    "FAN-OUT": "fan-out",
    "FAN-IN": "fan-in",
    "SCATTER-GATHER": "scatter-gather",
    "GATHER-SCATTER": "gather-scatter",
    "CYCLE": "cycle",
    "RANDOM": "random",
    "BIPARTITE": "bipartite",
    "STACK": "stack",
}

# The integer encoding in amlc.typology is keyed on these exact strings,
# so a rename on either side has to fail at import rather than quietly turn
# every typed edge into an unknown class.
assert set(_PATTERN_NAME_MAP.values()) == set(TYPOLOGY_CLASSES)


class AMLworldDataset:
    """AMLworld loaded into a MultiDiGraph, split 60/20/20 by row order.

    Two case builders sit on top of it. :meth:`create_cases` samples the test
    portion for evaluation; :meth:`create_train_cases` and
    :meth:`create_val_cases` sample the earlier portions for the supervised
    baselines, extracting context from a graph truncated at the split boundary.
    """

    def __init__(self, variant: str = DATASETS[0]):
        if variant not in DATASETS:
            raise ValueError(
                f"variant must be one of {DATASETS}, got {variant!r}. The "
                "Medium variants of AMLworld appear in no table in the paper."
            )
        self.variant = variant
        self.graph = nx.MultiDiGraph()

        # Edge bookkeeping, populated by load().
        self._edge_source: Dict[str, str] = {}    # eid -> source node
        self._edge_target: Dict[str, str] = {}    # eid -> target node
        self._illicit_eids: List[str] = []
        self._legit_eids: List[str] = []
        self._edge_typology: Dict[str, str] = {}  # eid -> typology

        # Temporal split tracking.
        self._edge_split: Dict[str, str] = {}     # eid -> train/val/test
        self._train_illicit_eids: List[str] = []
        self._train_legit_eids: List[str] = []
        self._val_illicit_eids: List[str] = []
        self._val_legit_eids: List[str] = []
        self._test_illicit_eids: List[str] = []
        self._test_legit_eids: List[str] = []

        # Split boundaries, as row indices.
        self._t1_idx: int = 0  # first validation row
        self._t2_idx: int = 0  # first test row

        # Case lists. val_cases is initialised here; the pre-release class
        # created the attribute inside create_val_cases only, so reading it
        # before that call raised AttributeError.
        self.cases: List[Case] = []
        self.train_cases: List[Case] = []
        self.val_cases: List[Case] = []
        self._sampled_test_eids: Set[str] = set()

        self._loaded = False

    # ─────────────────────────────────────────────────────────
    #  Loading
    # ─────────────────────────────────────────────────────────

    def load(self, data_path: Optional[Path] = None) -> "AMLworldDataset":
        """Load the transaction CSV and the pattern labels, and split them.

        The CSV arrives sorted by timestamp, which is what makes the split a
        pair of row indices. Edge ids are ``e_<row>``, so an edge id is also
        its position in the file, and every stage downstream indexes on that.
        """
        data_path = Path(data_path) if data_path is not None else paths.data()

        trans_file = trans_csv(self.variant, data_path)
        patterns_file = patterns_txt(self.variant, data_path)

        if not trans_file.exists():
            raise FileNotFoundError(
                f"Not found: {trans_file}. AMLworld is not redistributed with "
                "this repository; run python -m amlc.data.download_amlworld."
            )

        # ── 1. Parse Patterns.txt ──
        typology_map: Dict[str, str] = {}
        if patterns_file.exists():
            typology_map = self._parse_patterns(patterns_file)
            n_types = len(set(typology_map.values()))
            print(f"  Patterns: {len(typology_map):,} labeled edges, "
                  f"{n_types} typology types")

        # ── 2. Load CSV ──
        print(f"  Loading {trans_file.name} ...")
        df = pd.read_csv(
            trans_file,
            header=0,
            names=[
                "Timestamp", "From_Bank", "From_Account",
                "To_Bank", "To_Account",
                "Amount_Received", "Receiving_Currency",
                "Amount_Paid", "Payment_Currency",
                "Payment_Format", "Is_Laundering",
            ],
            dtype={
                "From_Bank": str, "From_Account": str,
                "To_Bank": str, "To_Account": str,
                "Is_Laundering": int,
            },
            low_memory=False,
        )

        n_total = len(df)
        n_illicit = int(df["Is_Laundering"].sum())
        print(f"  {n_total:,} txns  ({n_illicit:,} illicit, "
              f"{n_illicit / n_total * 100:.2f}%)")

        # ── 3. Temporal split boundaries ──
        self._n_edges = n_total
        self._t1_idx, self._t2_idx = splits.temporal_boundaries(n_total)
        print("  Temporal split: "
              + splits.describe(n_total, self._t1_idx, self._t2_idx))

        # ── 4. Node ids ──
        df["source"] = (df["From_Bank"].str.strip() + "_" +
                        df["From_Account"].str.strip())
        df["target"] = (df["To_Bank"].str.strip() + "_" +
                        df["To_Account"].str.strip())

        # ── 5. Match typology ──
        df["match_key"] = (
            df["Timestamp"].str.strip() + "," +
            df["From_Bank"].str.strip() + "," +
            df["From_Account"].str.strip() + "," +
            df["To_Bank"].str.strip() + "," +
            df["To_Account"].str.strip()
        )
        df["typology"] = df["match_key"].map(typology_map)
        n_matched = int(df["typology"].notna().sum())
        print(f"  Typology matched: {n_matched:,}/{n_illicit:,} illicit edges")

        # ── 6. Numpy arrays for speed ──
        sources = df["source"].values
        targets = df["target"].values
        amounts = df["Amount_Paid"].values.astype(np.float64)
        timestamps = df["Timestamp"].values.astype(str)
        currencies = df["Payment_Currency"].values.astype(str)
        payment_formats = df["Payment_Format"].values.astype(str)
        is_illicit = df["Is_Laundering"].values.astype(bool)
        typologies = df["typology"].values

        # ── 7. Build graph and bookkeeping with the temporal split ──
        print(f"  Building graph ({n_total:,} edges) with temporal split ...")

        edge_tuples = []
        for i in tqdm(range(n_total), desc="  Indexing",
                      unit="txn", miniters=200_000):
            eid = f"e_{i}"
            typ = typologies[i] if pd.notna(typologies[i]) else None
            split = splits.split_of(i, self._t1_idx, self._t2_idx)

            self._edge_source[eid] = sources[i]
            self._edge_target[eid] = targets[i]
            self._edge_split[eid] = split

            if is_illicit[i]:
                self._illicit_eids.append(eid)
                if typ:
                    self._edge_typology[eid] = typ
                if split == "train":
                    self._train_illicit_eids.append(eid)
                elif split == "val":
                    self._val_illicit_eids.append(eid)
                else:
                    self._test_illicit_eids.append(eid)
            else:
                self._legit_eids.append(eid)
                if split == "train":
                    self._train_legit_eids.append(eid)
                elif split == "val":
                    self._val_legit_eids.append(eid)
                else:
                    self._test_legit_eids.append(eid)

            edge_tuples.append((
                sources[i], targets[i], eid,
                {
                    "edge_id": eid,
                    "amount": float(amounts[i]),
                    "timestamp": timestamps[i],
                    "currency": currencies[i],
                    "payment_format": payment_formats[i],
                    "is_illicit": bool(is_illicit[i]),
                    "typology": typ,
                    "split": split,
                },
            ))

        print("  Adding edges to NetworkX ...")
        self.graph.add_edges_from(edge_tuples)

        del df, edge_tuples
        self._loaded = True

        print(f"  {self.graph.number_of_nodes():,} nodes, "
              f"{self.graph.number_of_edges():,} edges")
        print("  Temporal split statistics:")
        print(f"      Train: {len(self._train_illicit_eids):,} illicit, "
              f"{len(self._train_legit_eids):,} legit")
        print(f"      Val:   {len(self._val_illicit_eids):,} illicit, "
              f"{len(self._val_legit_eids):,} legit")
        print(f"      Test:  {len(self._test_illicit_eids):,} illicit, "
              f"{len(self._test_legit_eids):,} legit")
        return self

    @staticmethod
    def _parse_patterns(path: Path) -> Dict[str, str]:
        """Parse ``Patterns.txt`` into ``{match_key: typology}``.

        The file is a sequence of laundering attempts, each opening with a line
        that names its typology and closing with an END line. The transactions
        between them belong to that attempt. The key is the first five CSV
        fields, which is what identifies a transaction in the CSV as well.
        """
        typology_map: Dict[str, str] = {}
        current: Optional[str] = None

        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                    try:
                        raw = line.split(" - ", 1)[1].split(":")[0].strip()
                        current = _PATTERN_NAME_MAP.get(raw, raw.lower())
                    except IndexError:
                        current = "unknown"

                elif line.startswith("END LAUNDERING ATTEMPT"):
                    current = None

                elif current:
                    parts = line.split(",")
                    if len(parts) >= 5:
                        key = ",".join(p.strip() for p in parts[:5])
                        typology_map[key] = current

        return typology_map

    # ─────────────────────────────────────────────────────────
    #  Context graph extraction
    # ─────────────────────────────────────────────────────────

    def extract_k_hop_subgraph(
        self,
        center_node: str,
        k: int = K_HOP,
        max_neighbors_per_hop: int = MAX_NEIGHBOURS_PER_HOP,
    ) -> nx.MultiDiGraph:
        """BFS k-hop extraction around one account, with neighbour sampling.

        Direction is ignored when walking: a counterparty is a neighbour
        whether it received or sent. The per-hop cap keeps a hub account from
        exploding the subgraph, and it binds on hubs only. Sampling is from the
        module-level :mod:`random`, seeded by the calling case builder, so a
        given seed reproduces a given draw.
        """
        visited = {center_node}
        frontier = {center_node}

        for _ in range(k):
            new_frontier: Set[str] = set()
            for node in frontier:
                nbrs = (
                    set(self.graph.successors(node))
                    | set(self.graph.predecessors(node))
                ) - visited
                if max_neighbors_per_hop and len(nbrs) > max_neighbors_per_hop:
                    nbrs = set(random.sample(sorted(nbrs),
                                             max_neighbors_per_hop))
                new_frontier.update(nbrs)
            frontier = new_frontier - visited
            visited.update(frontier)

        return self.graph.subgraph(visited).copy()

    @staticmethod
    def _subgraph_to_transactions(sg: nx.MultiDiGraph) -> List[Transaction]:
        """Convert subgraph edges into a Transaction list."""
        txns: List[Transaction] = []
        for u, v, key, data in sg.edges(keys=True, data=True):
            txns.append(Transaction(
                edge_id=data.get("edge_id", key),
                source=u,
                target=v,
                amount=float(data.get("amount", 0)),
                timestamp=str(data.get("timestamp", "")),
                currency=str(data.get("currency", "")),
                is_illicit=bool(data.get("is_illicit", False)),
                typology=data.get("typology"),
                payment_format=data.get("payment_format"),
            ))
        return txns

    def _build_case(
        self,
        eid: str,
        case_id: str,
        label: int,
        k: Optional[int] = None,
        max_neighbors_per_hop: Optional[int] = None,
    ) -> Optional[Case]:
        """Build one Case from a focal edge id.

        Returns None when the edge is unknown or its subgraph is empty, and
        both callers drop those cases rather than emitting an empty prompt.

        ``k`` and ``max_neighbors_per_hop`` default to the paper's setting in
        :mod:`amlc.config`. They are parameters because the demonstration
        pool is built with a smaller context (k=1, 15 neighbours) to keep the
        in-context examples short; the pre-release generator got that by
        assigning to the global ``EXPERIMENT`` config object mid-run, which a
        module constant cannot support and which left the override in place for
        whatever ran next.
        """
        center_node = self._edge_source.get(eid)
        if center_node is None:
            return None

        sg = self.extract_k_hop_subgraph(
            center_node,
            k=K_HOP if k is None else k,
            max_neighbors_per_hop=(MAX_NEIGHBOURS_PER_HOP
                                   if max_neighbors_per_hop is None
                                   else max_neighbors_per_hop),
        )
        txns = self._subgraph_to_transactions(sg)
        if not txns:
            return None

        typ = self._edge_typology.get(eid) if label == 1 else None
        return Case(
            case_id=case_id,
            center_edge_id=eid,
            transactions=txns,
            label=label,
            typology=typ,
        )

    # ─────────────────────────────────────────────────────────
    #  Typology-balanced sampling
    # ─────────────────────────────────────────────────────────

    def _balanced_typology_sample(
        self,
        edge_ids: List[str],
        n_total: int,
        seed: int,
    ) -> List[str]:
        """Sample illicit edges spread evenly across typologies.

        Groups the pool by typology and distributes ``n_total`` round-robin.
        Typologies with too few edges contribute what they have and the
        shortfall goes to typologies with a surplus; untyped illicit edges fill
        any gap that remains. Without this the rare typologies vanish from a
        small draw, which is the coverage problem the HT-Coreset solves for the
        released evaluation set.
        """
        rng = random.Random(seed)

        typo_pools: Dict[str, List[str]] = defaultdict(list)
        untyped: List[str] = []
        for eid in edge_ids:
            typ = self._edge_typology.get(eid)
            if typ:
                typo_pools[typ].append(eid)
            else:
                untyped.append(eid)

        for pool in typo_pools.values():
            rng.shuffle(pool)
        rng.shuffle(untyped)

        available_typos = sorted(typo_pools.keys())
        n_typos = len(available_typos)

        if n_typos == 0:
            return rng.sample(edge_ids, min(n_total, len(edge_ids)))

        per_typo = n_total // n_typos
        remainder = n_total % n_typos

        target = {}
        for i, typ in enumerate(available_typos):
            target[typ] = per_typo + (1 if i < remainder else 0)

        # First pass: take min(target, available) from each typology.
        sampled = []
        shortfall = 0
        surplus_typos = []
        for typ in available_typos:
            pool = typo_pools[typ]
            take = min(target[typ], len(pool))
            sampled.extend(pool[:take])
            shortfall += target[typ] - take
            if len(pool) > take:
                surplus_typos.append((typ, pool[take:]))

        # Second pass: redistribute the shortfall over typologies with spare.
        if shortfall > 0 and surplus_typos:
            rng.shuffle(surplus_typos)
            for typ, remaining in surplus_typos:
                take = min(shortfall, len(remaining))
                sampled.extend(remaining[:take])
                shortfall -= take
                if shortfall == 0:
                    break

        if shortfall > 0 and untyped:
            take = min(shortfall, len(untyped))
            sampled.extend(untyped[:take])
            shortfall -= take

        dist = defaultdict(int)
        for eid in sampled:
            typ = self._edge_typology.get(eid, "untyped")
            dist[typ] += 1
        dist_str = ", ".join(f"{t}={c}" for t, c in sorted(dist.items()))
        print(f"  Typo-balanced: {len(sampled)} illicit [{dist_str}]")

        return sampled

    # ─────────────────────────────────────────────────────────
    #  Case creation
    # ─────────────────────────────────────────────────────────

    def create_cases(
        self,
        seed: int = SEEDS[0],
        n_illicit: Optional[int] = None,
        n_legit: Optional[int] = None,
        use_full_test: bool = False,
        balance_typology: bool = False,
        coreset_dir: Optional[Path] = None,
    ) -> List[Case]:
        """Build the test cases, from the temporal test portion only.

        Three ways to get them, in the order they are tried.

        ``coreset_dir``
            Read the released HT-Coreset back from its serialised form. This is
            what the LLM evaluation used: the prompt text was built once, so
            every model sees byte-identical input and no graph has to be
            loaded. Case identifiers are rewritten to the canonical ``amlc_``
            form on the way in; see :mod:`amlc.case_ids`.

        ``use_full_test``
            Every edge in the test portion. Over a million cases, so this is
            for the supervised baselines, not for an LLM.

        otherwise
            A stratified draw of ``n_illicit`` illicit and ``n_legit`` benign
            edges. This is the pre-coreset evaluation set and it is kept
            because the Naive Coreset comparison needs a draw to compare
            against.

        Context extraction uses the full graph, which for test cases is the
        AMLworld protocol: the label being predicted is the focal edge's own.
        """
        random.seed(seed)

        if coreset_dir:
            jsonl_file = Path(coreset_dir) / CORESET_CASES_FILENAME
            if jsonl_file.exists():
                self.cases = []
                n_retagged = 0
                with open(jsonl_file) as f:
                    for line in tqdm(f, desc="  Loading coreset cases",
                                     leave=False):
                        row = json.loads(line)
                        cid = row["case_id"]
                        if is_case_id(cid):
                            canonical = to_canonical(cid)
                            n_retagged += canonical != cid
                            cid = canonical
                        c = Case(
                            case_id=cid,
                            center_edge_id=row["center_edge_id"],
                            transactions=[],  # not needed, text is pre-built
                            label=int(row["label"]),
                            typology=row.get("typology") or None,
                            illicit_edges=set(),
                            _serialized_text=row["serialized_text"],
                        )
                        self.cases.append(c)

                random.shuffle(self.cases)

                n_pos = sum(1 for c in self.cases if c.label == 1)
                n_neg = len(self.cases) - n_pos
                print(f"  HT-Coreset: {n_pos} illicit + {n_neg} legit "
                      f"= {len(self.cases)} cases [pre-serialised]")
                if n_retagged:
                    print(f"  Retagged {n_retagged} pre-release case ids")
                return self.cases

        avail_ill = self._test_illicit_eids
        avail_leg = self._test_legit_eids

        if use_full_test:
            sampled_ill = avail_ill
            sampled_leg = avail_leg
            print(f"  Full test mode: {len(sampled_ill)} illicit + "
                  f"{len(sampled_leg)} legit (all test edges)")
        else:
            n_ill = min(n_illicit or N_SAMPLED_ILLICIT, len(avail_ill))
            n_leg = min(n_legit or N_SAMPLED_LEGIT, len(avail_leg))

            if n_ill < (n_illicit or N_SAMPLED_ILLICIT):
                print(f"  Warning: Only {len(avail_ill)} illicit edges in test "
                      f"portion, sampling {n_ill}")

            if balance_typology:
                sampled_ill = self._balanced_typology_sample(
                    avail_ill, n_ill, seed
                )
            else:
                sampled_ill = random.sample(avail_ill, n_ill)
            sampled_leg = random.sample(avail_leg, n_leg)

        self._sampled_test_eids = set(sampled_ill + sampled_leg)

        self.cases = []

        for i, eid in enumerate(
            tqdm(sampled_ill, desc="  Test-illicit", leave=False)
        ):
            c = self._build_case(eid, f"case_illicit_{i}", label=1)
            if c:
                self.cases.append(c)

        for i, eid in enumerate(
            tqdm(sampled_leg, desc="  Test-legit", leave=False)
        ):
            c = self._build_case(eid, f"case_legit_{i}", label=0)
            if c:
                self.cases.append(c)

        random.shuffle(self.cases)

        n_pos = sum(1 for c in self.cases if c.label == 1)
        n_neg = len(self.cases) - n_pos
        pct = n_pos / len(self.cases) * 100 if self.cases else 0
        mode_str = ("full temporal test portion" if use_full_test
                    else "stratified from temporal test portion")
        print(f"  Test: {len(self.cases)} cases  "
              f"({n_pos} illicit {pct:.0f}%, {n_neg} legit) "
              f"[{mode_str}]")
        return self.cases

    def create_train_cases(
        self,
        seed: int = SEEDS[0],
        n_illicit: Optional[int] = None,
        n_legit: Optional[int] = None,
        use_train_graph: bool = True,
        use_full_train: bool = False,
        n_workers: int = 1,
    ) -> List[Case]:
        """Build the training cases, from the temporal train portion only.

        ``use_train_graph`` swaps the full graph for the one truncated at t1
        while the cases are built, so a training case cannot draw context from
        an edge that had not happened yet. Leave it on.

        The seed is offset from the evaluation seed so a train draw and a test
        draw at the same nominal seed do not share the sampler's state.
        """
        random.seed(seed + 99_999)

        avail_ill = self._train_illicit_eids
        avail_leg = self._train_legit_eids

        if use_full_train:
            # The AMLworld protocol: train on the whole split, at its natural
            # imbalance, rather than on a balanced sample of it.
            sampled_ill = avail_ill
            sampled_leg = avail_leg
            ratio = len(sampled_leg) / max(len(sampled_ill), 1)
            print(f"  Full train mode: {len(sampled_ill)} illicit + "
                  f"{len(sampled_leg)} legit (all edges, ratio 1:{ratio:.0f})")
        else:
            n_ill = min(n_illicit or N_TRAIN_ILLICIT, len(avail_ill))
            n_leg = min(n_legit or N_TRAIN_LEGIT, len(avail_leg))

            if n_ill < (n_illicit or N_TRAIN_ILLICIT):
                print(f"  Warning: Only {len(avail_ill)} illicit edges in train "
                      f"portion, sampling {n_ill}")

            sampled_ill = random.sample(avail_ill, n_ill)
            sampled_leg = random.sample(avail_leg, n_leg)

        original_graph = None
        if use_train_graph:
            original_graph = self.graph
            self.graph = self.get_temporal_graph("train")

        all_args = (
            [(eid, f"train_illicit_{i}", 1) for i, eid in enumerate(sampled_ill)]
            + [(eid, f"train_legit_{i}", 0) for i, eid in enumerate(sampled_leg)]
        )

        if n_workers > 1 and len(all_args) > MIN_CASES_FOR_POOL:
            global _ds_ref
            _ds_ref = self
            print(f"  Building {len(all_args):,} cases with {n_workers} workers ...")
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_build_case_worker, all_args, chunksize=512),
                    total=len(all_args), desc="  Train-cases", leave=False,
                ))
            _ds_ref = None
            self.train_cases = [c for c in results if c is not None]
        else:
            self.train_cases = []
            for eid, case_id, label in tqdm(all_args, desc="  Train-cases",
                                            leave=False):
                c = self._build_case(eid, case_id, label)
                if c:
                    self.train_cases.append(c)

        if original_graph is not None:
            self.graph = original_graph

        random.shuffle(self.train_cases)

        n_pos = sum(1 for c in self.train_cases if c.label == 1)
        mode_str = "full temporal" if use_full_train else "sampled from temporal"
        print(f"  Train: {len(self.train_cases)} cases  "
              f"({n_pos} illicit, {len(self.train_cases) - n_pos} legit) "
              f"[{mode_str} train portion]")
        return self.train_cases

    def create_val_cases(
        self,
        seed: int = SEEDS[0],
        n_illicit: Optional[int] = None,
        n_legit: Optional[int] = None,
        use_val_graph: bool = True,
        use_full_val: bool = False,
        n_workers: int = 1,
    ) -> List[Case]:
        """Build the validation cases, from the temporal val portion only.

        ``use_val_graph`` truncates the graph at t2, so validation context may
        include training edges but never a test edge.
        """
        random.seed(seed + 88_888)

        avail_ill = self._val_illicit_eids
        avail_leg = self._val_legit_eids

        if use_full_val:
            sampled_ill = avail_ill
            sampled_leg = avail_leg
            print(f"  Full val mode: {len(sampled_ill)} illicit + "
                  f"{len(sampled_leg)} legit (all edges)")
        else:
            n_ill = min(n_illicit or N_SAMPLED_ILLICIT, len(avail_ill))
            n_leg = min(n_legit or N_SAMPLED_LEGIT, len(avail_leg))
            sampled_ill = random.sample(avail_ill, n_ill)
            sampled_leg = random.sample(avail_leg, n_leg)

        original_graph = None
        if use_val_graph:
            original_graph = self.graph
            self.graph = self.get_temporal_graph("val")

        all_args = (
            [(eid, f"val_illicit_{i}", 1) for i, eid in enumerate(sampled_ill)]
            + [(eid, f"val_legit_{i}", 0) for i, eid in enumerate(sampled_leg)]
        )

        if n_workers > 1 and len(all_args) > MIN_CASES_FOR_POOL:
            global _ds_ref
            _ds_ref = self
            print(f"  Building {len(all_args):,} val cases with {n_workers} "
                  "workers ...")
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_build_case_worker, all_args, chunksize=512),
                    total=len(all_args), desc="  Val-cases", leave=False,
                ))
            _ds_ref = None
            self.val_cases = [c for c in results if c is not None]
        else:
            self.val_cases = []
            for eid, case_id, label in tqdm(all_args, desc="  Val-cases",
                                            leave=False):
                c = self._build_case(eid, case_id, label)
                if c:
                    self.val_cases.append(c)

        if original_graph is not None:
            self.graph = original_graph

        random.shuffle(self.val_cases)

        n_pos = sum(1 for c in self.val_cases if c.label == 1)
        mode_str = "full temporal" if use_full_val else "sampled from temporal"
        print(f"  Val: {len(self.val_cases)} cases  "
              f"({n_pos} illicit, {len(self.val_cases) - n_pos} legit) "
              f"[{mode_str} val portion]")
        return self.val_cases

    # ─────────────────────────────────────────────────────────
    #  Temporal graphs and statistics
    # ─────────────────────────────────────────────────────────

    def get_temporal_graph(self, split: str) -> nx.MultiDiGraph:
        """The graph containing only edges up to a split boundary.

        ``train`` keeps edges before t1, ``val`` keeps everything before t2,
        and ``test`` or ``full`` returns the loaded graph itself, because a
        test case is allowed the account's whole history.

        The returned train and val graphs are fresh objects, so a caller can
        swap them in and drop them; the test graph is the loaded one and must
        not be mutated.
        """
        if split in ("test", "full"):
            return self.graph

        if split == "train":
            valid_splits = {"train"}
        elif split == "val":
            valid_splits = {"train", "val"}
        else:
            raise ValueError(f"Unknown split: {split}")

        edges_to_keep = [
            (u, v, k, d) for u, v, k, d in self.graph.edges(keys=True, data=True)
            if d.get("split") in valid_splits
        ]

        subgraph = nx.MultiDiGraph()
        subgraph.add_edges_from(edges_to_keep)

        return subgraph

    def get_split_statistics(self) -> Dict:
        """Edge counts per split, and the two boundary row indices.

        The pre-release version also reported a "purge" count, which only the
        purged-group split ever produced. That loader is gone, so every edge
        now lands in exactly one of the three splits and the counts add up to
        the file.
        """
        return {
            "train": {
                "illicit": len(self._train_illicit_eids),
                "legit": len(self._train_legit_eids),
                "total": len(self._train_illicit_eids) + len(self._train_legit_eids),
            },
            "val": {
                "illicit": len(self._val_illicit_eids),
                "legit": len(self._val_legit_eids),
                "total": len(self._val_illicit_eids) + len(self._val_legit_eids),
            },
            "test": {
                "illicit": len(self._test_illicit_eids),
                "legit": len(self._test_legit_eids),
                "total": len(self._test_illicit_eids) + len(self._test_legit_eids),
            },
            "boundaries": {
                "t1_idx": self._t1_idx,
                "t2_idx": self._t2_idx,
            },
        }


# ─────────────────────────────────────────────────────────────────────────
# Graph source for the serialisation stages
# ─────────────────────────────────────────────────────────────────────────

class _GraphSource:
    """Adapter from :class:`AMLworldDataset` to the interface the serialisation
    stages ask for.

    Those stages take the graph as an injected ``module:factory`` so that a
    clone which only consumes the released prompts never has to load AMLworld.
    Until now no factory shipped, so ``--loader`` was a required argument with
    nothing valid to pass it, and ``make serialize`` and ``make icl`` could not
    run at all. This is that factory's other half.
    """

    def __init__(self, dataset: "AMLworldDataset", k_hop: int, max_neighbours: int):
        self._dataset = dataset
        self._k_hop = k_hop
        self._max_neighbours = max_neighbours

    def test_split_start(self) -> int:
        """Row index of the first edge in the temporal test split."""
        return self._dataset._t2_idx

    def n_edges(self) -> int:
        return self._dataset._n_edges

    def build_case(self, edge_id: str, case_id: str, label: int):
        return self._dataset._build_case(
            edge_id, case_id, label, self._k_hop, self._max_neighbours,
        )


def graph_source(dataset: str, *, k_hop: int = K_HOP,
                 max_neighbours: int = MAX_NEIGHBOURS_PER_HOP) -> _GraphSource:
    """Open an AMLworld variant and return a graph source for serialisation.

    This is the default for ``--loader`` on the serialisation stages, spelled
    ``amlc.data.loader:graph_source``. Loading the variant reads the raw
    transaction CSVs, so it needs AMLworld on disk; see
    :mod:`amlc.data.download_amlworld`.
    """
    return _GraphSource(AMLworldDataset(dataset).load(), k_hop, max_neighbours)
