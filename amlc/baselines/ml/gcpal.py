"""GCPAL: a GIN over the line graph of the transaction graph, third ensemble member.

A reimplementation of Lu and Wang (2024), "Graph Contrastive Pre-training for
Anti-money Laundering", Int. J. Comput. Intell. Syst. 17:307. No official code
exists, so this follows the paper's description: each transaction becomes a node
in a line graph, two line-graph nodes are joined when their transactions share an
account and sit within ``lg_k`` positions of each other in time, and the task is
node classification with a two-layer GIN encoder and an ``MLP(H || X)``
classifier, after two-view or three-view contrastive pre-training.

Node features are the 79 Snap ML GFP dimensions plus 5 dimensions aggregated
from the line-graph edges, 84 in total. The aggregation is this repository's
addition and the largest single gain over the plain GIN. It exists because
per-edge message passing over 87 million line-graph edges does not fit in 80 GB:
scattering the edge attributes onto their destination nodes costs 0.1 GB where
GINEConv costs over 20 GB.

Reads
    ``<data>/<dataset>_Trans.csv`` and ``<dataset>_Patterns.txt`` through
    :func:`amlc.baselines.ml.gbt.build_pyg_link_data`, and the Snap ML
    feature matrix.
Writes
    the pre-trained encoder and one fine-tuned checkpoint per seed under
    ``<archive>/models/<dataset>/<run>/``, per-seed and ensemble probabilities
    under ``<archive>/test_probs/<dataset>/<run>/``, and per-seed metrics plus a
    summary row under ``results/ml/``.

The split, stated plainly
------------------------
``--random-split`` is kept because it is what produced the released checkpoints.
Fine-tuning used a **random** 60/20/20 split of the edges, not the temporal one
the other two ensemble members use. The consequence is direct: 609,857 of the
1,015,669 HI-Small temporal test edges and 830,310 of the 1,384,810 LI-Small
temporal test edges were inside GCPAL's fine-tuning set. The GCPAL+GFP row, and
therefore every ensemble figure that includes it, is optimistic to that extent.
The paper's own comparison against Lu and Wang requires the random split, since
that is the protocol they report, but nothing recovers the leakage into the
temporal test set. :mod:`amlc.baselines.ml.gcpal_infer` re-infers these
checkpoints on the temporal split, which fixes the alignment of the probability
arrays and does not fix this.

Dropped configurations
----------------------
Everything below was explored and superseded, and none of it is behind a number
in the paper.

``GINEEncoder`` and ``EdgeWeightedGINConv``
    Both out-of-memory at ``lg_k`` >= 3, and the model was never constructed
    with ``edge_dim > 0``, so both classes were unreachable. The pre-aggregation
    above is what replaced them.
``--gine``
    The flag that switched pre-aggregation on. It is now unconditional: the
    released checkpoints used it, the alternative is the superseded plain-GIN
    configuration, and :mod:`gcpal_infer` has to rebuild the identical 84
    dimensions or the checkpoints will not load.
``--no-pretrain``
    The v5 configuration. The released checkpoints were pre-trained.
``--classifier lgbm``
    A hybrid that fed the GIN embeddings to LightGBM instead of the paper's
    ``MLP(H || X)`` head.
``--focal-loss``, ``--focal-gamma``, ``FocalLoss``
    Better AUCPR, worse F1; not used.
``--minibatch`` and ``finetune_gcpal_minibatch``
    Balanced-loss sampling over 34k of 5M nodes. F1 0.464 against 0.596; it
    overfitted badly.
``--num-neighbors``, ``--train-batch-size``
    Unused.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ... import paths
from ...config import DATASETS, SEEDS
from ...archive import legacy_path
from ...typology import TYPOLOGY_CLASSES, TYPOLOGY_TO_IDX
from ..metrics import find_optimal_threshold
from .gbt import build_pyg_link_data, load_snapml_features

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ═════════════════════════════════════════════════════════════
#  Output locations inside the archived run directory
# ═════════════════════════════════════════════════════════════

def checkpoint_dir(archive: Path, dataset: str, run: str) -> Path:
    """Where one training run's GCPAL checkpoints live in the archived tree.

    ``archive.legacy_path`` has no kind for this directory, because the
    probability directory it pairs with was renamed for the paper and this one
    was not: ``run`` is composed from the model's own name, which the paper also
    uses. The pairing is the subject of :mod:`gcpal_infer`.
    """
    return Path(archive) / "models" / dataset / run


def probability_dir(archive: Path, dataset: str, run: str) -> Path:
    """Where one training run's per-seed test probabilities live."""
    return legacy_path(archive, "member_probs", dataset, member=run, seed=0).parent


# ═════════════════════════════════════════════════════════════
#  Line graph construction
# ═════════════════════════════════════════════════════════════

def build_line_graph(edge_index, k_neighbors: int = 1,
                     return_edge_attr: bool = False):
    """Turn the transaction graph into its line graph.

    Each transaction becomes a node. Two line-graph nodes are joined when their
    transactions share an account, as sender or receiver, and are within
    ``k_neighbors`` positions of each other in the account's own time-ordered
    list of transactions. ``k_neighbors=1`` joins only consecutive pairs;
    ``k_neighbors=5``, the released setting, joins each transaction to its five
    nearest per shared account.

    Parameters
    ----------
    edge_index : Tensor
        (2, n_edges) transaction graph, in temporal order.
    return_edge_attr : bool
        Also return the 3-dimensional edge features
        ``[log(1 + delta), is_sender_i, is_sender_j]``, which
        :func:`run_gcpal` scatters onto the nodes.

    Returns ``lg_edge_index``, or ``(lg_edge_index, lg_edge_attr)``. The line
    graph is undirected: every pair appears in both directions, and the sender
    flags are swapped in the reversed copy.
    """
    import torch

    n_edges = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]

    # Each transaction appears twice, once per endpoint.
    accounts = torch.cat([src, dst])
    edge_ids = torch.arange(n_edges, device=src.device).repeat(2)

    # 1 where this entry is the sending side, 0 where it is the receiving side.
    is_sender = torch.cat([torch.ones(n_edges, device=src.device),
                           torch.zeros(n_edges, device=src.device)])

    # Sort by account, then by edge id, which is temporal order.
    sort_key = accounts.long() * (n_edges + 1) + edge_ids.long()
    sort_idx = sort_key.argsort()
    sorted_accounts = accounts[sort_idx]
    sorted_edge_ids = edge_ids[sort_idx]
    sorted_is_sender = is_sender[sort_idx]

    lg_src_list = []
    lg_dst_list = []
    lg_delta_list = []
    lg_src_sender_list = []
    lg_dst_sender_list = []

    for delta in range(1, k_neighbors + 1):
        same_account = sorted_accounts[:-delta] == sorted_accounts[delta:]
        diff_edge = sorted_edge_ids[:-delta] != sorted_edge_ids[delta:]
        mask = same_account & diff_edge

        lg_src_list.append(sorted_edge_ids[:-delta][mask])
        lg_dst_list.append(sorted_edge_ids[delta:][mask])

        if return_edge_attr:
            lg_delta_list.append(torch.full(
                (mask.sum().item(),), delta, dtype=torch.float,
                device=src.device))
            lg_src_sender_list.append(sorted_is_sender[:-delta][mask])
            lg_dst_sender_list.append(sorted_is_sender[delta:][mask])

    lg_src = torch.cat(lg_src_list)
    lg_dst = torch.cat(lg_dst_list)

    lg_edge_index = torch.stack([
        torch.cat([lg_src, lg_dst]),
        torch.cat([lg_dst, lg_src]),
    ])

    if return_edge_attr:
        deltas = torch.cat(lg_delta_list)
        src_is_sender = torch.cat(lg_src_sender_list)
        dst_is_sender = torch.cat(lg_dst_sender_list)
        lg_edge_attr = torch.stack([
            torch.log1p(torch.cat([deltas, deltas])),
            torch.cat([src_is_sender, dst_is_sender]),
            torch.cat([dst_is_sender, src_is_sender]),
        ], dim=-1)
        return lg_edge_index, lg_edge_attr

    return lg_edge_index


def aggregate_edge_features(lg_edge_index, lg_edge_attr, n_total: int):
    """Scatter the line-graph edge attributes onto their destination nodes.

    Five dimensions per node: the mean of each of the three edge features, the
    log degree in the line graph, and the smallest temporal delta on any
    incoming edge. This is what makes the edge information usable, since
    per-edge message passing over 87 million edges does not fit in memory.
    """
    import torch

    dst = lg_edge_index[1]
    n_lg_edges = lg_edge_index.shape[1]

    feat_sum = torch.zeros(n_total, 3, dtype=torch.float)
    feat_sum.index_add_(0, dst, lg_edge_attr.float())
    edge_count = torch.zeros(n_total, dtype=torch.float)
    edge_count.index_add_(0, dst, torch.ones(n_lg_edges))
    edge_mean = feat_sum / edge_count.unsqueeze(1).clamp(min=1)

    log_degree = torch.log1p(edge_count).unsqueeze(1)

    min_delta = torch.full((n_total,), float("inf"))
    min_delta.scatter_reduce_(0, dst, lg_edge_attr[:, 0].float(),
                              reduce="amin", include_self=False)
    min_delta[min_delta.isinf()] = 0.0
    min_delta = min_delta.unsqueeze(1)

    return torch.cat([edge_mean, log_degree, min_delta], dim=1)


# ═════════════════════════════════════════════════════════════
#  The model
# ═════════════════════════════════════════════════════════════

_CLASSES: dict = {}


def gcpal_classes() -> dict:
    """Define the torch modules on first call and cache them.

    torch and torch-geometric are optional extras, so ``nn.Module`` subclasses
    cannot be declared at import time without making every import of this
    package depend on them. They are built once here instead. Returns a dict
    with ``GINEncoder`` and ``GCPAL``.
    """
    if _CLASSES:
        return _CLASSES

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import BatchNorm, GINConv

    class GINEncoder(nn.Module):
        """Graph Isomorphism Network encoder (Xu et al., 2019). Paper Sec. 4.2.2."""

        def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                     dropout: float = 0.1):
            super().__init__()
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(num_layers):
                in_ch = in_dim if i == 0 else hidden_dim
                mlp = nn.Sequential(
                    nn.Linear(in_ch, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
                self.norms.append(BatchNorm(hidden_dim))
            self.dropout = dropout

        def forward(self, x, edge_index):
            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return x

    class GCPAL(nn.Module):
        """GIN encoder, contrastive projector, and an ``MLP(H || X)`` classifier.

        Paper Eq. 7: the classifier sees the GIN embedding concatenated with the
        node's own features, so the boosted-tree-style GFP signal reaches the
        output head directly and not only through message passing.
        """

        def __init__(self, feat_dim: int, hidden_dim: int = 128,
                     num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.encoder = GINEncoder(feat_dim, hidden_dim, num_layers, dropout)

            # Projection head for the contrastive objective, Sec. 4.2.3.
            self.projector = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

        def encode(self, x, edge_index):
            return self.encoder(x, edge_index)

        def project(self, h):
            return self.projector(h)

        def classify(self, h, x):
            return self.classifier(torch.cat([h, x], dim=-1))

        def forward(self, x, edge_index):
            return self.classify(self.encode(x, edge_index), x)

    _CLASSES.update(GINEncoder=GINEncoder, GCPAL=GCPAL)
    return _CLASSES


def build_gcpal(feat_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                dropout: float = 0.1):
    """Construct a GCPAL model. The only way to get one; see :func:`gcpal_classes`."""
    return gcpal_classes()["GCPAL"](
        feat_dim=feat_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout,
    )


# ═════════════════════════════════════════════════════════════
#  Augmentation and the contrastive objective  (Paper Sec. 4.2.1, 4.2.3)
# ═════════════════════════════════════════════════════════════

def edge_dropping(edge_index, drop_ratio: float = 0.3):
    """ED(G): drop each edge independently. Paper Eq. 2."""
    import torch

    keep = torch.rand(edge_index.shape[1], device=edge_index.device) > drop_ratio
    return edge_index[:, keep]


def feature_dropping(x, drop_ratio: float = 0.1):
    """FD(G): mask whole feature dimensions. Paper Eq. 3."""
    import torch

    mask = torch.rand(x.shape[1], device=x.device) > drop_ratio
    return x * mask.unsqueeze(0)


def info_nce_loss(z1, z2, tau: float = 0.5):
    """InfoNCE between two views, positives on the diagonal."""
    import torch
    import torch.nn.functional as F

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    sim = torch.mm(z1, z2.t()) / tau
    labels = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(sim, labels)


def build_knn_graph(node_features, k: int = 5, device=None,
                    chunk_size: int = 1024):
    """Cosine-similarity k-nearest-neighbour graph, the paper's third view.

    Chunked so the similarity matrix never materialises in full. Returns an
    undirected edge index on CPU.
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        device = node_features.device

    n = node_features.shape[0]
    print(f"    Building KNN graph: {n:,} nodes, k={k}, chunk={chunk_size} ...")
    t0 = time.time()

    x_norm = F.normalize(node_features.float(), dim=-1)
    knn_indices = torch.zeros(n, k, dtype=torch.long)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        cs = end - start

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            sim = torch.mm(x_norm[start:end], x_norm.t())

        # Exclude self-similarity before the top-k.
        self_idx = torch.arange(cs, device=device)
        global_idx = torch.arange(start, end, device=device)
        sim[self_idx, global_idx] = -2.0

        _, topk = sim.topk(k, dim=-1)
        knn_indices[start:end] = topk.cpu()
        del sim, topk

        if (start // chunk_size) % 500 == 0 and start > 0:
            elapsed = time.time() - t0
            print(f"      KNN progress: {end / n * 100:.0f}% [{elapsed:.0f}s]")

    knn_src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
    knn_dst = knn_indices.reshape(-1)
    edge_index = torch.stack([
        torch.cat([knn_src, knn_dst]),
        torch.cat([knn_dst, knn_src]),
    ])

    print(f"    KNN graph: {edge_index.shape[1]:,} edges [{time.time() - t0:.1f}s]")
    return edge_index


# ═════════════════════════════════════════════════════════════
#  Phase 1: contrastive pre-training  (Paper Eq. 4-6)
# ═════════════════════════════════════════════════════════════

def pretrain_gcpal(
    model,
    data,
    device,
    knn_edge_index=None,
    epochs: int = 200,
    edge_drop_ratio: float = 0.3,
    feat_drop_ratio: float = 0.1,
    tau: float = 0.5,
    lam: float = 0.5,
    lr: float = 0.001,
    batch_size: int = 4096,
):
    """Self-supervised pre-training of the encoder and projector.

    Three views when ``knn_edge_index`` is given, which is the paper's full
    method and what the released checkpoints used::

        L = lam * L_NCE(G'1, G'2) + (1 - lam) * L_NCE(G'2, G_KNN)

    Two views otherwise, dropping the second term.
    """
    import torch

    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.projector.parameters()),
        lr=lr,
    )

    n_nodes = data.num_nodes
    use_knn = knn_edge_index is not None
    views_str = "3-view (aug+KNN)" if use_knn else "2-view (aug)"
    model.train()
    print(f"    Pre-training: {epochs} epochs, {views_str}, "
          f"tau={tau}, lam={lam}, batch={batch_size}")

    for epoch in range(epochs):
        optimizer.zero_grad()

        ei1 = edge_dropping(data.edge_index, edge_drop_ratio)
        ei2 = edge_dropping(data.edge_index, edge_drop_ratio)
        x1 = feature_dropping(data.x, feat_drop_ratio)
        x2 = feature_dropping(data.x, feat_drop_ratio)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h1 = model.encode(x1, ei1)
            h2 = model.encode(x2, ei2)

            idx = torch.randperm(n_nodes, device=device)[:batch_size]
            z1 = model.project(h1[idx])
            z2 = model.project(h2[idx])
            loss_aug = info_nce_loss(z1, z2, tau)

            if use_knn:
                h_knn = model.encode(data.x, knn_edge_index)
                z_knn = model.project(h_knn[idx])
                loss_knn = info_nce_loss(z2, z_knn, tau)
                loss = lam * loss_aug + (1 - lam) * loss_knn
            else:
                loss = loss_aug

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            extra = f" knn={loss_knn.item():.4f}" if use_knn else ""
            print(f"      [epoch {epoch+1}/{epochs}] loss={loss.item():.4f}{extra}")

    print("    Pre-training complete")
    return model


# ═════════════════════════════════════════════════════════════
#  Phase 2: supervised fine-tuning  (node classification)
# ═════════════════════════════════════════════════════════════

def finetune_gcpal(
    model,
    data,
    train_mask,
    val_mask,
    device,
    epochs: int = 500,
    lr: float = 0.001,
    w_illicit: float = 1.0,
    eval_every: int = 5,
    patience: int = 50,
) -> Tuple[dict, float, float]:
    """Full-graph fine-tuning with weighted cross-entropy and early stopping.

    Every ``eval_every`` epochs the validation split is scored with a full
    threshold sweep, and the best state and threshold are kept. Returns
    ``(state_dict on CPU, best validation F1, best threshold)``.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight = torch.tensor([1.0, w_illicit], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_f1 = 0.0
    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    no_improve = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(data.x, data.edge_index)
            logits = model.classify(h, data.x)
            loss = criterion(logits[train_mask], data.y[train_mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = model.encode(data.x, data.edge_index)
                logits_all = model.classify(h, data.x)
                val_probs = F.softmax(logits_all[val_mask], dim=-1)[:, 1] \
                             .cpu().float().numpy()
                val_labels = data.y[val_mask].cpu().numpy()

            opt_t, val_f1 = find_optimal_threshold(val_labels, val_probs)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_threshold = opt_t
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += eval_every

            if (epoch + 1) % 50 == 0:
                print(f"      [ep {epoch+1}] val_f1={val_f1:.4f} "
                      f"best={best_val_f1:.4f}@{best_epoch} "
                      f"loss={loss.item():.4f}")

            model.train()

            if no_improve >= patience:
                print(f"    Early stop at ep {epoch+1}, "
                      f"best_val_f1={best_val_f1:.4f}@{best_epoch}")
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_f1, best_threshold


# ═════════════════════════════════════════════════════════════
#  Pipeline
# ═════════════════════════════════════════════════════════════

def run_gcpal(
    dataset: str = "HI-Small",
    seeds: List[int] = list(SEEDS),
    hidden_dim: int = 128,
    num_layers: int = 2,
    pretrain_epochs: int = 200,
    finetune_epochs: int = 500,
    tau: float = 0.5,
    lam: float = 0.5,
    edge_drop: float = 0.3,
    feat_drop: float = 0.1,
    w_illicit: float = 0.0,
    patience: int = 50,
    knn_k: int = 5,
    use_knn: bool = False,
    random_split: bool = True,
    lg_k: int = 1,
    contrastive_batch: int = 4096,
    workers: int = 8,
    data_path: Optional[str] = None,
    archive: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    output_tag: str = "",
) -> Dict:
    """Line graph, pre-training, then fine-tuning once per seed.

    ``random_split`` selects the paper's random 60/20/20 over a permutation
    fixed at seed 0, which is what produced the released checkpoints. See the
    module docstring for what that means for the reported numbers.
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_mode = "random" if random_split else "temporal"
    method_name = "GCPAL" + (f"_{output_tag}" if output_tag else "")
    archive = Path(archive) if archive is not None else paths.archive()
    results_dir = Path(results_dir) if results_dir is not None \
        else paths.results() / "ml"

    print(f"\n{'='*60}")
    print(f"  {method_name} --- {dataset}")
    print(f"  Split: {split_mode}, Device: {device}")
    print(f"  hidden={hidden_dim}, layers={num_layers}, lg_k={lg_k}")
    print(f"  Pre-train: {pretrain_epochs} ep, Fine-tune: {finetune_epochs} ep")
    print(f"  patience={patience}, KNN={'yes k='+str(knn_k) if use_knn else 'no'}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    # ── Load the transaction graph ──
    print(f"\n  Loading {dataset} ...")
    graph_data = build_pyg_link_data(dataset, data_path=data_path,
                                     num_threads=workers)
    te_data = graph_data["te_data"]
    t1 = graph_data["t1"]
    t2 = graph_data["t2"]
    n_total = graph_data["n_total"]
    edge_typologies = graph_data.get("edge_typologies", [None] * n_total)

    # ── GFP features: the node features of the line graph ──
    print("  Loading GFP features ...")
    gfp_result = load_snapml_features(dataset, data_path=data_path,
                                      num_threads=workers)
    X_all = np.concatenate([
        gfp_result["X_train"], gfp_result["X_val"], gfp_result["X_test"]
    ], axis=0)
    feat_dim = X_all.shape[1]
    del gfp_result
    print(f"  GFP features: {feat_dim} dims")

    # ── Line graph, then the edge features folded into the nodes ──
    print("  Building line graph (transactions become nodes) ...")
    t0 = time.time()
    lg_edge_index, lg_edge_attr = build_line_graph(
        te_data.edge_index, k_neighbors=lg_k, return_edge_attr=True)
    print(f"  Line graph: {n_total:,} nodes, {lg_edge_index.shape[1]:,} edges"
          f", edge_attr={tuple(lg_edge_attr.shape)} [{time.time() - t0:.1f}s]")

    print("  Pre-aggregating edge features into node features ...")
    extra_node_feats = aggregate_edge_features(lg_edge_index, lg_edge_attr, n_total)
    extra_dim = extra_node_feats.shape[1]
    print(f"  Edge-derived node features: {extra_dim} dims "
          f"[mean_log_delta, mean_sender_src, mean_sender_dst, "
          f"log_degree, min_delta]")
    del lg_edge_attr
    gc.collect()

    # ── Splits ──
    if random_split:
        # Paper protocol: random 60/20/20 over a permutation fixed at seed 0.
        perm = torch.randperm(n_total,
                              generator=torch.Generator().manual_seed(0))
        rt1 = int(n_total * 0.6)
        rt2 = int(n_total * 0.8)
        train_idx = np.sort(perm[:rt1].numpy())
        val_idx = np.sort(perm[rt1:rt2].numpy())
        test_idx = np.sort(perm[rt2:].numpy())
    else:
        train_idx = np.arange(t1)
        val_idx = np.arange(t1, t2)
        test_idx = np.arange(t2, n_total)

    # Z-normalise on train statistics only.
    X_train_feats = X_all[train_idx]
    gfp_mean = X_train_feats.mean(axis=0)
    gfp_std = X_train_feats.std(axis=0) + 1e-8
    X_all_norm = (X_all - gfp_mean) / gfp_std
    X_all_norm = np.clip(X_all_norm, -10, 10)
    del X_train_feats, X_all

    extra_np = extra_node_feats.numpy()
    extra_train = extra_np[train_idx]
    extra_mean = extra_train.mean(axis=0)
    extra_std = extra_train.std(axis=0) + 1e-8
    extra_norm = np.clip((extra_np - extra_mean) / extra_std, -10, 10)
    X_all_norm = np.concatenate([X_all_norm, extra_norm], axis=1)
    feat_dim = X_all_norm.shape[1]
    print(f"  Total node features: {feat_dim} dims "
          f"({feat_dim - extra_dim} GFP + {extra_dim} edge-derived)")
    del extra_node_feats, extra_np, extra_train, extra_norm

    node_features = torch.tensor(X_all_norm, dtype=torch.float, device=device)
    lg_edge_index = lg_edge_index.to(device)
    labels = te_data.y.to(device)

    knn_edge_index_dev = None
    if use_knn:
        knn_edge_index = build_knn_graph(
            node_features, k=knn_k, device=device, chunk_size=1024)
        knn_edge_index_dev = knn_edge_index.to(device)
        del knn_edge_index
        gc.collect()
        torch.cuda.empty_cache()

    lg_data = Data(x=node_features, edge_index=lg_edge_index, y=labels,
                   num_nodes=n_total)

    train_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    train_mask[torch.from_numpy(train_idx).long()] = True
    val_mask[torch.from_numpy(val_idx).long()] = True
    test_mask[torch.from_numpy(test_idx).long()] = True

    n_train = train_mask.sum().item()
    n_val = val_mask.sum().item()
    n_test = test_mask.sum().item()
    n_train_ill = labels[train_mask].sum().item()
    n_val_ill = labels[val_mask].sum().item()
    n_test_ill = labels[test_mask].sum().item()

    print(f"  {split_mode} split:")
    print(f"    Train: {n_train:,} ({n_train_ill:,} illicit, "
          f"{n_train_ill/n_train*100:.3f}%)")
    print(f"    Val:   {n_val:,} ({n_val_ill:,} illicit)")
    print(f"    Test:  {n_test:,} ({n_test_ill:,} illicit)")

    if w_illicit <= 0:
        n_neg = n_train - n_train_ill
        w_illicit = float(np.sqrt(n_neg / max(n_train_ill, 1)))
        print(f"  Auto w_illicit: {w_illicit:.1f} (sqrt({n_neg}/{n_train_ill}))")
    else:
        print(f"  w_illicit: {w_illicit:.1f}")

    # ── Typology head, on the same GFP features ──
    import lightgbm as lgb

    gfp_feat_typ = graph_data["gfp_feat"]

    tv_idx = np.concatenate([train_idx, val_idx])
    y_all_np = labels.cpu().numpy()
    y_tv = y_all_np[tv_idx]

    # Vectorised per class; a Python loop over 4M edges is minutes.
    edge_typ_arr = np.array(edge_typologies, dtype=object)
    tv_typs = edge_typ_arr[tv_idx]
    combined_typ = np.full(len(tv_idx), -1, dtype=int)
    for typ_name, typ_idx in TYPOLOGY_TO_IDX.items():
        combined_typ[(y_tv == 1) & (tv_typs == typ_name)] = typ_idx

    test_typs = edge_typ_arr[test_idx]
    X_typ_pool = gfp_feat_typ[tv_idx]
    X_test_feat = gfp_feat_typ[test_idx]

    typ_mask_arr = combined_typ >= 0
    typ_model = None
    if typ_mask_arr.sum() >= 10:
        typ_params = {
            "objective": "multiclass", "num_class": len(TYPOLOGY_CLASSES),
            "metric": "multi_logloss", "boosting_type": "gbdt",
            "num_leaves": 15, "learning_rate": 0.05, "verbose": -1,
            "n_estimators": 50, "class_weight": "balanced",
        }
        typ_model = lgb.LGBMClassifier(**typ_params)
        typ_model.fit(X_typ_pool[typ_mask_arr], combined_typ[typ_mask_arr])
        print(f"  Typology classifier trained on {typ_mask_arr.sum()} illicit")
    del X_typ_pool

    models_dir = paths.ensure(checkpoint_dir(archive, dataset, method_name))
    probs_dir = paths.ensure(probability_dir(archive, dataset, method_name))

    # ── Phase 1: pre-training, shared by every seed ──
    print("\n  Phase 1: Contrastive pre-training ...")
    torch.manual_seed(42)

    pretrain_model = build_gcpal(feat_dim=feat_dim, hidden_dim=hidden_dim,
                                 num_layers=num_layers, dropout=0.1).to(device)
    pretrain_model = pretrain_gcpal(
        pretrain_model, lg_data, device,
        knn_edge_index=knn_edge_index_dev,
        epochs=pretrain_epochs,
        edge_drop_ratio=edge_drop,
        feat_drop_ratio=feat_drop,
        tau=tau, lam=lam, lr=0.001,
        batch_size=contrastive_batch,
    )
    pretrained_state = {k: v.cpu().clone()
                        for k, v in pretrain_model.state_dict().items()}
    del pretrain_model
    gc.collect()
    torch.cuda.empty_cache()

    pretrain_path = models_dir / "pretrained_encoder.pt"
    torch.save(pretrained_state, pretrain_path)
    print(f"    Saved pretrained encoder: {pretrain_path}")

    # ── Phase 2: fine-tuning, once per seed ──
    print(f"\n  Phase 2: Fine-tuning ({len(seeds)} seeds) ...")

    all_results = []
    for seed in tqdm(seeds, desc=f"  {method_name}|{dataset}", unit="seed"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0_train = time.time()

        model = build_gcpal(feat_dim=feat_dim, hidden_dim=hidden_dim,
                            num_layers=num_layers, dropout=0.1).to(device)
        model.load_state_dict(
            {k: pretrained_state[k].to(device)
             for k in pretrained_state if k in model.state_dict()},
            strict=False,
        )

        best_state, best_val_f1, best_t = finetune_gcpal(
            model, lg_data, train_mask, val_mask, device,
            epochs=finetune_epochs,
            lr=0.001,
            w_illicit=w_illicit,
            eval_every=5,
            patience=patience,
        )

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(lg_data.x, lg_data.edge_index)
            logits = model.classify(h, lg_data.x)
            te_probs = F.softmax(logits[test_mask], dim=-1)[:, 1] \
                        .cpu().float().numpy()
            te_labels = lg_data.y[test_mask].cpu().numpy()

        seed_model_path = models_dir / f"finetuned_seed_{seed}.pt"
        torch.save({
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "best_val_f1": best_val_f1,
            "best_threshold": best_t,
            "seed": seed,
            "feat_dim": feat_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }, seed_model_path)
        print(f"    Saved model: {seed_model_path}")
        del model, best_state

        train_time = time.time() - t0_train

        te_preds = (te_probs >= best_t).astype(int)
        te_preds_05 = (te_probs >= 0.5).astype(int)

        np.save(probs_dir / f"seed_{seed}.npy", te_probs)
        # Beside the probabilities, not one level up as in the pre-release tree.
        # Under a random split these labels are a permuted slice, so they are
        # not the split's test labels and must not sit next to the file that is.
        if not (probs_dir / "test_labels.npy").exists():
            np.save(probs_dir / "test_labels.npy", te_labels)

        typ_acc = 0.0
        typ_macro_f1 = 0.0
        if typ_model is not None:
            typ_pred_all = typ_model.predict(X_test_feat).astype(int)
            np.save(probs_dir / f"seed_{seed}_typ.npy", typ_pred_all)

            test_gt_typ = np.full(n_test, -1, dtype=int)
            for typ_name, typ_i in TYPOLOGY_TO_IDX.items():
                test_gt_typ[(te_labels == 1) & (test_typs == typ_name)] = typ_i

            # Scored on true positives that carry a typology, as for the GBTs.
            tp_with_typ = (te_preds == 1) & (te_labels == 1) & (test_gt_typ >= 0)
            if tp_with_typ.sum() > 0:
                from sklearn.metrics import (accuracy_score as sk_acc_score,
                                             f1_score as sk_f1_score)
                typ_acc = sk_acc_score(test_gt_typ[tp_with_typ],
                                       typ_pred_all[tp_with_typ])
                typ_macro_f1 = sk_f1_score(test_gt_typ[tp_with_typ],
                                           typ_pred_all[tp_with_typ],
                                           average="macro", zero_division=0)

        from sklearn.metrics import (accuracy_score, average_precision_score,
                                     f1_score, precision_score, recall_score)
        prec = precision_score(te_labels, te_preds, zero_division=0)
        rec = recall_score(te_labels, te_preds, zero_division=0)
        f1 = f1_score(te_labels, te_preds, zero_division=0)
        det_acc = accuracy_score(te_labels, te_preds)
        aucpr = average_precision_score(te_labels, te_probs)
        f1_05 = f1_score(te_labels, te_preds_05, zero_division=0)
        oracle_t, oracle_f1 = find_optimal_threshold(te_labels, te_probs)

        all_results.append({
            "method": method_name,
            "dataset": dataset,
            "seed": seed,
            "split": split_mode,
            "threshold": best_t,
            "detection_f1": f1,
            "detection_precision": prec,
            "detection_recall": rec,
            "detection_accuracy": det_acc,
            "detection_pr_auc": aucpr,
            "typology_macro_f1": typ_macro_f1,
            "typology_accuracy": typ_acc,
            "f1_at_05": f1_05,
            "oracle_threshold": oracle_t,
            "oracle_f1": oracle_f1,
            "val_f1": best_val_f1,
            "train_time_s": train_time,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        })

        print(f"    [seed={seed}] F1={f1:.4f}@{best_t:.2f} "
              f"(P={prec:.4f}, R={rec:.4f}) F1@0.5={f1_05:.4f} "
              f"AUCPR={aucpr:.4f} oracle={oracle_f1:.4f}@{oracle_t:.2f} "
              f"typ_acc={typ_acc:.3f} typ_f1={typ_macro_f1:.3f} "
              f"[{train_time:.0f}s]")

        gc.collect()
        torch.cuda.empty_cache()

    _report(all_results, method_name, dataset, split_mode, seeds, probs_dir,
            results_dir)
    return {"results": all_results, "dataset": dataset}


def _report(all_results, method_name, dataset, split_mode, seeds, probs_dir,
            results_dir) -> None:
    """Print the per-seed summary, score the seed ensemble, write both out."""
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score)

    f1s = [r["detection_f1"] for r in all_results]
    aucprs = [r["detection_pr_auc"] for r in all_results]
    oracle_f1s = [r["oracle_f1"] for r in all_results]
    precs = [r["detection_precision"] for r in all_results]
    recs = [r["detection_recall"] for r in all_results]
    det_accs = [r["detection_accuracy"] for r in all_results]
    typ_f1s = [r["typology_macro_f1"] for r in all_results]
    typ_accs = [r["typology_accuracy"] for r in all_results]

    print(f"\n  {method_name}|{dataset} ({split_mode}) Summary:")
    print(f"    F1 (val-t):  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"    Precision:   {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
    print(f"    Recall:      {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
    print(f"    AUCPR:       {np.mean(aucprs):.4f} +/- {np.std(aucprs):.4f}")
    print(f"    Oracle F1:   {np.mean(oracle_f1s):.4f} +/- {np.std(oracle_f1s):.4f}")
    print(f"    Typ macro_f1:{np.mean(typ_f1s):.4f} +/- {np.std(typ_f1s):.4f}")
    print(f"    Typ accuracy:{np.mean(typ_accs):.4f} +/- {np.std(typ_accs):.4f}")

    ens_f1_vt = None
    ens_aucpr = None
    ens_f1_opt = None
    if len(all_results) >= 2:
        seed_probs_list = [np.load(probs_dir / f"seed_{s}.npy")
                           for s in seeds if (probs_dir / f"seed_{s}.npy").exists()]
        if len(seed_probs_list) >= 2:
            ens_probs = np.mean(seed_probs_list, axis=0)
            ens_labels = np.load(probs_dir / "test_labels.npy")
            ens_t, ens_f1_opt = find_optimal_threshold(ens_labels, ens_probs)
            ens_preds = (ens_probs >= ens_t).astype(int)
            ens_prec = precision_score(ens_labels, ens_preds, zero_division=0)
            ens_rec = recall_score(ens_labels, ens_preds, zero_division=0)
            ens_aucpr = average_precision_score(ens_labels, ens_probs)
            mean_val_t = np.mean([r["threshold"] for r in all_results])
            ens_preds_vt = (ens_probs >= mean_val_t).astype(int)
            ens_f1_vt = f1_score(ens_labels, ens_preds_vt, zero_division=0)
            print(f"\n  Ensemble ({len(seed_probs_list)} seeds):")
            print(f"    F1 (val-t={mean_val_t:.2f}): {ens_f1_vt:.4f}")
            print(f"    AUCPR:     {ens_aucpr:.4f}")
            print(f"    Oracle F1: {ens_f1_opt:.4f}@{ens_t:.2f} "
                  f"(P={ens_prec:.4f}, R={ens_rec:.4f})")
            np.save(probs_dir / "ensemble_probs.npy", ens_probs)

    out_dir = paths.ensure(Path(results_dir))
    results_path = out_dir / f"{method_name.lower()}_{split_mode}_metrics.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"    Saved: {results_path}")

    def _fmt(vals):
        return f"{np.mean(vals)*100:.1f}±{np.std(vals)*100:.1f}"

    def _fmt_single(val):
        return f"{val * 100:.1f}" if val is not None else "N/A"

    # Column names and the mean-plus-minus-std string format match
    # results/ml/summary.json, which the table generator reads.
    summary_entry = {
        "method": method_name,
        "model": "non-llm",
        "dataset": dataset,
        "n_seeds": len(all_results),
        "detection_f1": _fmt(f1s),
        "detection_precision": _fmt(precs),
        "detection_recall": _fmt(recs),
        "detection_accuracy": _fmt(det_accs),
        "typology_macro_f1": _fmt(typ_f1s),
        "typology_accuracy": _fmt(typ_accs),
        "verifier_pass_rate": "0.0±0.0",
        "evidence_precision": "0.0±0.0",
        "evidence_recall": "0.0±0.0",
        "evidence_f1": "0.0±0.0",
        "unsupported_claim_rate": "0.0±0.0",
        "ensemble_f1": _fmt_single(ens_f1_vt),
        "ensemble_aucpr": _fmt_single(ens_aucpr),
        "ensemble_oracle_f1": _fmt_single(ens_f1_opt),
    }

    # Kept in its own file rather than merged into results/ml/summary.json:
    # this row carries three ensemble-over-seeds columns the boosted-tree rows
    # do not have, and merging would make that table ragged.
    summary_path = out_dir / f"{method_name.lower()}_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary_list = json.load(f)
        summary_list = [e for e in summary_list
                        if not (e.get("dataset") == dataset
                                and e.get("method") == method_name)]
        summary_list.append(summary_entry)
    else:
        summary_list = [summary_entry]

    with open(summary_path, "w") as f:
        json.dump(summary_list, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {summary_path}")


# ═════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GCPAL, the graph neural ensemble member")
    parser.add_argument("--dataset", type=str, default="HI-Small",
                        choices=list(DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--edge-drop", type=float, default=0.3)
    parser.add_argument("--feat-drop", type=float, default=0.1)
    parser.add_argument("--w-illicit", type=float, default=0.0,
                        help="Class weight for illicit. 0 = auto (sqrt ratio)")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--use-knn", action="store_true",
                        help="Add the KNN third contrastive view (paper's full "
                             "method; used for the released checkpoints)")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Three-view loss mix: lam*L_aug + (1-lam)*L_knn")
    parser.add_argument("--lg-k", type=int, default=1,
                        help="Line-graph neighbours per shared account "
                             "(1 = consecutive, 5 = five nearest in time)")
    parser.add_argument("--random-split", action="store_true",
                        help="Random 60/20/20 instead of temporal. This is what "
                             "produced the released checkpoints, and it leaks "
                             "most of the temporal test split into fine-tuning; "
                             "see the module docstring")
    parser.add_argument("--contrastive-batch", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-path", type=str, default=None,
                        help="AMLworld CSV directory. Default: data/")
    parser.add_argument("--archive", type=str, default=None,
                        help="Run directory for checkpoints and probabilities. "
                             "Default: $AMLC_ARCHIVE")
    parser.add_argument("--results", type=str, default=None,
                        help="Where metrics land. Default: results/ml")
    parser.add_argument("--output-tag", type=str, default="",
                        help="Suffix for the run's output directories")
    args = parser.parse_args()

    import torch

    print("=== GCPAL ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 30 + "\n")

    run_gcpal(
        dataset=args.dataset,
        seeds=args.seeds,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        tau=args.tau,
        lam=args.lam,
        edge_drop=args.edge_drop,
        feat_drop=args.feat_drop,
        w_illicit=args.w_illicit,
        patience=args.patience,
        knn_k=args.knn_k,
        use_knn=args.use_knn,
        random_split=args.random_split,
        lg_k=args.lg_k,
        contrastive_batch=args.contrastive_batch,
        workers=args.workers,
        data_path=args.data_path,
        archive=args.archive,
        results_dir=args.results,
        output_tag=args.output_tag,
    )


if __name__ == "__main__":
    main()
