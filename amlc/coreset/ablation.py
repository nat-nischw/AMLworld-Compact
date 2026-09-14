"""Construction ablation: seven alternative samplers against the HT-Coreset.

Every method draws a subset of the same size as the released HT-Coreset and is
scored with k random draws, so the comparison isolates the construction and not
the budget. This is the paper's ablation table.

    B1  Random Uniform            ablates everything
    B2  Stratified Proportional   ablates deterministic minority retention
    B3  Keep-Illicit + Random     ablates importance weighting
    B4  No Hard-Negatives         ablates hard-negative oversampling
    B5  Fogliato (Neyman)         ECCV 2024, k-means strata + Neyman allocation
    B6  Leskovec random walk      KDD 2006, topology-preserving graph sampling
    B7  Gao stratified            VLDB 2019, stratified by a data attribute
    HT-Coreset                    the reference

B1, B2, B3 and B6 attach no weights, which is the ablation: with a 0.1% illicit
rate an unweighted subset holding roughly a third illicit edges reports a
precision that has nothing to do with the population. B5 and B7 do weight, but
they allocate the minority class proportionally rather than retaining it, so
recall is estimated from a handful of edges.

Reads (the archive is required; B6 and B7 additionally need AMLworld):

    <archive>/test_probs/<dataset>/         labels, typologies, scorer probabilities
    <amlworld>/<dataset>_Trans.csv          accounts and payment format, for B6, B7
    <coreset-dir>/<dataset>/ht_coreset_summary_<dataset>.json   the subset size

Writes, under ``<out>/``:

    baselines_agg.csv          one row per (dataset, method), mean and std over draws
    baselines_draws.csv        one row per draw
    baselines_permodel.csv     per scorer and seed on the canonical draw
    baselines_summary.json     the same aggregate as JSON
    baselines_summary_fmt.json mean +/- std strings, for the table generator
    <dataset>/ablation_<method>_{indices,weights}.npy   the canonical draw

Two defects fixed against the archived script
---------------------------------------------
The HT-Coreset reference held a verbatim copy of the benign sampler including
the spare-fill weighting defect, where the leftover pool was weighted as though
it were an unrepresented stratum. It now calls
:func:`amlc.coreset.sampler.sample_benign_stratified`, the one
implementation that assigns weights from realised stratum membership.

The subset size was a hardcoded table of 3,753 and 2,268 edges, which silently
went stale whenever the construction was rerun. It is now read from the
HT-Coreset summary JSON, so the ablation cannot drift away from the coreset it
claims to match.

The archived script also cached the AMLworld account and payment-format columns
inside the run directory. The release treats the archive as read-only, so the
cache goes to the output directory instead.

Usage:
    python -m amlc.coreset.ablation --datasets HI-Small LI-Small \\
        --amlworld /path/to/amlworld
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .. import paths
from ..config import (
    DATASETS,
    CONSTRUCTION_MEMBERS,
    HARD_NEG_RATIO,
    HARD_NEG_THRESHOLD_FRAC,
    SEEDS,
)
from .common import (
    EVALUATION_THRESHOLD_GRID,
    compute_difficulty,
    ensemble_soft_avg,
    ensemble_typology_vote,
    find_threshold,
    load_member_typologies,
    load_test_data,
    typology_metrics,
    unweighted_metrics,
    weighted_metrics,
)
from .sampler import sample_benign_stratified

#: Ablation methods, in table order. The last one is the reference.
BASELINES = (
    "B1_random_uniform",
    "B2_stratified_proportional",
    "B3_keep_ill_no_iw",
    "B4_no_hardneg",
    "B5_fogliato_neyman",
    "B6_leskovec_rw",
    "B7_gao_stratified",
    "ht_coreset",
)

BASELINE_LABELS = {
    "B1_random_uniform": "B1: Random Uniform",
    "B2_stratified_proportional": "B2: Stratified Proportional",
    "B3_keep_ill_no_iw": "B3: Keep-Illicit + Random (no IW)",
    "B4_no_hardneg": "B4: No Hard-Negatives",
    "B5_fogliato_neyman": "B5: Fogliato et al. (Neyman)",
    "B6_leskovec_rw": "B6: Leskovec random walk",
    "B7_gao_stratified": "B7: Gao et al. stratified",
    "ht_coreset": "HT-Coreset (ours)",
}

#: Draws per method.
K_REPEATS = 50

#: Fraction of the transaction file that is the temporal test split.
TEST_SPLIT_FRACTION = 0.2


def coreset_size(dataset: str, coreset_dir: Optional[Path] = None) -> int:
    """Subset size to match, read from the HT-Coreset summary for this dataset."""
    root = Path(coreset_dir) if coreset_dir else paths.results() / "coreset"
    path = root / dataset / f"ht_coreset_summary_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no HT-Coreset summary at {path}. The ablation matches the coreset's "
            "size, so build it first with "
            f"'python -m amlc.coreset.ht_coreset --datasets {dataset}', or "
            "point --coreset-dir at a directory that holds the summary.")
    with open(path) as f:
        summary = json.load(f)
    best = summary.get("best_feasible")
    if not best or not best.get("n_total"):
        raise ValueError(
            f"{path} records no feasible subset, so there is no size to match.")
    return int(best["n_total"])


# ─────────────────────────────────────────────────────────────────────────
#  Selection strategies
# ─────────────────────────────────────────────────────────────────────────

def select_B1_random_uniform(n_full: int, n_select: int, rng):
    """Uniform sample of edges, no stratification and no weights.

    At a 0.1% illicit rate this catches about four illicit edges in 3,753, so
    recall is estimated from almost nothing and the variance is enormous.
    """
    indices = rng.choice(n_full, n_select, replace=False)
    return indices, np.ones(n_select, dtype=float)


def select_B2_stratified_proportional(labels: np.ndarray, n_select: int, rng):
    """Proportional allocation over the two classes, no weights.

    Neyman's proportional allocation preserves the marginal class ratio, which
    is exactly the problem: the ratio it preserves is 0.1% illicit.
    """
    ill_idx = np.where(labels == 1)[0]
    ben_idx = np.where(labels == 0)[0]
    n = len(labels)

    n_ill_select = max(1, round(n_select * len(ill_idx) / n))
    n_ben_select = n_select - n_ill_select
    n_ill_select = min(n_ill_select, len(ill_idx))
    n_ben_select = min(n_ben_select, len(ben_idx))

    indices = np.sort(np.concatenate([
        rng.choice(ill_idx, n_ill_select, replace=False),
        rng.choice(ben_idx, n_ben_select, replace=False),
    ]))
    return indices, np.ones(len(indices), dtype=float)


def select_B3_keep_ill_no_iw(labels: np.ndarray, n_select: int, rng):
    """Every illicit edge plus uniform benign edges, with no weights.

    Recall is exact; precision is not, because the subset's illicit share is
    about a third against 0.1% in the population.
    """
    ill_idx = np.where(labels == 1)[0]
    ben_idx = np.where(labels == 0)[0]

    n_ben_select = n_select - len(ill_idx)
    if n_ben_select <= 0:
        return ill_idx.copy(), np.ones(len(ill_idx), dtype=float)

    n_ben_select = min(n_ben_select, len(ben_idx))
    sel_ben = rng.choice(ben_idx, n_ben_select, replace=False)
    indices = np.sort(np.concatenate([ill_idx, sel_ben]))
    return indices, np.ones(len(indices), dtype=float)


def select_B4_no_hardneg(labels: np.ndarray, ens_probs: np.ndarray,
                         difficulty: np.ndarray, n_select: int, rng):
    """Every illicit edge plus difficulty-tercile benign edges, with weights.

    The one component removed is hard-negative oversampling, so the benign
    sample is drawn from the whole pool by difficulty tercile.
    """
    ill_idx = np.where(labels == 1)[0]
    ben_idx = np.where(labels == 0)[0]

    n_ben_select = n_select - len(ill_idx)
    if n_ben_select <= 0:
        return ill_idx.copy(), np.ones(len(ill_idx), dtype=float)
    n_ben_select = min(n_ben_select, len(ben_idx))

    diff_ben = difficulty[ben_idx]
    q33, q67 = np.percentile(diff_ben, [33, 67])
    strata = [diff_ben < q33,
              (diff_ben >= q33) & (diff_ben < q67),
              diff_ben >= q67]

    n_per_stratum = max(1, n_ben_select // 3)
    selected_parts, weight_parts = [], []
    for mask in strata:
        pool = ben_idx[mask]
        n_take = min(n_per_stratum, len(pool))
        if n_take > 0:
            selected_parts.append(rng.choice(pool, n_take, replace=False))
            weight_parts.append(np.full(n_take, len(pool) / n_take))

    if not selected_parts:
        return ill_idx.copy(), np.ones(len(ill_idx), dtype=float)

    sel_ben = np.concatenate(selected_parts)[:n_ben_select]
    ben_w = np.concatenate(weight_parts)[:n_ben_select]

    indices = np.sort(np.concatenate([ill_idx, sel_ben]))
    weights = np.ones(len(indices), dtype=float)
    weights[np.searchsorted(indices, sel_ben)] = ben_w
    return indices, weights


def precompute_fogliato_strata(ens_probs: np.ndarray, labels: np.ndarray,
                               n_strata: int = 10) -> dict:
    """K-means strata for B5, computed once rather than once per draw.

    Repeated KMeans calls inside the draw loop corrupted OpenBLAS memory on the
    run machine, and the strata do not depend on the draw anyway.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_strata, random_state=42, n_init=10)
    strata_labels = km.fit_predict(ens_probs.reshape(-1, 1))
    stratum_ids = np.unique(strata_labels)

    # Fogliato allocates on the spread of binary correctness inside a stratum.
    correct = ((ens_probs >= 0.5).astype(float) == labels).astype(float)

    N_h, S_h, idx_h = {}, {}, {}
    for h in stratum_ids:
        mask = strata_labels == h
        idx_h[h] = np.where(mask)[0]
        N_h[h] = int(mask.sum())
        S_h[h] = max(float(correct[mask].std()), 1e-8)

    return dict(stratum_ids=stratum_ids, N_h=N_h, S_h=S_h, idx_h=idx_h)


def select_B5_fogliato_neyman(fogliato_strata: dict, n_select: int, rng):
    """Stratified sampling with Neyman allocation and HT weights.

    Fogliato et al., "A Statistical Framework for Efficient Model Evaluation",
    ECCV 2024 (arXiv:2406.07320). Strata come from k-means on the ensemble
    probability; stratum h takes n_h proportional to N_h * S_h and is weighted
    by N_h / n_h.

    The difference from the HT-Coreset is that nothing retains the minority
    class: at a 0.1% positive rate most strata are pure benign, so the illicit
    edges are undersampled and recall estimation collapses.
    """
    stratum_ids = fogliato_strata["stratum_ids"]
    N_h, S_h, idx_h = (fogliato_strata["N_h"], fogliato_strata["S_h"],
                       fogliato_strata["idx_h"])

    total_weight = sum(N_h[h] * S_h[h] for h in stratum_ids)
    n_h_alloc = {h: max(1, int(round(n_select * N_h[h] * S_h[h] / total_weight)))
                 for h in stratum_ids}

    diff = n_select - sum(n_h_alloc.values())
    if diff != 0:
        largest_h = max(stratum_ids, key=lambda h: N_h[h])
        n_h_alloc[largest_h] = max(1, n_h_alloc[largest_h] + diff)

    all_indices, all_weights = [], []
    for h in stratum_ids:
        pool = idx_h[h]
        n_take = min(n_h_alloc[h], len(pool))
        if n_take > 0:
            all_indices.append(rng.choice(pool, n_take, replace=False))
            all_weights.append(np.full(n_take, len(pool) / n_take))

    indices = np.concatenate(all_indices)
    weights = np.concatenate(all_weights)
    order = np.argsort(indices)
    return indices[order], weights[order]


def select_B6_leskovec_rw(adjacency: dict, all_accounts: np.ndarray,
                          n_full: int, n_select: int, rng,
                          teleport_prob: float = 0.15):
    """Random-walk sampling over the account graph, no weights.

    Leskovec and Faloutsos, "Sampling from Large Graphs", KDD 2006. The walk
    follows a random incident edge and teleports with probability 0.15 so it
    cannot be trapped in a component. It preserves topology and ignores labels,
    so it behaves like B1 on class balance.
    """
    visited_edges: set[int] = set()
    n_accounts = len(all_accounts)
    current = all_accounts[rng.integers(n_accounts)]

    for _ in range(n_select * 100):
        if len(visited_edges) >= n_select:
            break
        if rng.random() < teleport_prob:
            current = all_accounts[rng.integers(n_accounts)]
            continue
        neighbours = adjacency.get(current, [])
        if not neighbours:
            current = all_accounts[rng.integers(n_accounts)]
            continue
        edge_idx, next_account = neighbours[rng.integers(len(neighbours))]
        visited_edges.add(edge_idx)
        current = next_account

    if len(visited_edges) < n_select:
        available = np.setdiff1d(np.arange(n_full), list(visited_edges))
        if len(available) > 0:
            extra = rng.choice(available,
                               min(n_select - len(visited_edges), len(available)),
                               replace=False)
            visited_edges.update(extra.tolist())

    indices = np.sort(np.array(list(visited_edges))[:n_select])
    return indices, np.ones(len(indices), dtype=float)


def select_B7_gao_stratified(payment_format: np.ndarray, n_select: int, rng):
    """Stratified sampling by a data attribute with HT weights.

    Gao et al., "Efficient Knowledge Graph Accuracy Evaluation", VLDB 2019.
    Strata are payment formats, the AML analogue of a relation type, allocated
    proportionally and weighted by N_h / n_h. Unlike B5 the strata come from the
    data rather than from the model, and like B5 nothing retains the minority
    class.
    """
    n_full = len(payment_format)
    strata_info = [(fmt, np.where(payment_format == fmt)[0])
                   for fmt in np.unique(payment_format)]
    strata_info = [(fmt, pool, len(pool)) for fmt, pool in strata_info]
    strata_info.sort(key=lambda x: -x[2])

    all_indices, all_weights = [], []
    remaining = n_select
    for i, (_fmt, pool, N_h) in enumerate(strata_info):
        # The last stratum absorbs the rounding remainder.
        n_h = remaining if i == len(strata_info) - 1 else max(
            1, round(n_select * N_h / n_full))
        n_h = min(n_h, N_h, remaining)
        if n_h > 0:
            all_indices.append(rng.choice(pool, n_h, replace=False))
            all_weights.append(np.full(n_h, N_h / n_h))
            remaining -= n_h
        if remaining <= 0:
            break

    indices = np.concatenate(all_indices)
    weights = np.concatenate(all_weights)
    order = np.argsort(indices)
    return indices[order], weights[order]


def select_ht_coreset(labels: np.ndarray, ens_probs: np.ndarray,
                      difficulty: np.ndarray, threshold: float,
                      n_select: int, rng,
                      hard_neg_ratio: float = HARD_NEG_RATIO):
    """The reference: every illicit edge plus the stratified benign sample.

    The benign half is drawn by :func:`sample_benign_stratified`, so this is the
    same code path the released coreset was built with, spare-fill fix included.
    """
    ill_idx = np.where(labels == 1)[0]
    ben_idx = np.where(labels == 0)[0]

    n_ben_select = n_select - len(ill_idx)
    if n_ben_select <= 0:
        return ill_idx.copy(), np.ones(len(ill_idx), dtype=float)
    n_ben_select = min(n_ben_select, len(ben_idx))

    sel_ben, ben_w, _ = sample_benign_stratified(
        rng, ben_idx, ens_probs, threshold, n_ben_select,
        hard_neg_ratio=hard_neg_ratio, difficulty=difficulty)

    indices = np.sort(np.concatenate([ill_idx, sel_ben]))
    weights = np.ones(len(indices), dtype=float)
    weights[np.searchsorted(indices, sel_ben)] = ben_w
    return indices, weights


# ─────────────────────────────────────────────────────────────────────────
#  Graph structure and edge attributes, for B6 and B7
# ─────────────────────────────────────────────────────────────────────────

def load_test_edge_data(dataset: str, amlworld_dir: Optional[Path],
                        cache_dir: Path):
    """Account pairs and payment format for the test-split edges.

    B6 walks the account graph and B7 stratifies by payment format, so both need
    the raw transaction file. Returns ``(None, None, None)`` when it is not
    available, and the two baselines are skipped rather than faked.

    The parsed columns are cached under ``cache_dir``; the archived script wrote
    that cache into the run directory, which the release keeps read-only.
    """
    cache_dir = paths.ensure(Path(cache_dir))
    src_cache = cache_dir / f"{dataset}_test_src_accounts.npy"
    dst_cache = cache_dir / f"{dataset}_test_dst_accounts.npy"
    pf_cache = cache_dir / f"{dataset}_test_payment_format.npy"

    if src_cache.exists() and dst_cache.exists() and pf_cache.exists():
        print(f"  Loading cached edge attributes for {dataset}")
        return (np.load(src_cache, allow_pickle=True),
                np.load(dst_cache, allow_pickle=True),
                np.load(pf_cache, allow_pickle=True))

    if amlworld_dir is None:
        print("  No --amlworld directory given; B6 and B7 will be skipped.")
        return None, None, None

    csv_path = Path(amlworld_dir) / f"{dataset}_Trans.csv"
    if not csv_path.exists():
        print(f"  {csv_path} not found; B6 and B7 will be skipped.")
        return None, None, None

    import pandas as pd

    print(f"  Reading edge attributes from {csv_path} (cached afterwards)")
    # Positional columns, because the file repeats the header name "Account".
    df = pd.read_csv(csv_path, header=None, skiprows=1,
                     usecols=[1, 2, 3, 4, 9], dtype=str)
    df.columns = ["From_Bank", "From_Account", "To_Bank", "To_Account",
                  "Payment_Format"]

    # The temporal split puts the last 20% of the file in the test set.
    df_test = df.iloc[int(len(df) * (1 - TEST_SPLIT_FRACTION)):].reset_index(
        drop=True)
    del df

    src = (df_test["From_Bank"].str.strip() + "_"
           + df_test["From_Account"].str.strip()).values
    dst = (df_test["To_Bank"].str.strip() + "_"
           + df_test["To_Account"].str.strip()).values
    pf = df_test["Payment_Format"].str.strip().values

    np.save(src_cache, src)
    np.save(dst_cache, dst)
    np.save(pf_cache, pf)
    print(f"  Cached {len(src):,} test edges, "
          f"{len(np.unique(pf))} payment formats")
    return src, dst, pf


def build_test_adjacency(src_accounts: np.ndarray,
                         dst_accounts: np.ndarray) -> dict:
    """Undirected adjacency ``account -> [(edge index, neighbour), ...]``."""
    adj = defaultdict(list)
    for i in range(len(src_accounts)):
        s, d = src_accounts[i], dst_accounts[i]
        adj[s].append((i, d))
        if s != d:
            adj[d].append((i, s))
    return adj


# ─────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────

def eval_on_subset(probs: np.ndarray, labels: np.ndarray, threshold: float,
                   indices: np.ndarray, weights: np.ndarray):
    """Weighted and unweighted metrics for one scorer on one subset."""
    sub_labels = labels[indices]
    sub_preds = (probs[indices] >= threshold).astype(int)
    return (weighted_metrics(sub_labels, sub_preds, weights),
            unweighted_metrics(sub_labels, sub_preds))


def run_ablation(dataset: str, out_dir: Path,
                 members: Sequence[str] = CONSTRUCTION_MEMBERS,
                 seeds: Sequence[int] = SEEDS,
                 archive: Optional[Path] = None,
                 coreset_dir: Optional[Path] = None,
                 amlworld_dir: Optional[Path] = None,
                 k_repeats: int = K_REPEATS,
                 seed_base: int = 0,
                 hard_neg_ratio: float = HARD_NEG_RATIO) -> dict:
    """Run every ablation method on one dataset at the HT-Coreset's size."""
    n_target = coreset_size(dataset, coreset_dir)

    data = load_test_data(dataset, members, seeds, archive=archive)
    labels = data["labels"]
    gt_typ = data["typologies"]
    n_full = data["n"]
    n_ill = int(labels.sum())

    print(f"\n{'='*100}")
    print(f"  Construction ablation - {dataset}")
    print(f"  Target subset size: {n_target:,} edges "
          f"({(1 - n_target/n_full)*100:.1f}% reduction), k={k_repeats} draws")
    print(f"  Full test set: {n_full:,} edges "
          f"({n_ill:,} illicit {n_ill/n_full*100:.3f}%)")
    print(f"{'='*100}")

    model_probs = data["probs"]
    ens_probs = ensemble_soft_avg(data)
    difficulty = compute_difficulty(data)
    threshold, _ = find_threshold(labels, ens_probs,
                                  grid=EVALUATION_THRESHOLD_GRID)

    full_preds = (ens_probs >= threshold).astype(int)
    full_met = unweighted_metrics(labels, full_preds)

    member_typ = load_member_typologies(dataset, members, seeds, archive=archive)
    has_typ = len(member_typ) >= 2
    full_typ_met = dict(typ_macro_f1=0.0, typ_accuracy=0.0, typ_n_eval=0)
    ens_typ_full = None
    if has_typ:
        ens_typ_full = ensemble_typology_vote(list(member_typ.values()), n_full)
        full_typ_met = typology_metrics(gt_typ, full_preds, ens_typ_full)
        print(f"  Typology: {(gt_typ >= 0).sum():,} typed edges, "
              f"{len(member_typ)} typology heads")

    print(f"  Ensemble threshold {threshold:.2f}: "
          f"P={full_met['precision']*100:.2f}% R={full_met['recall']*100:.2f}% "
          f"F1={full_met['f1']*100:.2f}%")

    full_model_met = {}
    for key, probs in model_probs.items():
        t, _ = find_threshold(labels, probs, grid=EVALUATION_THRESHOLD_GRID)
        full_model_met[key] = unweighted_metrics(labels,
                                                 (probs >= t).astype(int))
        full_model_met[key]["threshold"] = t

    hard_cut = threshold * HARD_NEG_THRESHOLD_FRAC
    print(f"  Hard negatives (prob >= {hard_cut:.2f}): "
          f"{int((ens_probs[labels == 0] >= hard_cut).sum()):,}")

    print("  Pre-computing the Fogliato k-means strata (K=10) ...")
    fogliato_strata = precompute_fogliato_strata(ens_probs, labels)

    src_accounts, dst_accounts, payment_format = load_test_edge_data(
        dataset, amlworld_dir, Path(out_dir) / "edge_attributes")
    has_edge_data = src_accounts is not None
    adjacency = all_accounts = None
    if has_edge_data:
        adjacency = build_test_adjacency(src_accounts, dst_accounts)
        all_accounts = np.array(list(adjacency.keys()))
        print(f"  Adjacency: {len(all_accounts):,} accounts")

    selectors = {
        "B1_random_uniform": lambda rng: select_B1_random_uniform(
            n_full, n_target, rng),
        "B2_stratified_proportional": lambda rng: select_B2_stratified_proportional(
            labels, n_target, rng),
        "B3_keep_ill_no_iw": lambda rng: select_B3_keep_ill_no_iw(
            labels, n_target, rng),
        "B4_no_hardneg": lambda rng: select_B4_no_hardneg(
            labels, ens_probs, difficulty, n_target, rng),
        "B5_fogliato_neyman": lambda rng: select_B5_fogliato_neyman(
            fogliato_strata, n_target, rng),
        "ht_coreset": lambda rng: select_ht_coreset(
            labels, ens_probs, difficulty, threshold, n_target, rng,
            hard_neg_ratio),
    }
    if has_edge_data:
        selectors["B6_leskovec_rw"] = lambda rng: select_B6_leskovec_rw(
            adjacency, all_accounts, n_full, n_target, rng)
        selectors["B7_gao_stratified"] = lambda rng: select_B7_gao_stratified(
            payment_format, n_target, rng)

    all_draws, all_agg, all_permodel = [], [], []
    subset_dir = paths.ensure(Path(out_dir) / dataset)

    print(f"\n{'-'*118}")
    print(f"  {'Method':<35s} | {'wP':>6} {'wR':>6} {'wF1':>6} | "
          f"{'dP':>6} {'dR':>6} {'dF1':>6} | {'uwP':>6} {'uwR':>6} {'uwF1':>6} | "
          f"{'n_ill':>6} {'sd(wF1)':>8}")
    print(f"{'-'*118}")

    for name in BASELINES:
        if name not in selectors:
            print(f"  {BASELINE_LABELS[name]:<35s} | skipped, no AMLworld edge data")
            continue
        select = selectors[name]

        draw_results = []
        for i in range(k_repeats):
            rng = np.random.default_rng(seed_base + i * 7919)
            indices, weights = select(rng)
            w_met, uw_met = eval_on_subset(ens_probs, labels, threshold,
                                           indices, weights)

            sub_typ_macro_f1 = sub_typ_accuracy = 0.0
            if has_typ:
                sub_typ_met = typology_metrics(
                    gt_typ[indices],
                    (ens_probs[indices] >= threshold).astype(int),
                    ens_typ_full[indices])
                sub_typ_macro_f1 = sub_typ_met["typ_macro_f1"]
                sub_typ_accuracy = sub_typ_met["typ_accuracy"]

            draw = dict(
                dataset=dataset, method=name, draw_id=i,
                n_subset=len(indices), n_illicit=int(labels[indices].sum()),
                w_precision=w_met["precision"], w_recall=w_met["recall"],
                w_f1=w_met["f1"],
                delta_p=abs(w_met["precision"] - full_met["precision"]),
                delta_r=abs(w_met["recall"] - full_met["recall"]),
                delta_f1=abs(w_met["f1"] - full_met["f1"]),
                uw_precision=uw_met["precision"], uw_recall=uw_met["recall"],
                uw_f1=uw_met["f1"],
                delta_p_uw=abs(uw_met["precision"] - full_met["precision"]),
                delta_r_uw=abs(uw_met["recall"] - full_met["recall"]),
                delta_f1_uw=abs(uw_met["f1"] - full_met["f1"]),
                typ_macro_f1=sub_typ_macro_f1,
                typ_accuracy=sub_typ_accuracy,
                delta_typ_macro_f1=abs(sub_typ_macro_f1
                                       - full_typ_met["typ_macro_f1"]),
                delta_typ_accuracy=abs(sub_typ_accuracy
                                       - full_typ_met["typ_accuracy"]),
            )
            draw_results.append(draw)
            all_draws.append(draw)

        agg = dict(dataset=dataset, method=name, label=BASELINE_LABELS[name],
                   n_subset=draw_results[0]["n_subset"],
                   reduction_pct=(1 - draw_results[0]["n_subset"] / n_full) * 100,
                   k_repeats=k_repeats)
        for k in ("w_precision", "w_recall", "w_f1",
                  "delta_p", "delta_r", "delta_f1",
                  "uw_precision", "uw_recall", "uw_f1",
                  "delta_p_uw", "delta_r_uw", "delta_f1_uw",
                  "typ_macro_f1", "typ_accuracy",
                  "delta_typ_macro_f1", "delta_typ_accuracy", "n_illicit"):
            vals = [d[k] for d in draw_results]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
            agg[f"{k}_min"] = float(np.min(vals))
            agg[f"{k}_max"] = float(np.max(vals))
        all_agg.append(agg)

        print(f"  {BASELINE_LABELS[name]:<35s} | "
              f"{agg['w_precision_mean']*100:>5.1f}% "
              f"{agg['w_recall_mean']*100:>5.1f}% "
              f"{agg['w_f1_mean']*100:>5.1f}% | "
              f"{agg['delta_p_mean']*100:>5.1f}% "
              f"{agg['delta_r_mean']*100:>5.1f}% "
              f"{agg['delta_f1_mean']*100:>5.1f}% | "
              f"{agg['uw_precision_mean']*100:>5.1f}% "
              f"{agg['uw_recall_mean']*100:>5.1f}% "
              f"{agg['uw_f1_mean']*100:>5.1f}% | "
              f"{agg['n_illicit_mean']:>6.0f} {agg['w_f1_std']*100:>7.2f}%")

        # Per-scorer evaluation on the canonical draw (draw 0).
        canon_idx, canon_w = select(np.random.default_rng(seed_base))
        np.save(subset_dir / f"ablation_{name}_indices.npy", canon_idx)
        np.save(subset_dir / f"ablation_{name}_weights.npy", canon_w)

        for key, probs in model_probs.items():
            member, seed = key
            thresh = full_model_met[key]["threshold"]
            w_met_m, uw_met_m = eval_on_subset(probs, labels, thresh,
                                               canon_idx, canon_w)
            full_m = full_model_met[key]

            pm_full_typ = dict(typ_macro_f1=0.0, typ_accuracy=0.0)
            pm_sub_typ = dict(typ_macro_f1=0.0, typ_accuracy=0.0)
            if key in member_typ:
                typ_pred = member_typ[key]
                pm_full_typ = typology_metrics(
                    gt_typ, (probs >= thresh).astype(int), typ_pred)
                pm_sub_typ = typology_metrics(
                    gt_typ[canon_idx],
                    (probs[canon_idx] >= thresh).astype(int),
                    typ_pred[canon_idx])

            all_permodel.append(dict(
                dataset=dataset, baseline=name, label=BASELINE_LABELS[name],
                model=member, seed=seed, threshold=thresh,
                n_full=n_full, n_subset=len(canon_idx),
                reduction_pct=(1 - len(canon_idx) / n_full) * 100,
                full_precision=full_m["precision"],
                full_recall=full_m["recall"], full_f1=full_m["f1"],
                w_precision=w_met_m["precision"], w_recall=w_met_m["recall"],
                w_f1=w_met_m["f1"],
                delta_f1_w=abs(w_met_m["f1"] - full_m["f1"]),
                uw_precision=uw_met_m["precision"],
                uw_recall=uw_met_m["recall"], uw_f1=uw_met_m["f1"],
                delta_f1_uw=abs(uw_met_m["f1"] - full_m["f1"]),
                full_typ_macro_f1=pm_full_typ["typ_macro_f1"],
                full_typ_accuracy=pm_full_typ["typ_accuracy"],
                typ_macro_f1=pm_sub_typ["typ_macro_f1"],
                typ_accuracy=pm_sub_typ["typ_accuracy"],
                delta_typ_macro_f1=abs(pm_sub_typ["typ_macro_f1"]
                                       - pm_full_typ["typ_macro_f1"]),
                delta_typ_accuracy=abs(pm_sub_typ["typ_accuracy"]
                                       - pm_full_typ["typ_accuracy"]),
            ))

    print(f"{'-'*118}")
    print(f"  Full-set reference: P={full_met['precision']*100:.2f}%  "
          f"R={full_met['recall']*100:.2f}%  F1={full_met['f1']*100:.2f}%")

    print("\n  Per-scorer mean |dF1| on the canonical draw")
    for name in BASELINES:
        rows = [r for r in all_permodel if r["baseline"] == name]
        if not rows:
            continue
        parts = []
        for member in members:
            m_rows = [r["delta_f1_w"] for r in rows if r["model"] == member]
            if m_rows:
                parts.append(f"{member}={np.mean(m_rows)*100:.1f}%")
        ens = [a for a in all_agg if a["method"] == name]
        ens_txt = f"ensemble={ens[0]['delta_f1_mean']*100:.1f}%" if ens else ""
        print(f"    {BASELINE_LABELS[name]:<35s} " + "  ".join(parts + [ens_txt]))

    return dict(dataset=dataset, n_full=n_full, n_target=n_target,
                n_ill=n_ill, threshold=threshold, full_metrics=full_met,
                agg=all_agg, draws=all_draws, permodel=all_permodel)


# ─────────────────────────────────────────────────────────────────────────
#  Result files
# ─────────────────────────────────────────────────────────────────────────

_AGG_COLS = ["dataset", "method", "label", "n_subset", "reduction_pct",
             "k_repeats",
             "w_precision_mean", "w_precision_std",
             "w_recall_mean", "w_recall_std",
             "w_f1_mean", "w_f1_std",
             "delta_p_mean", "delta_p_std",
             "delta_r_mean", "delta_r_std",
             "delta_f1_mean", "delta_f1_std",
             "uw_precision_mean", "uw_precision_std",
             "uw_recall_mean", "uw_recall_std",
             "uw_f1_mean", "uw_f1_std",
             "typ_macro_f1_mean", "typ_macro_f1_std",
             "typ_accuracy_mean", "typ_accuracy_std",
             "delta_typ_macro_f1_mean", "delta_typ_macro_f1_std",
             "delta_typ_accuracy_mean", "delta_typ_accuracy_std",
             "n_illicit_mean", "n_illicit_std"]

_DRAW_COLS = ["dataset", "method", "draw_id", "n_subset", "n_illicit",
              "w_precision", "w_recall", "w_f1",
              "delta_p", "delta_r", "delta_f1",
              "uw_precision", "uw_recall", "uw_f1",
              "delta_p_uw", "delta_r_uw", "delta_f1_uw",
              "typ_macro_f1", "typ_accuracy",
              "delta_typ_macro_f1", "delta_typ_accuracy"]

_PERMODEL_COLS = ["dataset", "baseline", "label", "model", "seed", "threshold",
                  "n_full", "n_subset", "reduction_pct",
                  "full_precision", "full_recall", "full_f1",
                  "w_precision", "w_recall", "w_f1", "delta_f1_w",
                  "uw_precision", "uw_recall", "uw_f1", "delta_f1_uw",
                  "full_typ_macro_f1", "full_typ_accuracy",
                  "typ_macro_f1", "typ_accuracy",
                  "delta_typ_macro_f1", "delta_typ_accuracy"]


def _write_csv(path: Path, cols: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def save_results(results_by_ds: dict, out_dir: Path) -> None:
    """Write the three CSVs and the two JSON summaries."""
    out = paths.ensure(Path(out_dir))

    _write_csv(out / "baselines_agg.csv", _AGG_COLS,
               [r for res in results_by_ds.values() for r in res["agg"]])
    _write_csv(out / "baselines_draws.csv", _DRAW_COLS,
               [r for res in results_by_ds.values() for r in res["draws"]])
    _write_csv(out / "baselines_permodel.csv", _PERMODEL_COLS,
               [r for res in results_by_ds.values() for r in res["permodel"]])

    summary = {
        ds: dict(n_full=res["n_full"], n_target=res["n_target"],
                 n_ill=res["n_ill"], threshold=float(res["threshold"]),
                 full_metrics={k: float(v) for k, v in res["full_metrics"].items()},
                 baselines=res["agg"])
        for ds, res in results_by_ds.items()
    }
    with open(out / "baselines_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out / 'baselines_summary.json'}")

    def fmt(mean, std):
        return f"{mean*100:.1f}+/-{std*100:.1f}"

    rows = []
    for ds, res in results_by_ds.items():
        for a in res["agg"]:
            rows.append({
                "method": a["label"], "dataset": ds,
                "k_repeats": a["k_repeats"], "n_subset": a["n_subset"],
                "reduction_pct": f"{a['reduction_pct']:.1f}",
                "w_precision": fmt(a["w_precision_mean"], a["w_precision_std"]),
                "w_recall": fmt(a["w_recall_mean"], a["w_recall_std"]),
                "w_f1": fmt(a["w_f1_mean"], a["w_f1_std"]),
                "delta_f1": fmt(a["delta_f1_mean"], a["delta_f1_std"]),
                "uw_precision": fmt(a["uw_precision_mean"], a["uw_precision_std"]),
                "uw_recall": fmt(a["uw_recall_mean"], a["uw_recall_std"]),
                "uw_f1": fmt(a["uw_f1_mean"], a["uw_f1_std"]),
                "typ_macro_f1": fmt(a["typ_macro_f1_mean"], a["typ_macro_f1_std"]),
                "typ_accuracy": fmt(a["typ_accuracy_mean"], a["typ_accuracy_std"]),
                "delta_typ_macro_f1": fmt(a["delta_typ_macro_f1_mean"],
                                          a["delta_typ_macro_f1_std"]),
                "delta_typ_accuracy": fmt(a["delta_typ_accuracy_mean"],
                                          a["delta_typ_accuracy_std"]),
                "n_illicit": f"{a['n_illicit_mean']:.0f}+/-{a['n_illicit_std']:.0f}",
            })
    with open(out / "baselines_summary_fmt.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out / 'baselines_summary_fmt.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Construction ablation for the HT-Coreset")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--members", nargs="+", default=list(CONSTRUCTION_MEMBERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--archive", type=Path, default=None,
                    help="run directory outputs/; defaults to AMLC_ARCHIVE")
    ap.add_argument("--out", type=Path, default=paths.results() / "coreset")
    ap.add_argument("--coreset-dir", type=Path, default=None,
                    help="directory holding <dataset>/ht_coreset_summary_"
                         "<dataset>.json; defaults to --out")
    ap.add_argument("--amlworld", type=Path, default=None,
                    help="directory holding <dataset>_Trans.csv; without it B6 "
                         "and B7 are skipped")
    ap.add_argument("--k-repeats", type=int, default=K_REPEATS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hard-neg-ratio", type=float, default=HARD_NEG_RATIO)
    args = ap.parse_args()

    print(f"{'='*100}")
    print("  Construction ablation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Datasets: {args.datasets}   draws: {args.k_repeats}")
    print(f"{'='*100}")

    coreset_dir = args.coreset_dir or args.out
    results_by_ds = {}
    for ds in args.datasets:
        results_by_ds[ds] = run_ablation(
            ds, out_dir=args.out, members=args.members, seeds=args.seeds,
            archive=args.archive, coreset_dir=coreset_dir,
            amlworld_dir=args.amlworld, k_repeats=args.k_repeats,
            seed_base=args.seed, hard_neg_ratio=args.hard_neg_ratio)

    if results_by_ds:
        save_results(results_by_ds, args.out)

    print(f"\n{'='*100}")
    print("  Ablation summary")
    for ds, res in results_by_ds.items():
        print(f"\n  {ds}  (full {res['n_full']:,} -> {res['n_target']:,} edges)")
        for a in res["agg"]:
            print(f"    {a['label']:<35s} wF1={a['w_f1_mean']*100:>5.1f}% "
                  f"+/-{a['w_f1_std']*100:.1f}%  "
                  f"dF1={a['delta_f1_mean']*100:>5.1f}%")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
