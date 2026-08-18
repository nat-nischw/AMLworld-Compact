"""Re-inference of the GCPAL checkpoints on the temporal test split.

The paper's third ensemble member, GCPAL+GFP, is not separately trained. It is
the checkpoints that :mod:`amlc.baselines.ml.gcpal` fine-tuned under the
paper-faithful random 60/20/20 split, loaded here and run forward over the
temporal split that LightGBM+GFP and XGBoost+GFP were trained and evaluated on.
Without this step the three members' probability arrays index different edges
and cannot be averaged at all.

That is the whole reason the two artefacts live in different places in the
archived run directory:

    <archive>/models/<dataset>/GCPAL_knn/finetuned_seed_<seed>.pt   the weights
    <archive>/test_probs/<dataset>/GCPAL_knn_temporal/seed_<seed>.npy  the probs

The weights are named for the training run. The probabilities are named for the
inference pass, and that directory name is the one
:mod:`amlc.archive` maps to the paper's ``GCPAL+GFP``. One directory
is not a stale copy of the other; they are two stages.

What re-inference does and does not fix. It fixes alignment: after it, all three
members score the same 1,015,669 or 1,384,810 edges in the same order, and the
decision threshold is retuned on the temporal validation split rather than
carried over. It does not fix leakage. Fine-tuning saw a random 60 percent of
all edges, which includes most of the temporal test split, and no amount of
re-inference removes that. See the note in :mod:`gcpal`.

Reads
    the AMLworld dataset, the Snap ML feature matrix, and the fine-tuned
    checkpoints.
Writes
    ``seed_<seed>.npy`` and ``seed_<seed>_typ.npy`` under the member's archive
    directory, and per-seed metrics under ``results/ml/``.

The feature matrix is rebuilt here rather than reloaded, and it has to come out
identical to the training one or the checkpoint will not load: the same line
graph at ``lg_k=5``, the same five edge-derived node dimensions, 84 in total.
The one deliberate difference is that normalisation statistics come from the
temporal training window, matching the convention the boosted trees use. Every
checkpoint records its own ``feat_dim`` and a mismatch is skipped rather than
forced.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ... import paths
from ...config import DATASETS, SEEDS
from ...archive import legacy_path, member_dir
from ...typology import TYPOLOGY_CLASSES, TYPOLOGY_TO_IDX
from ..metrics import find_optimal_threshold
from .gbt import build_pyg_link_data, load_snapml_features
from .gcpal import (aggregate_edge_features, build_gcpal, build_line_graph,
                    checkpoint_dir)

#: The paper name of the member these checkpoints become.
MEMBER = "GCPAL+GFP"

#: Suffix of the training run whose checkpoints were released. The run was
#: launched with ``--output-tag knn`` because it enabled the KNN third
#: contrastive view, so its checkpoint directory is ``GCPAL_knn``.
DEFAULT_RUN_TAG = "knn"

#: Line-graph neighbours per shared account. Must match the training run.
DEFAULT_LG_K = 5


def run_inference(
    dataset: str,
    seeds: List[int],
    lg_k: int = DEFAULT_LG_K,
    workers: int = 16,
    run_tag: str = DEFAULT_RUN_TAG,
    data_path: Optional[str] = None,
    archive: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> List[dict]:
    """Load one dataset's checkpoints and score the temporal test split."""
    import lightgbm as lgb
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score, precision_score, recall_score
    from torch_geometric.data import Data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archive = Path(archive) if archive is not None else paths.archive()
    results_dir = Path(results_dir) if results_dir is not None \
        else paths.results() / "ml"

    run_name = "GCPAL" + (f"_{run_tag}" if run_tag else "")
    models_dir = checkpoint_dir(archive, dataset, run_name)
    probs_dir = paths.ensure(
        legacy_path(archive, "member_probs", dataset,
                    member=member_dir(MEMBER), seed=0).parent)

    print(f"\n{'='*60}")
    print(f"  {MEMBER} temporal inference --- {dataset}")
    print(f"  Checkpoints : {models_dir}")
    print(f"  Probabilities: {probs_dir}")
    print(f"{'='*60}")

    # ── 1. Transaction graph ──────────────────────────────────
    print(f"\n  Loading {dataset} ...")
    graph_data = build_pyg_link_data(dataset, data_path=data_path,
                                     num_threads=workers)
    te_data = graph_data["te_data"]
    t1, t2 = graph_data["t1"], graph_data["t2"]
    n_total = graph_data["n_total"]
    edge_typologies = graph_data.get("edge_typologies", [None] * n_total)

    # ── 2. GFP features ───────────────────────────────────────
    gfp_result = load_snapml_features(dataset, data_path=data_path,
                                      num_threads=workers)
    X_all = np.concatenate(
        [gfp_result["X_train"], gfp_result["X_val"], gfp_result["X_test"]],
        axis=0)
    del gfp_result

    # ── 3. Temporal split ─────────────────────────────────────
    train_idx = np.arange(t1)
    val_idx = np.arange(t1, t2)
    test_idx = np.arange(t2, n_total)
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

    # Temporal train statistics, the same convention as LightGBM and XGBoost.
    gfp_mean = X_all[train_idx].mean(axis=0)
    gfp_std = X_all[train_idx].std(axis=0) + 1e-8
    X_all_norm = np.clip((X_all - gfp_mean) / gfp_std, -10, 10)
    del X_all

    # ── 4. Line graph, identical to the training run ──────────
    print(f"\n  Building line graph (lg_k={lg_k}) ...")
    t0 = time.time()
    lg_edge_index, lg_edge_attr = build_line_graph(
        te_data.edge_index, k_neighbors=lg_k, return_edge_attr=True)
    print(f"  Line graph: {n_total:,} nodes, {lg_edge_index.shape[1]:,} edges "
          f"[{time.time()-t0:.1f}s]")

    print("  Pre-aggregating edge features into node features ...")
    extra_node_feats = aggregate_edge_features(lg_edge_index, lg_edge_attr,
                                               n_total)
    del lg_edge_attr

    extra_np = extra_node_feats.numpy()
    extra_mean = extra_np[train_idx].mean(axis=0)
    extra_std = extra_np[train_idx].std(axis=0) + 1e-8
    extra_norm = np.clip((extra_np - extra_mean) / extra_std, -10, 10)
    X_all_norm = np.concatenate([X_all_norm, extra_norm], axis=1)
    feat_dim = X_all_norm.shape[1]
    del extra_node_feats, extra_np, extra_norm
    print(f"  Total node features: {feat_dim} dims (GFP + 5 edge-derived)")

    # ── 5. Labels ─────────────────────────────────────────────
    labels_all = te_data.y.numpy()
    y_test = labels_all[test_idx]
    y_val = labels_all[val_idx]
    print("\n  Temporal split:")
    print(f"    Train: {n_train:,} ({labels_all[train_idx].sum():,} illicit)")
    print(f"    Val:   {n_val:,} ({y_val.sum():,} illicit)")
    print(f"    Test:  {n_test:,} ({y_test.sum():,} illicit)")

    # ── 6. Typology head, on the temporal train and validation illicit edges ──
    # Fitted on the 84-dimensional node features rather than on the raw GFP
    # matrix the training run used, because that is the matrix this stage has in
    # hand. It is a separate model from the one gcpal.py fits and it is the one
    # behind the member's typology column.
    typ_model = None
    X_typ_pool = X_all_norm[:t2]
    combined_typ = np.full(t2, -1, dtype=int)
    for i in range(t2):
        typ = edge_typologies[i]
        if labels_all[i] == 1 and typ and typ in TYPOLOGY_TO_IDX:
            combined_typ[i] = TYPOLOGY_TO_IDX[typ]
    typ_mask = combined_typ >= 0
    if typ_mask.sum() >= 10:
        typ_model = lgb.LGBMClassifier(
            objective="multiclass", num_class=len(TYPOLOGY_CLASSES),
            metric="multi_logloss", verbose=-1,
            n_estimators=50, class_weight="balanced",
        )
        typ_model.fit(X_typ_pool[typ_mask], combined_typ[typ_mask])
        print(f"  Typology classifier trained on {typ_mask.sum()} illicit")
    del X_typ_pool

    # ── 7. Graph on device ────────────────────────────────────
    node_features = torch.tensor(X_all_norm, dtype=torch.float, device=device)
    lg_edge_index = lg_edge_index.to(device)
    labels_dev = torch.tensor(labels_all, dtype=torch.long, device=device)

    lg_data = Data(x=node_features, edge_index=lg_edge_index, y=labels_dev,
                   num_nodes=n_total)

    test_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    test_mask[torch.from_numpy(test_idx).long()] = True
    val_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    val_mask[torch.from_numpy(val_idx).long()] = True

    # ── 8. One forward pass per seed ──────────────────────────
    print(f"\n  Inference ({len(seeds)} seeds) ...")
    X_test_feat = X_all_norm[test_idx]
    all_results = []

    for seed in seeds:
        ck_path = models_dir / f"finetuned_seed_{seed}.pt"
        if not ck_path.exists():
            print(f"  Checkpoint not found: {ck_path}, skipping")
            continue

        ck = torch.load(ck_path, map_location=device)
        saved_feat_dim = ck["feat_dim"]
        saved_hidden = ck["hidden_dim"]
        saved_layers = ck["num_layers"]

        if saved_feat_dim != feat_dim:
            print(f"  seed={seed}: feat_dim mismatch "
                  f"(checkpoint {saved_feat_dim} against current {feat_dim}), "
                  "skipping")
            continue

        model = build_gcpal(feat_dim=feat_dim, hidden_dim=saved_hidden,
                            num_layers=saved_layers, dropout=0.1).to(device)
        model.load_state_dict({k: v.to(device)
                               for k, v in ck["model_state_dict"].items()})
        model.eval()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(lg_data.x, lg_data.edge_index)
            logits = model.classify(h, lg_data.x)
            te_probs = (F.softmax(logits[test_mask], dim=-1)[:, 1]
                        .cpu().float().numpy())
            val_probs = (F.softmax(logits[val_mask], dim=-1)[:, 1]
                         .cpu().float().numpy())

        del model, ck, h, logits
        gc.collect()
        torch.cuda.empty_cache()

        # Threshold retuned on the temporal validation split.
        best_t, best_val_f1 = find_optimal_threshold(y_val, val_probs)
        oracle_t, oracle_f1 = find_optimal_threshold(y_test, te_probs)
        te_preds = (te_probs >= best_t).astype(int)
        f1 = f1_score(y_test, te_preds, zero_division=0)
        prec = precision_score(y_test, te_preds, zero_division=0)
        rec = recall_score(y_test, te_preds, zero_division=0)
        print(f"    [seed={seed}] F1={f1:.4f}@{best_t:.2f} "
              f"(P={prec:.4f}, R={rec:.4f}) "
              f"oracle={oracle_f1:.4f}@{oracle_t:.2f}")

        np.save(legacy_path(archive, "member_probs", dataset,
                            member=member_dir(MEMBER), seed=seed), te_probs)
        if typ_model is not None:
            typ_preds = typ_model.predict(X_test_feat).astype(int)
            np.save(legacy_path(archive, "member_typology", dataset,
                                member=member_dir(MEMBER), seed=seed), typ_preds)

        all_results.append({
            "method": MEMBER, "dataset": dataset, "seed": seed,
            "threshold": float(best_t), "val_f1": float(best_val_f1),
            "detection_f1": float(f1), "detection_precision": float(prec),
            "detection_recall": float(rec),
            "oracle_threshold": float(oracle_t), "oracle_f1": float(oracle_f1),
        })

    if all_results:
        f1s = [r["detection_f1"] for r in all_results]
        ofs = [r["oracle_f1"] for r in all_results]
        print(f"\n  {MEMBER}|{dataset} Summary:")
        print(f"    F1 (val-t):  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"    Oracle F1:   {np.mean(ofs):.4f} +/- {np.std(ofs):.4f}")
        out_dir = paths.ensure(results_dir)
        out_path = out_dir / f"gcpal_gfp_temporal_metrics_{dataset}.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"    Saved: {out_path}")

    del node_features, lg_edge_index, labels_dev, lg_data
    gc.collect()
    torch.cuda.empty_cache()
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description=f"{MEMBER}: re-infer the GCPAL checkpoints on the temporal "
                    "test split")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--lg-k", type=int, default=DEFAULT_LG_K,
                        help="Line-graph neighbours; must match the run that "
                             "produced the checkpoints")
    parser.add_argument("--run-tag", type=str, default=DEFAULT_RUN_TAG,
                        help="Suffix of the training run's checkpoint directory")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--archive", type=str, default=None)
    parser.add_argument("--results", type=str, default=None)
    args = parser.parse_args()

    for dataset in args.datasets:
        run_inference(dataset, seeds=args.seeds, lg_k=args.lg_k,
                      workers=args.workers, run_tag=args.run_tag,
                      data_path=args.data_path, archive=args.archive,
                      results_dir=args.results)

    print("\n" + "=" * 60)
    print("  All datasets done. Combine the members with:")
    print("  python -m amlc.baselines.ml.ensemble")
    print("=" * 60)


if __name__ == "__main__":
    main()
