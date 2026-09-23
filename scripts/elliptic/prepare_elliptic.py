#!/usr/bin/env python3
"""Prepare the Elliptic Bitcoin Dataset for downstream training/eval.

Reads the three raw CSVs and writes:

    data/elliptic/proc/
        train.npz   features X (n_train, 165), labels y, txid, time_step
        test.npz    features X (n_test , 165), labels y, txid, time_step
        edges.npz   src, dst (parallel int32 arrays of edge endpoints,
                              both reindexed into the union [train+test+unknown]
                              integer-id space)
        node_index.npz  txid → row id mapping for the full 203k node set
        meta.json   counts, time-step ranges, label encoding

Conventions:
    label = 1  illicit
    label = 0  licit
    label =-1  unknown (kept in `edges.npz` for k-hop context but excluded
                       from train/test feature arrays)

Temporal split (Weber et al. 2019, Table 1):
    train: time_step ∈ [1, 34]
    test : time_step ∈ [35, 49]

We *retain unknowns* in the edge index because they participate in k-hop
context expansion later (subgraph serialization for LLM eval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "data" / "elliptic" / "raw"
PROC_DIR  = REPO_ROOT / "data" / "elliptic" / "proc"

CLASS_TO_LABEL = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}

DEFAULT_SPLIT = (1, 34, 49)   # train_end=34, test_end=49


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats_path  = raw_dir / "elliptic_txs_features.csv"
    cls_path    = raw_dir / "elliptic_txs_classes.csv"
    edges_path  = raw_dir / "elliptic_txs_edgelist.csv"
    if not feats_path.exists():
        raise FileNotFoundError(
            f"missing {feats_path}; run download_elliptic.py first")

    feats = pd.read_csv(feats_path, header=None)
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
    cls   = pd.read_csv(cls_path)
    cls.columns = ["txId", "class"]
    edges = pd.read_csv(edges_path)
    edges.columns = ["src_txId", "dst_txId"]
    return feats, cls, edges


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir",  type=Path, default=RAW_DIR)
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--train-end", type=int, default=DEFAULT_SPLIT[1],
                    help=f"last time-step in train split (default {DEFAULT_SPLIT[1]})")
    ap.add_argument("--test-end",  type=int, default=DEFAULT_SPLIT[2],
                    help=f"last time-step in test  split (default {DEFAULT_SPLIT[2]})")
    args = ap.parse_args()

    args.proc_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] reading {args.raw_dir}")
    feats, cls, edges = load_raw(args.raw_dir)
    print(f"  features: {feats.shape}  classes: {cls.shape}  edges: {edges.shape}")

    df = feats.merge(cls, on="txId", how="left")
    df["class"] = df["class"].fillna("unknown")
    df["label"] = df["class"].map(CLASS_TO_LABEL).astype(np.int8)

    n_total = len(df)
    n_lab = int((df["label"] >= 0).sum())
    n_ill = int((df["label"] == 1).sum())
    n_lic = int((df["label"] == 0).sum())
    n_unk = int((df["label"] == -1).sum())
    print(f"  labelled: {n_lab:,} ({n_ill:,} illicit + {n_lic:,} licit); unknown: {n_unk:,}")

    # Reindex txId → row id so edges become integer arrays
    df = df.sort_values(["time_step", "txId"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int32)
    txid_to_row = dict(zip(df["txId"].values, df["row_id"].values))

    src = edges["src_txId"].map(txid_to_row).to_numpy()
    dst = edges["dst_txId"].map(txid_to_row).to_numpy()
    valid_mask = (~np.isnan(src)) & (~np.isnan(dst))
    src = src[valid_mask].astype(np.int32)
    dst = dst[valid_mask].astype(np.int32)
    print(f"  edges (valid): {len(src):,}")

    # Split
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int8)
    t = df["time_step"].to_numpy(dtype=np.int16)
    txid = df["txId"].to_numpy(dtype=np.int64)

    train_mask = (t <= args.train_end) & (y >= 0)
    test_mask  = (t >  args.train_end) & (t <= args.test_end) & (y >= 0)

    print(f"  train: {int(train_mask.sum()):,} (illicit {int((y[train_mask]==1).sum()):,})")
    print(f"  test : {int(test_mask.sum()):,} (illicit {int((y[test_mask]==1).sum()):,})")

    # Save
    np.savez(args.proc_dir / "train.npz",
             X=X[train_mask], y=y[train_mask],
             txid=txid[train_mask], time_step=t[train_mask],
             row_id=df["row_id"].to_numpy()[train_mask].astype(np.int32))
    np.savez(args.proc_dir / "test.npz",
             X=X[test_mask], y=y[test_mask],
             txid=txid[test_mask], time_step=t[test_mask],
             row_id=df["row_id"].to_numpy()[test_mask].astype(np.int32))
    np.savez(args.proc_dir / "edges.npz", src=src, dst=dst)
    np.savez(args.proc_dir / "node_index.npz",
             txid=txid, time_step=t, label=y)

    meta = dict(
        n_total=int(n_total),
        n_labelled=n_lab,
        n_illicit=n_ill,
        n_licit=n_lic,
        n_unknown=n_unk,
        n_features=len(feat_cols),
        n_edges=int(len(src)),
        train_time_steps=[1, int(args.train_end)],
        test_time_steps=[int(args.train_end + 1), int(args.test_end)],
        train_size=int(train_mask.sum()),
        test_size=int(test_mask.sum()),
        train_illicit=int((y[train_mask] == 1).sum()),
        test_illicit=int((y[test_mask]  == 1).sum()),
        illicit_rate_train=float((y[train_mask] == 1).mean()) if train_mask.any() else 0.0,
        illicit_rate_test=float((y[test_mask]  == 1).mean()) if test_mask.any() else 0.0,
        label_encoding={"illicit": 1, "licit": 0, "unknown": -1},
    )
    (args.proc_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[prepare] wrote {args.proc_dir}")


if __name__ == "__main__":
    main()
