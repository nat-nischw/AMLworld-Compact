"""Graph Feature Preprocessor features for the supervised baselines.

Wraps IBM Snap ML's Graph Feature Preprocessor in the configuration AMLworld's
own paper reports (Altman et al., 2023, Appendix D) and turns a raw AMLworld
CSV into the feature matrix every table's LightGBM+GFP, XGBoost+GFP and
GCPAL+GFP rows were trained on.

Reads ``<data>/<dataset>_Trans.csv`` and ``<dataset>_Patterns.txt``. Writes one
cache file, ``<data>/gfp_cache/<dataset>_gfp_b<batch>.npy``, because the
extraction is the expensive part of a rerun and its result is deterministic.
The cache key carries the batch size, since two batch sizes give different
features.

Streaming order and batch size
------------------------------
The preprocessor is a streaming engine: it keeps an in-memory graph and each
call adds edges to it, so an edge's features describe the graph as it stood
when that edge arrived. Feeding all five million edges in one
``fit_transform`` builds the whole graph first and then labels every edge with
statistics computed from later rows. The GFP paper (arXiv:2402.08593, Table 4)
uses 128 for the AML datasets, and that is the default here. Smaller batches
limit this exposure but do not establish chronological causality: the
published loader supplies CSV row order, whose timestamps are not sorted,
and a batch can include multiple times. ``batch_size <= 0`` restores the
historical single-pass behaviour for reproducing that configuration.

Not ported
----------
The 39-feature NetworkX extractor that used to sit in the same pre-release
module. It fed the PNA baseline, which appears in no table in the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .. import paths
from . import splits
from .download_amlworld import patterns_txt, trans_csv

#: Subdirectory of the data directory holding cached feature matrices.
GFP_CACHE_DIRNAME = "gfp_cache"

#: Streaming batch size for the paper's runs. See the module docstring.
DEFAULT_GFP_BATCH_SIZE = 128


class SnapMLGFP:
    """Snap ML Graph Feature Preprocessor in AMLworld's Appendix D setting.

    Processes edges in supplied row order and streaming batches. This class
    does not sort timestamps or guarantee strictly past-only features.
    """

    AMLWORLD_PARAMS = {
        "num_threads": 4,
        "time_window": 86400,                           # 1 day (seconds)
        # Vertex statistics on Timestamp (col 3) and Amount (col 4)
        "vertex_stats": True,
        "vertex_stats_cols": [3, 4],
        "vertex_stats_feats": [0, 1, 2, 3, 4, 8, 9, 10],
        # Fan pattern, 1-day window
        "fan": True,
        "fan_tw": 86400,
        "fan_bins": [2, 3],
        # Degree pattern, 1-day window
        "degree": True,
        "degree_tw": 86400,
        "degree_bins": [2, 3],
        # Scatter-gather, 6-hour window (the paper's key setting)
        "scatter-gather": True,
        "scatter-gather_tw": 21600,
        "scatter-gather_bins": [2, 3],
        # Temporal cycles, 1-day window
        "temp-cycle": True,
        "temp-cycle_tw": 86400,
        "temp-cycle_bins": [2, 3],
        # Length-constrained simple cycles, max length 10
        "lc-cycle": True,
        "lc-cycle_tw": 86400,
        "lc-cycle_len": 10,
        "lc-cycle_bins": [2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    N_RAW_COLS = 5  # edge_id, src, dst, timestamp, amount

    def __init__(self, num_threads: int = 4):
        from snapml import GraphFeaturePreprocessor as _SnapGFP

        self._gfp = _SnapGFP()
        params = dict(self.AMLWORLD_PARAMS)
        params["num_threads"] = num_threads
        self._gfp.set_params(params)

    def fit_transform(
        self,
        X: np.ndarray,
        batch_size: int = DEFAULT_GFP_BATCH_SIZE,
    ) -> np.ndarray:
        """Process all edges through the preprocessor in supplied row order.

        Args:
            X: float64 array with columns
               [edge_id, src_id, dst_id, timestamp_epoch, amount].
               The published loader supplies file order, not timestamp order.
            batch_size: edges per streaming batch. The first batch calls
               fit_transform, which resets the in-memory graph; the rest call
               transform, which appends to it. A value <= 0 processes every
               edge in one call, allowing later rows to affect earlier
               features. Smaller batches alone do not guarantee chronological
               causality when timestamps are unsorted or vary within a batch.

        Returns:
            Feature matrix, engineered columns only, raw columns stripped.
        """
        X_f64 = X.astype(np.float64)

        if batch_size <= 0 or batch_size >= len(X_f64):
            enriched = self._gfp.fit_transform(X_f64)
            return enriched[:, self.N_RAW_COLS:]

        n = len(X_f64)
        results = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = X_f64[start:end]
            if start == 0:
                enriched = self._gfp.fit_transform(batch)
            else:
                enriched = self._gfp.transform(batch)
            results.append(enriched[:, self.N_RAW_COLS:])
            if (start // batch_size) % 5000 == 0 and start > 0:
                print(f"    GFP batch {start // batch_size}/"
                      f"{(n + batch_size - 1) // batch_size} "
                      f"({start:,}/{n:,} edges)")
        return np.vstack(results)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Add edges to the existing in-memory graph and compute features."""
        enriched = self._gfp.transform(X.astype(np.float64))
        return enriched[:, self.N_RAW_COLS:]


def load_amlworld_for_snapml(
    variant: str,
    data_path: Optional[Path] = None,
    num_threads: int = 4,
    train_ratio: float = splits.TRAIN_RATIO,
    val_ratio: float = splits.VAL_RATIO,
    gfp_batch_size: int = DEFAULT_GFP_BATCH_SIZE,
) -> Dict:
    """Load an AMLworld CSV, run GFP, and cut the published file-order split.

    This loader does not timestamp-sort rows. Its partitions overlap in
    transaction time and do not establish a strictly temporal evaluation.

    Returns a dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test   numpy arrays
        edge_typologies  list[str | None] for *all* edges (len == n_total)
        t1_idx, t2_idx   file-order split boundaries
        n_total, n_features
        gfp_time         wall-clock seconds for the feature extraction
    """
    import time as _time

    import polars as pl

    data_path = Path(data_path) if data_path is not None else paths.data()
    csv_path = trans_csv(variant, data_path)
    patterns_path = patterns_txt(variant, data_path)

    # ── 1. Load CSV (Polars, much faster than Pandas for 5M rows) ──
    print(f"  Loading {csv_path.name} ...")
    t_load = _time.time()
    col_names = [
        "Timestamp", "From_Bank", "From_Account",
        "To_Bank", "To_Account",
        "Amount_Received", "Receiving_Currency",
        "Amount_Paid", "Payment_Currency",
        "Payment_Format", "Is_Laundering",
    ]
    df = pl.read_csv(
        csv_path, has_header=True, new_columns=col_names,
        # Override uses ORIGINAL header names (before rename)
        schema_overrides={"From Bank": pl.Utf8, "To Bank": pl.Utf8},
    )
    # Strip whitespace from all string columns
    str_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    df = df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])

    n_total = len(df)
    n_illicit = int(df["Is_Laundering"].sum())
    print(f"  {n_total:,} txns  ({n_illicit:,} illicit, "
          f"{n_illicit / n_total * 100:.2f}%)  [{_time.time() - t_load:.1f}s]")

    # ── 2. Temporal split boundaries ─────────────────────────
    t1_idx, t2_idx = splits.temporal_boundaries(n_total, train_ratio, val_ratio)
    print("  Temporal split: " + splits.describe(n_total, t1_idx, t2_idx))

    # ── 3. Parse Patterns.txt for typology labels (Polars join) ──
    edge_typologies: List[Optional[str]] = [None] * n_total
    if patterns_path.exists():
        from .loader import AMLworldDataset

        typology_map = AMLworldDataset._parse_patterns(patterns_path)
        if typology_map:
            match_keys = (
                df["Timestamp"] + "," + df["From_Bank"] + "," +
                df["From_Account"] + "," + df["To_Bank"] + "," +
                df["To_Account"]
            )
            typ_df = pl.DataFrame({
                "key": list(typology_map.keys()),
                "typology": list(typology_map.values()),
            })
            matched = match_keys.to_frame("key").join(typ_df, on="key", how="left")
            edge_typologies = matched["typology"].to_list()
            n_matched = matched["typology"].is_not_null().sum()
            print(f"  Typology matched: {n_matched:,}/{n_illicit:,} illicit edges")

    # ── 4. Build the preprocessor input (vectorised, no Python loops) ──
    src_ser = df["From_Bank"] + "_" + df["From_Account"]
    dst_ser = df["To_Bank"] + "_" + df["To_Account"]

    # Node mapping via Polars categorical encoding (no Python dict/loop)
    all_nodes = pl.concat([
        src_ser.rename("node"), dst_ser.rename("node")
    ]).unique().sort()
    node_lut = all_nodes.to_frame().with_row_index("nid")
    print(f"  {len(node_lut):,} unique nodes")

    src_ids = (src_ser.to_frame("node")
               .join(node_lut, on="node", how="left")["nid"]
               .to_numpy().astype(np.float64))
    dst_ids = (dst_ser.to_frame("node")
               .join(node_lut, on="node", how="left")["nid"]
               .to_numpy().astype(np.float64))

    ts_col = df["Timestamp"].str.to_datetime("%Y/%m/%d %H:%M")
    ts_epoch = (ts_col.cast(pl.Int64) // 1_000_000).to_numpy().astype(np.float64)

    edge_ids = np.arange(n_total, dtype=np.float64)
    amounts_paid = df["Amount_Paid"].to_numpy().astype(np.float64)
    amounts_recv = df["Amount_Received"].to_numpy().astype(np.float64)

    # Input: [edge_id, src, dst, timestamp, amount]
    # Use Amount_Received (actual transferred amount) for the vertex statistics
    X_raw = np.column_stack([edge_ids, src_ids, dst_ids, ts_epoch, amounts_recv])

    # ── 4b. Extra edge attributes (appended after the graph features) ──
    # Minimal set: avoid high-cardinality categoricals (bank codes) and
    # redundant temporals (hour/dow already in the vertex statistics).
    # Matches IBM Multi-GNN edge feature spirit: amount, currency, format.
    pf_codes = (df["Payment_Format"].cast(pl.Categorical)
                .to_physical().to_numpy().astype(np.float64))
    pay_cur_codes = (df["Payment_Currency"].cast(pl.Categorical)
                     .to_physical().to_numpy().astype(np.float64))
    recv_cur_codes = (df["Receiving_Currency"].cast(pl.Categorical)
                      .to_physical().to_numpy().astype(np.float64))
    cross_currency = ((df["Payment_Currency"] != df["Receiving_Currency"])
                      .to_numpy().astype(np.float64))

    y = df["Is_Laundering"].to_numpy()
    del df  # free memory

    # ── 5. Run the preprocessor (with disk cache) ────────────
    cache_dir = data_path / GFP_CACHE_DIRNAME
    # Cache key includes batch_size: batch=128 and single-pass differ.
    bs_tag = f"_b{gfp_batch_size}" if gfp_batch_size > 0 else ""
    cache_path = cache_dir / f"{variant}_gfp{bs_tag}.npy"

    if cache_path.exists():
        print(f"  Loading cached GFP features from {cache_path} ...")
        t0 = _time.time()
        X_gfp = np.load(cache_path)
        gfp_time = _time.time() - t0
        print(f"  GFP loaded: {X_gfp.shape[1]} features in {gfp_time:.1f}s (cached)")
        del X_raw
    else:
        bs_str = str(gfp_batch_size) if gfp_batch_size > 0 else "single pass"
        print(f"  Extracting features with Snap ML GFP "
              f"(threads: {num_threads}, batch_size: {bs_str}) ...")
        gfp = SnapMLGFP(num_threads=num_threads)
        t0 = _time.time()
        X_gfp = gfp.fit_transform(X_raw, batch_size=gfp_batch_size)
        gfp_time = _time.time() - t0
        print(f"  GFP done: {X_gfp.shape[1]} graph features in {gfp_time:.1f}s")
        del X_raw
        paths.ensure(cache_dir)
        np.save(cache_path, X_gfp)
        print(f"  GFP cached to {cache_path}")

    # Graph features plus the minimal raw edge attributes (6 features)
    X_features = np.column_stack([
        X_gfp,
        amounts_paid, amounts_recv,
        pay_cur_codes.reshape(-1, 1), recv_cur_codes.reshape(-1, 1),
        cross_currency,
        pf_codes.reshape(-1, 1),
    ])
    n_extra = X_features.shape[1] - X_gfp.shape[1]
    print(f"  GFP: {X_gfp.shape[1]} + {n_extra} raw edge attrs = "
          f"{X_features.shape[1]} total features")

    # ── 6. Split by published file-order boundaries ──────────
    X_train, X_val, X_test = splits.split_three(X_features, t1_idx, t2_idx)
    y_train, y_val, y_test = splits.split_three(y, t1_idx, t2_idx)

    tr_ill = int(y_train.sum())
    va_ill = int(y_val.sum())
    te_ill = int(y_test.sum())
    print(f"  Train: {len(y_train):,} edges ({tr_ill:,} illicit)")
    print(f"  Val:   {len(y_val):,} edges ({va_ill:,} illicit)")
    print(f"  Test:  {len(y_test):,} edges ({te_ill:,} illicit)")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "edge_typologies": edge_typologies,
        "t1_idx": t1_idx,
        "t2_idx": t2_idx,
        "n_total": n_total,
        "n_features": X_features.shape[1],
        "gfp_time": gfp_time,
    }
