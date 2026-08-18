"""LightGBM+GFP and XGBoost+GFP: two of the three ensemble members.

Each baseline is two models over the same feature matrix. Stage 1 is a binary
detector over every edge. Stage 2 is an eight-class typology head fitted only on
the illicit edges that carry a typology label, and consulted only where stage 1
fires. The features are IBM Snap ML Graph-Feature-Preprocessor output, 73 graph
features plus 6 raw edge attributes, extracted once for the whole dataset by the
data-preparation stage and sliced by temporal split.

Reads
    the Snap ML feature matrices and labels for one AMLworld dataset, and the
    Optuna result under ``data/tuned_params/<dataset>/`` when the caller passes
    it in.
Writes
    nothing. The trainers return fitted models and arrays; the orchestrator
    decides where per-seed probabilities and typology predictions land. Those
    are what the ensemble and the coreset read back.

``build_pyg_link_data`` also lives here, because it is where the pre-release
tree kept it and because both GCPAL stages import it. It builds one transaction
graph for the whole dataset rather than millions of k-hop subgraphs, following
IBM Multi-GNN.

Dropped from the pre-release ``non_llm_baselines.py``
-----------------------------------------------------
**PNA.** The third baseline in that file appears in no table in the paper. Its
class, its degree-histogram helper, its PyG conversion helpers and its factory
entry are gone.

**The four objects PNA was the last consumer of inside ``build_pyg_link_data``:**
the train-only and validation graphs, the four raw z-normalised edge features
(timestamp, amount, currency, payment format) that were attached as
``edge_attr``, and the ``edge_dim`` they implied. Both GCPAL stages read only
``te_data.edge_index``, ``te_data.y``, the split indices and the GFP matrix, so
the output they consume is unchanged and three full copies of the edge tensors
no longer get built.

**The case-based API.** ``NonLLMBaseline``, ``fit``, ``predict``,
``predict_batch`` and ``predict_proba`` over ``Case`` objects were reached only
in synthetic mode and on the PNA path; every published GBT number came through
``fit_from_features``. Dropping them also drops this module's dependency on the
``Case`` dataclass and the per-case feature preprocessor.

**The sklearn GradientBoosting fallback.** Both trainers used to catch a missing
lightgbm or xgboost and silently substitute ``GradientBoostingClassifier``. That
is a different model with different numbers under the same method name, which is
the worst possible failure mode for a reproduction. Missing dependencies now
raise.

**``predict_batch_from_features``.** It wrapped the probability vector in the
LLM path's ``PredictionResult`` dataclass so a shared metrics function could
score it against ``Case`` objects. Both of those belong to the LLM path.
:meth:`GFPBaseline.predict_proba_from_features` and
:meth:`GFPBaseline.predict_typology_from_features` return the two arrays that
wrapper was built out of, which are also exactly what the orchestrator saved to
disk and what every downstream stage reads.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ... import paths
from ...typology import TYPOLOGY_CLASSES

# Snap ML hands LightGBM a bare ndarray, and sklearn warns once per predict call.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


#: Raised when the shared AMLworld reader or feature extractor is not available.
_UPSTREAM_MSG = (
    "This stage needs the AMLworld reader and the Snap ML feature extractor, "
    "which are shared with the serialisation stage and live outside "
    "amlc.baselines. Install the 'ml' extra and make sure that stage is "
    "present, or use the released coreset under data/ and the result tables "
    "under results/, neither of which needs AMLworld."
)


def _snapml_gfp():
    """The Snap ML Graph-Feature-Preprocessor wrapper, imported at call time."""
    try:
        from ...features import SnapMLGFP
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(_UPSTREAM_MSG) from exc
    return SnapMLGFP


def _pattern_parser():
    """The ``<dataset>_Patterns.txt`` reader, imported at call time."""
    try:
        from ...data_loader import AMLworldDataset
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(_UPSTREAM_MSG) from exc
    return AMLworldDataset._parse_patterns


def load_snapml_features(dataset: str, data_path=None, num_threads: int = 8) -> Dict:
    """Snap ML GFP features and labels for one dataset, split 60/20/20 in time.

    A pass-through to the shared extractor, kept here so the cross-stage import
    and its error message live in one place. Returns that function's dict:
    ``X_train``/``y_train``, ``X_val``/``y_val``, ``X_test``/``y_test``, the
    split points ``t1_idx`` and ``t2_idx``, and ``edge_typologies`` for every
    edge in the dataset.
    """
    try:
        from ...features import load_amlworld_for_snapml
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(_UPSTREAM_MSG) from exc
    if data_path is None:
        data_path = paths.data()
    return load_amlworld_for_snapml(
        variant=dataset, data_path=Path(data_path), num_threads=num_threads,
    )


# =============================================================================
# Shared state
# =============================================================================

class GFPBaseline:
    """A detector plus a typology head, both over Snap ML GFP features.

    Subclasses own the library-specific training and probability calls. What is
    common is the pair of models, the native-booster flag that says which
    prediction API to use, and the typology head, which is an sklearn-API
    multiclass classifier in both subclasses.
    """

    def __init__(self):
        self.name = "GFPBaseline"
        self.model = None            # binary detection model
        self.typology_model = None   # eight-class typology model
        self.is_fitted = False
        self._native_booster = False

    def predict_proba_from_features(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_typology_from_features(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Typology class index per row, or None when no head was fitted.

        Integers are indices into :data:`amlc.typology.TYPOLOGY_CLASSES`.
        This is the array the pipeline stores as ``seed_<seed>_typ.npy``.
        """
        if self.typology_model is None:
            return None
        return self.typology_model.predict(X).astype(int)


def _typology_pool(X_train: np.ndarray, X_val: Optional[np.ndarray],
                   typology_labels: np.ndarray) -> np.ndarray:
    """Feature rows the typology head is fitted on.

    The caller may pass typology labels covering train alone or train and
    validation concatenated; the second is what the pipeline does, to give the
    head more illicit rows. Which one it is can only be told from the length.
    """
    if X_val is not None and len(typology_labels) == len(X_train) + len(X_val):
        return np.vstack([X_train, X_val])
    return X_train


# =============================================================================
# LightGBM + GFP
# =============================================================================

class LightGBMGFP(GFPBaseline):
    """LightGBM over Graph-Feature-Preprocessor features.

    The defaults below are the non-tunable essentials only. Everything the
    Optuna search selects, ``scale_pos_weight`` included, arrives through
    ``**lgb_params`` and overrides them.
    """

    def __init__(self, **lgb_params):
        super().__init__()
        self.name = "LightGBM+GFP"
        self.lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.05,
            **lgb_params,
        }

    def fit_from_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        typology_labels: np.ndarray = None,
        typology_params: Optional[Dict] = None,
        num_threads: int = None,
        seed: int = None,
    ):
        """Train detection and typology from pre-computed feature matrices.

        Uses the native ``lgb.train`` API rather than the sklearn wrapper, which
        is what the Optuna search used and is substantially faster here.

        Parameters
        ----------
        typology_labels : ndarray
            One entry per row, -1 for a legitimate or untyped edge and 0..7 for
            a typology class. May span train, or train and validation
            concatenated.
        seed : int
            Seeds the sampler, the bagging and the feature fraction together.
            The five paper seeds differ only through this.
        """
        import lightgbm as lgb

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

        params = {k: v for k, v in self.lgb_params.items() if k != "n_estimators"}
        params["num_threads"] = num_threads or 8
        n_rounds = self.lgb_params.get("n_estimators", 100)

        if seed is not None:
            params["seed"] = seed
            params["bagging_seed"] = seed
            params["feature_fraction_seed"] = seed

        valid_sets = []
        valid_names = []
        callbacks = [lgb.log_evaluation(0)]
        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain,
                               free_raw_data=False)
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(20, verbose=False))

        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._native_booster = True

        if typology_labels is not None:
            X_typ_pool = _typology_pool(X_train, X_val, typology_labels)
            mask = typology_labels >= 0
            if mask.sum() >= 10:
                typ_params = {
                    "objective": "multiclass",
                    "num_class": len(TYPOLOGY_CLASSES),
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 15,
                    "learning_rate": 0.05,
                    "verbose": -1,
                    "n_estimators": 50,
                    "class_weight": "balanced",
                }
                if typology_params:
                    typ_params.update(typology_params)
                self.typology_model = lgb.LGBMClassifier(**typ_params)
                self.typology_model.fit(X_typ_pool[mask], typology_labels[mask])

        self.is_fitted = True
        return self

    def predict_proba_from_features(self, X: np.ndarray) -> np.ndarray:
        """Probability of illicit per row."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit_from_features() first.")
        if self._native_booster:
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]


# =============================================================================
# XGBoost + GFP
# =============================================================================

class XGBoostGFP(GFPBaseline):
    """XGBoost over Graph-Feature-Preprocessor features.

    ``device="cuda"`` matches the tuning runs. As with LightGBM, the tuned
    parameters override these defaults.
    """

    def __init__(self, **xgb_params):
        super().__init__()
        self.name = "XGBoost+GFP"
        self.xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "use_label_encoder": False,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            **xgb_params,
        }

    def fit_from_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        typology_labels: np.ndarray = None,
        typology_params: Optional[Dict] = None,
        num_threads: int = None,
        seed: int = None,
    ):
        """Train detection and typology from pre-computed feature matrices.

        Uses the native ``xgb.train`` API, matching the tuning code and the GPU
        path. See :meth:`LightGBMGFP.fit_from_features` for the arguments.
        """
        import xgboost as xgb

        dtrain = xgb.DMatrix(X_train, label=y_train)

        # The native API rejects the two sklearn-wrapper-only keys.
        params = {k: v for k, v in self.xgb_params.items()
                  if k not in ("n_estimators", "use_label_encoder")}
        if num_threads:
            params["nthread"] = num_threads
        if seed is not None:
            params["seed"] = seed
        n_rounds = self.xgb_params.get("n_estimators", 100)

        evals = [(dtrain, "train")]
        es_rounds = None
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))
            es_rounds = 20

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=es_rounds,
            verbose_eval=False,
        )
        self._native_booster = True

        if typology_labels is not None:
            X_typ_pool = _typology_pool(X_train, X_val, typology_labels)
            mask = typology_labels >= 0
            if mask.sum() >= 10:
                typ_params = {
                    "objective": "multi:softprob",
                    "num_class": len(TYPOLOGY_CLASSES),
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "n_estimators": 50,
                    "verbosity": 0,
                }
                if typology_params:
                    typ_params.update(typology_params)
                self.typology_model = xgb.XGBClassifier(**typ_params)
                self.typology_model.fit(X_typ_pool[mask], typology_labels[mask],
                                        verbose=False)

        self.is_fitted = True
        return self

    def predict_proba_from_features(self, X: np.ndarray) -> np.ndarray:
        """Probability of illicit per row."""
        import xgboost as xgb

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit_from_features() first.")
        if self._native_booster:
            return self.model.predict(xgb.DMatrix(X))
        return self.model.predict_proba(X)[:, 1]


# =============================================================================
# Link-level graph builder (IBM Multi-GNN style)
# =============================================================================

def build_pyg_link_data(dataset: str, data_path=None, num_threads: int = 32):
    """Build one PyTorch Geometric graph for a whole AMLworld dataset.

    One graph over every transaction, built in seconds, instead of millions of
    k-hop subgraphs. Following IBM Multi-GNN
    (https://github.com/IBM/Multi-GNN).

    Reads ``<data_path>/<dataset>_Trans.csv`` and, when present,
    ``<dataset>_Patterns.txt`` for the typology of each illicit edge. The Snap
    ML feature matrix is cached at ``<data_path>/gfp_cache/<dataset>_gfp.npy``
    and reused, because extracting it is the expensive part.

    Returns a dict with:

    ==================  =======================================================
    ``te_data``         PyG ``Data`` over all edges: node features, edge index,
                        labels
    ``t1``, ``t2``      temporal split points at 60 and 80 percent
    ``n_nodes``         unique accounts, keyed bank and account together
    ``n_total``         edges
    ``gfp_feat``        (n_total, ~75) z-normalised GFP matrix plus the two
                        amount columns, used as node features by GCPAL and by
                        its typology head
    ``edge_typologies`` per-edge typology name or None, aligned to the CSV
    ==================  =======================================================
    """
    import time as _time

    import polars as pl
    import torch
    from torch_geometric.data import Data

    if data_path is None:
        data_path = paths.data()
    data_path = Path(data_path)

    csv_path = data_path / f"{dataset}_Trans.csv"
    print(f"  Loading {csv_path.name} ...")

    col_names = [
        "Timestamp", "From_Bank", "From_Account",
        "To_Bank", "To_Account",
        "Amount_Received", "Receiving_Currency",
        "Amount_Paid", "Payment_Currency",
        "Payment_Format", "Is_Laundering",
    ]
    df = pl.read_csv(
        csv_path, has_header=True, new_columns=col_names,
        schema_overrides={"From Bank": pl.Utf8, "To Bank": pl.Utf8},
    )
    str_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]
    df = df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])

    n_total = len(df)
    t1 = int(n_total * 0.6)
    t2 = int(n_total * 0.8)
    n_ill = int(df["Is_Laundering"].sum())

    print(f"  {n_total:,} txns ({n_ill:,} illicit, {n_ill/n_total*100:.2f}%)")
    print(f"  Temporal split: train[:{t1:,}] val[{t1:,}:{t2:,}] test[{t2:,}:]")

    # Typology labels from Patterns.txt, matched on the five-column edge key.
    patterns_path = data_path / f"{dataset}_Patterns.txt"
    edge_typologies: List[Optional[str]] = [None] * n_total
    if patterns_path.exists():
        typology_map = _pattern_parser()(patterns_path)
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
            print(f"  Typology matched: {n_matched:,}/{n_ill:,} illicit edges")

    # Node ids, keyed on bank and account together, in sorted order so the
    # mapping is identical to the one the Snap ML loader builds.
    src_ser = df["From_Bank"] + "_" + df["From_Account"]
    dst_ser = df["To_Bank"] + "_" + df["To_Account"]
    all_nodes = pl.concat([
        src_ser.rename("node"), dst_ser.rename("node")
    ]).unique().sort()
    node_lut = all_nodes.to_frame().with_row_index("nid")
    n_nodes = len(node_lut)
    print(f"  {n_nodes:,} unique nodes")

    src_ids = (src_ser.to_frame("node")
               .join(node_lut, on="node", how="left")["nid"]
               .to_numpy().astype(np.int64))
    dst_ids = (dst_ser.to_frame("node")
               .join(node_lut, on="node", how="left")["nid"]
               .to_numpy().astype(np.int64))
    edge_index = torch.from_numpy(np.stack([src_ids, dst_ids]))

    amounts_paid = df["Amount_Paid"].to_numpy().astype(np.float64)
    amounts_recv = df["Amount_Received"].to_numpy().astype(np.float64)

    ts_col = df["Timestamp"].str.to_datetime("%Y/%m/%d %H:%M")
    ts_epoch = (ts_col.cast(pl.Int64) // 1_000_000).to_numpy().astype(np.float64)
    # Zero the clock at the first transaction.
    ts_epoch = ts_epoch - ts_epoch[0]

    def znorm(a):
        return (a - a.mean()) / (a.std() + 1e-8)

    cache_path = data_path / "gfp_cache" / f"{dataset}_gfp.npy"
    if cache_path.exists():
        print(f"  Loading cached GFP features from {cache_path} ...")
        t0 = _time.time()
        X_gfp = np.load(str(cache_path))
        gfp_time = _time.time() - t0
        print(f"  GFP loaded: {X_gfp.shape[1]} features in {gfp_time:.1f}s (cached)")
    else:
        print(f"  Extracting features with Snap ML GFP (threads: {num_threads}) ...")
        edge_ids = np.arange(n_total, dtype=np.float64)
        X_raw = np.column_stack([
            edge_ids, src_ids.astype(np.float64),
            dst_ids.astype(np.float64), ts_epoch + ts_epoch[0], amounts_paid,
        ])
        gfp = _snapml_gfp()(num_threads=num_threads)
        t0 = _time.time()
        X_gfp = gfp.fit_transform(X_raw)
        gfp_time = _time.time() - t0
        print(f"  GFP done: {X_gfp.shape[1]} graph features in {gfp_time:.1f}s")
        del X_raw
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), X_gfp)
        print(f"  GFP cached to {cache_path}")

    gfp_mean = X_gfp.mean(axis=0, keepdims=True)
    gfp_std = X_gfp.std(axis=0, keepdims=True) + 1e-8
    X_gfp_norm = (X_gfp - gfp_mean) / gfp_std
    X_gfp_norm = np.clip(X_gfp_norm, -10, 10)
    gfp_feat = np.column_stack([X_gfp_norm, znorm(amounts_paid), znorm(amounts_recv)])
    del X_gfp, X_gfp_norm
    print(f"  GFP features: {gfp_feat.shape[1]}")

    y = torch.tensor(df["Is_Laundering"].to_numpy(), dtype=torch.long)
    x = torch.ones(n_nodes, 1, dtype=torch.float)
    del df

    te_data = Data(x=x, edge_index=edge_index, y=y)
    te_data.num_nodes = n_nodes

    ill_tr = int(y[:t1].sum())
    ill_val = int(y[t1:t2].sum())
    ill_te = int(y[t2:].sum())
    print(f"  Graph built: {n_nodes:,} nodes")
    print(f"    Train: {t1:,} edges ({ill_tr:,} illicit)")
    print(f"    Val:   {t2-t1:,} edges ({ill_val:,} illicit)")
    print(f"    Test:  {n_total-t2:,} edges ({ill_te:,} illicit)")

    return {
        "te_data": te_data,
        "t1": t1, "t2": t2, "n_nodes": n_nodes, "n_total": n_total,
        "gfp_feat": gfp_feat,
        "edge_typologies": edge_typologies,
    }


# =============================================================================
# Factory
# =============================================================================

#: Paper name -> class. PNA is deliberately absent; see the module docstring.
SUPERVISED_BASELINES = {
    "LightGBM+GFP": LightGBMGFP,
    "XGBoost+GFP": XGBoostGFP,
}


def get_supervised_baseline(method_name: str, **kwargs) -> GFPBaseline:
    """Construct a boosted-tree baseline by its paper name."""
    if method_name not in SUPERVISED_BASELINES:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available: {list(SUPERVISED_BASELINES)}"
        )
    return SUPERVISED_BASELINES[method_name](**kwargs)
