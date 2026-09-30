"""Optuna search for the boosted-tree baselines. Optional: the results are shipped.

``data/tuned_params/<dataset>/`` already holds the outcome of every search
behind the paper, so nothing in the pipeline needs this stage to run. It is here
because a reader should be able to see how those numbers were produced and rerun
them, not because a reproduction depends on it. The LightGBM+GFP search on
HI-Small alone took 28,923 seconds over 50 trials.

Four searches per dataset, all Tree-structured Parzen Estimator over a
train-and-validation holdout rather than cross-validation, which is the protocol
of the AMLworld paper's Appendix C:

    <method>.json            binary detection, objective F1 at a fixed 0.5
    <method>_typology.json   eight-class typology, objective macro F1

Each search uses the file-order 60/20/20 split: fit on the first 60 percent,
score on the next 20, never touch the last 20. The detection searches early-stop
on the validation split and record ``actual_n_rounds``, the round count that
early stopping actually reached, because the final training run uses that rather
than the sampled ``n_estimators``.

The detection objective is F1 at a fixed 0.5 rather than at a swept threshold.
That is deliberate and it is why ``best_threshold`` is written as the constant
0.5: threshold selection happens later, on the validation split, in the training
stage. The two are different decisions and the search does not pre-empt the
second.

Reads
    the Snap ML feature matrices for one dataset.
Writes
    ``data/tuned_params/<dataset>/<method>.json`` and the ``_typology``
    companion. Set ``AMLC_DATA`` to write somewhere else.

Dropped
-------
``tune_pna`` and ``PNA_PARAM_SPACE``
    PNA appears in no table in the paper. Roughly 250 lines, including the
    edge-level PNA model that existed only inside the objective.
``_prepare_data_legacy``
    Case-by-case feature extraction, reached only from the PNA path.
``--dry-run``
    It printed three scipy distribution dictionaries that were never sampled
    from: the search space is declared inline with ``trial.suggest_*``, and the
    dictionaries had drifted away from it, so the mode reported ranges that were
    not the ones being searched. The paper's Table 10 ranges are recorded as
    comments where they are applied.
``--split-mode``
    Only ``_prepare_data_legacy`` read it.
``--factor``
    A successive-halving parameter left over from before the switch to Optuna.
    It was passed to both tuners and neither looked at it.

This stage writes ``data/tuned_params/``. The reader is ``load_tuned`` in
:mod:`amlc.llm.runner`, so there is one writer and one reader.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ... import paths
from ...config import DATASETS
from ...typology import TYPOLOGY_CLASSES, TYPOLOGY_TO_IDX
from .gbt import load_snapml_features

#: CLI name -> the paper's method name, which is also the JSON filename stem.
MODEL_MAP = {
    "lightgbm": "LightGBM+GFP",
    "xgboost": "XGBoost+GFP",
}


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{int(s)}s"


# ═════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════

def _build_typology_labels(edge_typologies: List[Optional[str]], n: int,
                           offset: int = 0) -> np.ndarray:
    """Encode a slice of typology names: -1 untyped, otherwise the class index."""
    labels = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        typ = edge_typologies[offset + i]
        if typ is not None and typ in TYPOLOGY_TO_IDX:
            labels[i] = TYPOLOGY_TO_IDX[typ]
    return labels


def prepare_data(dataset: str, data_path: Optional[Path] = None,
                 n_workers: int = 1) -> Dict:
    """Snap ML features for the train and validation windows, plus typologies.

    Returns ``X_train``/``y_train``, ``X_val``/``y_val`` and the matching
    ``typ_train``/``typ_val`` integer arrays. The test window is loaded by the
    extractor but is not returned, so it cannot be reached from here.
    """
    print(f"\n{'='*60}")
    print(f"  Loading {dataset} (Snap ML GFP)")
    print(f"{'='*60}")

    result = load_snapml_features(dataset, data_path=data_path,
                                  num_threads=n_workers)

    t1 = result["t1_idx"]
    t2 = result["t2_idx"]
    typ_train = _build_typology_labels(result["edge_typologies"], t1, offset=0)
    typ_val = _build_typology_labels(result["edge_typologies"], t2 - t1, offset=t1)

    print(f"  Typology labels: train={int((typ_train >= 0).sum())}, "
          f"val={int((typ_val >= 0).sum())}")

    return {
        "X_train": result["X_train"], "y_train": result["y_train"],
        "X_val": result["X_val"], "y_val": result["y_val"],
        "typ_train": typ_train, "typ_val": typ_val,
    }


def _serialize_params(params: Dict) -> Dict:
    """Numpy scalars to Python natives, so the result is JSON-writable."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# ═════════════════════════════════════════════════════════════
# Detection searches
# ═════════════════════════════════════════════════════════════

def tune_lightgbm(X_train, y_train, X_val, y_val, n_candidates=20,
                  random_state=42) -> Dict:
    """TPE search for LightGBM detection. Objective: validation F1 at 0.5."""
    import os

    import lightgbm as lgb
    import optuna
    from sklearn.metrics import f1_score
    from tqdm import tqdm

    print(f"\n  {'-'*50}")
    print("  Tuning LightGBM (Optuna TPE, train/val holdout)")
    print(f"  Trials: {n_candidates}")
    print(f"  Train: {len(y_train):,} edges, Val: {len(y_val):,} edges")
    print(f"  {'-'*50}")

    pbar = tqdm(total=n_candidates, desc="  LightGBM", unit="trial",
                bar_format="  {desc} |{bar:35}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}{postfix}]")

    # LightGBM segfaults under parallel Optuna trials because of OpenMP
    # conflicts, so trials are sequential and each one gets most of the cores.
    total_cores = os.cpu_count() or 32
    threads_per_trial = int(total_cores * 0.8)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "device": "cpu",
            "num_threads": threads_per_trial,
            # AMLworld paper Table 10, the range-large row for the Small datasets.
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 2, 16384, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 10**-2.5, 10**-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 10**-2, 10**2, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 10**0.01, 10**0.5, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

        # A fresh Dataset per trial: lgb.Dataset is not safe to share.
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

        model = lgb.train(params, dtrain, num_boost_round=params["n_estimators"],
                          valid_sets=[dval], valid_names=["val"],
                          callbacks=[lgb.early_stopping(20, verbose=False),
                                     lgb.log_evaluation(0)])

        actual_n_rounds = model.current_iteration()
        val_probs = model.predict(X_val)
        score = f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)

        trial.set_user_attr("actual_n_rounds", actual_n_rounds)
        pbar.update(1)
        pbar.set_postfix_str(f"F1={score:.4f} n={actual_n_rounds}")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state))

    # Seed TPE with two configurations from the paper's reported region.
    study.enqueue_trial({
        "n_estimators": 500, "num_leaves": 512, "learning_rate": 0.03,
        "reg_lambda": 1.0, "scale_pos_weight": 1.5, "reg_alpha": 1.5,
        "subsample": 0.8,
    })
    study.enqueue_trial({
        "n_estimators": 800, "num_leaves": 2048, "learning_rate": 0.01,
        "reg_lambda": 10.0, "scale_pos_weight": 3.0, "reg_alpha": 1.2,
        "subsample": 0.7,
    })

    print(f"  Sequential trials (CPU, {threads_per_trial} threads/trial)")
    t0 = time.time()
    study.optimize(objective, n_trials=n_candidates, n_jobs=1, catch=(Exception,))
    pbar.close()
    elapsed = time.time() - t0

    return _finish_detection_study(study, "LightGBM", n_candidates, elapsed)


def tune_xgboost(X_train, y_train, X_val, y_val, n_candidates=20,
                 random_state=42) -> Dict:
    """TPE search for XGBoost detection. Objective: validation F1 at 0.5."""
    import optuna
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from tqdm import tqdm

    print(f"\n  {'-'*50}")
    print("  Tuning XGBoost (Optuna TPE, train/val holdout)")
    print(f"  Trials: {n_candidates}")
    print(f"  Train: {len(y_train):,} edges, Val: {len(y_val):,} edges")
    print(f"  {'-'*50}")

    pbar = tqdm(total=n_candidates, desc="  XGBoost", unit="trial",
                bar_format="  {desc} |{bar:35}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}{postfix}]")

    # XGBoost DMatrix is immutable once built, so both are shared across trials.
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            # AMLworld paper Table 10.
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 10**-2.5, 10**-1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        n_rounds = trial.suggest_int("n_estimators", 10, 1000)

        model = xgb.train(params, dtrain, num_boost_round=n_rounds,
                          evals=[(dval, "val")], early_stopping_rounds=20,
                          verbose_eval=False)

        actual_n_rounds = model.best_iteration
        val_probs = model.predict(dval)
        score = f1_score(y_val, (val_probs >= 0.5).astype(int), zero_division=0)

        trial.set_user_attr("actual_n_rounds", actual_n_rounds)
        pbar.update(1)
        pbar.set_postfix_str(f"F1={score:.4f} n={actual_n_rounds}")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state))

    study.enqueue_trial({
        "max_depth": 8, "learning_rate": 0.03, "reg_lambda": 1.0,
        "scale_pos_weight": 1.5, "colsample_bytree": 0.8, "subsample": 0.8,
        "n_estimators": 500,
    })
    study.enqueue_trial({
        "max_depth": 12, "learning_rate": 0.01, "reg_lambda": 10.0,
        "scale_pos_weight": 3.0, "colsample_bytree": 0.7, "subsample": 0.9,
        "n_estimators": 800,
    })

    print("  Sequential trials (CUDA, 1 GPU)")
    t0 = time.time()
    study.optimize(objective, n_trials=n_candidates, n_jobs=1, catch=(Exception,))
    pbar.close()
    elapsed = time.time() - t0

    return _finish_detection_study(study, "XGBoost", n_candidates, elapsed)


def _finish_detection_study(study, label: str, n_candidates: int,
                            elapsed: float) -> Dict:
    """Print the winner and assemble the JSON payload for a detection search."""
    import optuna

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    if not completed:
        print("  All trials failed")
        return {}

    best = study.best_params
    actual_n_rounds = study.best_trial.user_attrs.get("actual_n_rounds")
    print(f"  Best {label} params (F1@0.5={study.best_value:.4f}, "
          f"n_rounds={actual_n_rounds}, {fmt_time(elapsed)}, "
          f"{len(completed)} ok / {len(failed)} failed):")
    for k, v in sorted(best.items()):
        print(f"    {k}: {v}")

    result = {
        "best_params": _serialize_params(best),
        "best_score": float(study.best_value),
        # Constant by design; threshold selection is the training stage's job.
        "best_threshold": 0.5,
        "n_candidates": n_candidates,
        "n_trials_completed": len(completed),
        "n_pruned": 0,
        "n_failed": len(failed),
        "tuning_time_s": round(elapsed, 1),
    }
    if actual_n_rounds is not None:
        result["actual_n_rounds"] = actual_n_rounds
    return result


# ═════════════════════════════════════════════════════════════
# Typology searches
# ═════════════════════════════════════════════════════════════

def _extract_typology_splits(X_train, typ_train, X_val, typ_val):
    """Keep only the illicit edges that carry a typology label."""
    tr_mask = typ_train >= 0
    va_mask = typ_val >= 0
    return X_train[tr_mask], typ_train[tr_mask], X_val[va_mask], typ_val[va_mask]


def tune_lightgbm_typology(X_train, typ_train, X_val, typ_val,
                           n_candidates: int = 20,
                           random_state: int = 42) -> Dict:
    """TPE search for the LightGBM typology head. Objective: macro F1."""
    import lightgbm as lgb
    import optuna
    from sklearn.metrics import f1_score
    from tqdm import tqdm

    Xt, yt, Xv, yv = _extract_typology_splits(X_train, typ_train, X_val, typ_val)
    n_classes = len(TYPOLOGY_CLASSES)

    print(f"\n  {'-'*50}")
    print(f"  Tuning LightGBM typology (multiclass, {n_classes} classes)")
    print(f"  Train: {len(yt)} illicit edges, Val: {len(yv)} illicit edges")
    print(f"  Trials: {n_candidates}")
    print(f"  {'-'*50}")

    if len(yt) < 10 or len(yv) < 10:
        print("  Too few typology samples, skipping")
        return {}

    pbar = tqdm(total=n_candidates, desc="  LGB-Typ", unit="trial",
                bar_format="  {desc} |{bar:35}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}{postfix}]")

    def objective(trial):
        dtrain = lgb.Dataset(Xt, label=yt, free_raw_data=False)
        dval = lgb.Dataset(Xv, label=yv, reference=dtrain, free_raw_data=False)

        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "verbose": -1,
            "device": "cpu",
            "num_threads": 8,
            "class_weight": "balanced",
            "n_estimators": trial.suggest_int("n_estimators", 20, 300),
            "num_leaves": trial.suggest_int("num_leaves", 5, 64, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 5.0, log=True),
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=params.pop("n_estimators"),
            valid_sets=[dval], valid_names=["val"],
            callbacks=[lgb.early_stopping(15, verbose=False),
                       lgb.log_evaluation(0)],
        )

        score = f1_score(yv, model.predict(Xv).argmax(axis=1),
                         average="macro", zero_division=0)
        pbar.update(1)
        pbar.set_postfix_str(f"macroF1={score:.4f}")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state))

    t0 = time.time()
    # Four workers here, unlike the detection search: each trial fits at most a
    # few thousand illicit rows, which is small enough not to trip OpenMP.
    study.optimize(objective, n_trials=n_candidates, n_jobs=4, catch=(Exception,))
    pbar.close()

    return _finish_typology_study(study, "LGB", n_candidates, time.time() - t0)


def tune_xgboost_typology(X_train, typ_train, X_val, typ_val,
                          n_candidates: int = 20,
                          random_state: int = 42) -> Dict:
    """TPE search for the XGBoost typology head. Objective: macro F1."""
    import optuna
    import xgboost as xgb
    from sklearn.metrics import f1_score
    from tqdm import tqdm

    Xt, yt, Xv, yv = _extract_typology_splits(X_train, typ_train, X_val, typ_val)
    n_classes = len(TYPOLOGY_CLASSES)

    print(f"\n  {'-'*50}")
    print(f"  Tuning XGBoost typology (multiclass, {n_classes} classes)")
    print(f"  Train: {len(yt)} illicit edges, Val: {len(yv)} illicit edges")
    print(f"  Trials: {n_candidates}")
    print(f"  {'-'*50}")

    if len(yt) < 10 or len(yv) < 10:
        print("  Too few typology samples, skipping")
        return {}

    dtrain = xgb.DMatrix(Xt, label=yt)
    dval = xgb.DMatrix(Xv, label=yv)

    pbar = tqdm(total=n_candidates, desc="  XGB-Typ", unit="trial",
                bar_format="  {desc} |{bar:35}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}{postfix}]")

    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        n_rounds = trial.suggest_int("n_estimators", 20, 300)

        model = xgb.train(params, dtrain, num_boost_round=n_rounds,
                          evals=[(dval, "val")], early_stopping_rounds=15,
                          verbose_eval=False)

        score = f1_score(yv, model.predict(dval).argmax(axis=1),
                         average="macro", zero_division=0)
        pbar.update(1)
        pbar.set_postfix_str(f"macroF1={score:.4f}")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state))

    t0 = time.time()
    study.optimize(objective, n_trials=n_candidates, n_jobs=1, catch=(Exception,))
    pbar.close()

    return _finish_typology_study(study, "XGB", n_candidates, time.time() - t0)


def _finish_typology_study(study, label: str, n_candidates: int,
                           elapsed: float) -> Dict:
    """Print the winner and assemble the JSON payload for a typology search."""
    import optuna

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    if not completed:
        print("  All trials failed")
        return {}

    best = study.best_params
    print(f"  Best {label}-typology params (macroF1={study.best_value:.4f}, "
          f"{fmt_time(elapsed)}, {len(completed)} ok / {len(failed)} failed):")
    for k, v in sorted(best.items()):
        print(f"    {k}: {v}")

    return {
        "best_params": _serialize_params(best),
        "best_score": float(study.best_value),
        "n_candidates": n_candidates,
        "n_trials_completed": len(completed),
        "n_failed": len(failed),
        "tuning_time_s": round(elapsed, 1),
    }


# ═════════════════════════════════════════════════════════════
# Reading and writing data/tuned_params/
# ═════════════════════════════════════════════════════════════

def save_tuned_params(dataset: str, method: str, result: Dict,
                      typology: bool = False) -> Path:
    """Write one search result. Location comes from :mod:`amlc.paths`."""
    fn = paths.tuned_params(dataset, method, typology)
    paths.ensure(fn.parent)
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {fn}")
    return fn


def _load_field(dataset: str, method: str, typology: bool, key: str):
    fn = paths.tuned_params(dataset, method, typology)
    if not fn.exists():
        return None
    with open(fn) as f:
        return json.load(f).get(key)


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Optuna search for the boosted-tree baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m amlc.baselines.ml.tuning --dataset HI-Small --n-candidates 50
  python -m amlc.baselines.ml.tuning --dataset HI-Small --models lightgbm

data/tuned_params/ already holds the results of these searches, so this stage
is optional.
        """,
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_MAP),
                        choices=list(MODEL_MAP))
    parser.add_argument("--n-candidates", type=int, default=30,
                        help="Optuna trials per search (default: 30)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="AMLworld CSV directory. Default: data/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Threads for feature extraction")
    args = parser.parse_args()

    print("=" * 60)
    print("  Hyperparameter tuning")
    print("=" * 60)
    print(f"  Dataset:  {args.dataset}")
    print(f"  Models:   {args.models}")
    print(f"  Trials:   {args.n_candidates}")
    print(f"  Seed:     {args.seed}")
    print(f"  Workers:  {args.workers}")
    print(f"  Output:   {paths.tuned_params(args.dataset, 'MODEL').parent}")

    t_total = time.time()

    # Offset by the dataset name so the two searches do not walk the same path.
    dataset_seed = args.seed + hash(args.dataset) % 10000

    data_path = Path(args.data_path) if args.data_path else None
    data = prepare_data(args.dataset, data_path=data_path, n_workers=args.workers)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    typ_train, typ_val = data["typ_train"], data["typ_val"]

    searches = {
        "lightgbm": (tune_lightgbm, tune_lightgbm_typology),
        "xgboost": (tune_xgboost, tune_xgboost_typology),
    }

    for model in args.models:
        method = MODEL_MAP[model]
        detect, typology = searches[model]

        result = detect(X_train, y_train, X_val, y_val,
                        n_candidates=args.n_candidates,
                        random_state=dataset_seed)
        if result:
            save_tuned_params(args.dataset, method, result)

        typ_result = typology(X_train, typ_train, X_val, typ_val,
                              n_candidates=args.n_candidates,
                              random_state=dataset_seed)
        if typ_result:
            save_tuned_params(args.dataset, method, typ_result, typology=True)

    print(f"\n{'='*60}")
    print(f"  Done: {fmt_time(time.time() - t_total)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
