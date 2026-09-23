#!/usr/bin/env python3
"""Train LightGBM and XGBoost baselines on the Elliptic train split.

Mirrors the AMLworld ML baselines (5 seeds, F1@0.5 tuning objective on a 10%
val cut from the train split, soft-vote ensemble), but with two simplifications
appropriate for Elliptic's smaller scale:

  - No GFP feature extraction: Elliptic ships 165 pre-engineered features
    per transaction (94 local + 72 aggregated, per Weber et al. 2019).
  - No GCPAL: the line-graph contrastive baseline is heavyweight and adds
    little for the HT-Coreset generalisation claim. The 2-model soft-vote ensemble
    is sufficient to define ensemble difficulty for the coreset stage.

Outputs (`outputs/elliptic/ml/`):
    {model}_seed{s}.npz       per-seed val/test predictions
    summary.json              full-set test P/R/F1 mean ± std

Usage:
    python scripts/elliptic/train_ml_baselines.py --seeds 42 123 456 789 1011
    python scripts/elliptic/train_ml_baselines.py --seeds 42 --threshold-mode fixed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR  = REPO_ROOT / "data" / "elliptic" / "proc"
OUT_DIR   = REPO_ROOT / "outputs" / "elliptic" / "ml"


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep [0.01, 0.99] for argmax-F1 — same protocol as the AMLworld ML."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def train_lightgbm(X_tr, y_tr, X_val, y_val, seed: int) -> lgb.Booster:
    params = dict(
        objective="binary", metric="binary_logloss", learning_rate=0.05,
        num_leaves=64, min_data_in_leaf=50, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5, lambda_l2=1.0,
        is_unbalance=True, verbosity=-1, seed=seed,
    )
    return lgb.train(
        params,
        lgb.Dataset(X_tr, label=y_tr),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )


def train_xgboost(X_tr, y_tr, X_val, y_val, seed: int) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)
    pos_weight = float((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    params = dict(
        objective="binary:logistic", eval_metric="logloss",
        learning_rate=0.05, max_depth=8, min_child_weight=5,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        scale_pos_weight=pos_weight, seed=seed, nthread=4, verbosity=0,
    )
    return xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dval, "val")], early_stopping_rounds=20, verbose_eval=False,
    )


def predict_proba(model, X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "lightgbm":
        return model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    return model.predict(xgb.DMatrix(X),
                         iteration_range=(0, getattr(model, "best_iteration", 0) + 1))


def run_one_seed(X_tr, y_tr, X_te, y_te, seed: int,
                 threshold_mode: str, fixed_threshold: float) -> dict:
    X_tr_, X_val, y_tr_, y_val = train_test_split(
        X_tr, y_tr, test_size=0.10, random_state=seed,
        stratify=y_tr if (y_tr == 1).sum() > 1 else None,
    )

    print(f"  [seed {seed}] train={len(X_tr_)} val={len(X_val)} (illicit "
          f"{int((y_tr_==1).sum())}/{int((y_val==1).sum())})")

    lgbm = train_lightgbm(X_tr_, y_tr_, X_val, y_val, seed)
    xgbm = train_xgboost (X_tr_, y_tr_, X_val, y_val, seed)

    p_lgb_val = predict_proba(lgbm, X_val, "lightgbm")
    p_xgb_val = predict_proba(xgbm, X_val, "xgboost")
    p_ens_val = 0.5 * (p_lgb_val + p_xgb_val)

    p_lgb_te = predict_proba(lgbm, X_te, "lightgbm")
    p_xgb_te = predict_proba(xgbm, X_te, "xgboost")
    p_ens_te = 0.5 * (p_lgb_te + p_xgb_te)

    if threshold_mode == "tuned":
        thr = best_threshold(y_val, p_ens_val)
    else:
        thr = fixed_threshold

    yhat = (p_ens_te >= thr).astype(int)
    metrics = dict(
        precision=float(precision_score(y_te, yhat, zero_division=0)),
        recall   =float(recall_score   (y_te, yhat, zero_division=0)),
        f1       =float(f1_score       (y_te, yhat, zero_division=0)),
        threshold=float(thr),
        n_test=int(len(y_te)),
        n_test_illicit=int((y_te == 1).sum()),
    )
    return dict(
        seed=seed, metrics=metrics,
        probs=dict(lightgbm=p_lgb_te, xgboost=p_xgb_te, ensemble=p_ens_te),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    ap.add_argument("--threshold-mode", choices=["tuned", "fixed"], default="tuned")
    ap.add_argument("--fixed-threshold", type=float, default=0.5)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = np.load(args.proc_dir / "train.npz")
    test  = np.load(args.proc_dir / "test.npz")
    X_tr, y_tr = train["X"], train["y"].astype(np.int32)
    X_te, y_te = test ["X"], test ["y"].astype(np.int32)
    print(f"[train] train {X_tr.shape}  test {X_te.shape}  "
          f"illicit rate (test) {(y_te==1).mean()*100:.2f}%")

    summary = {"seeds": [], "metrics": []}
    for seed in args.seeds:
        result = run_one_seed(
            X_tr, y_tr, X_te, y_te, seed,
            args.threshold_mode, args.fixed_threshold,
        )
        np.savez(args.out_dir / f"ensemble_seed{seed}.npz",
                 lightgbm=result["probs"]["lightgbm"],
                 xgboost =result["probs"]["xgboost"],
                 ensemble=result["probs"]["ensemble"],
                 y_test=y_te, threshold=result["metrics"]["threshold"])
        summary["seeds"].append(seed)
        summary["metrics"].append(result["metrics"])
        m = result["metrics"]
        print(f"  [seed {seed}] P={m['precision']*100:5.2f}  "
              f"R={m['recall']*100:5.2f}  F1={m['f1']*100:5.2f}  "
              f"(thr={m['threshold']:.3f})")

    metrics = summary["metrics"]
    summary["mean"] = {
        k: float(np.mean([m[k] for m in metrics]))
        for k in ("precision", "recall", "f1", "threshold")
    }
    summary["std"] = {
        k: float(np.std ([m[k] for m in metrics]))
        for k in ("precision", "recall", "f1", "threshold")
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[train] mean ± std (test, full):")
    for k in ("precision", "recall", "f1"):
        print(f"  {k:9s}: {summary['mean'][k]*100:5.2f} ± "
              f"{summary['std'][k]*100:.2f}")
    print(f"\n[train] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
