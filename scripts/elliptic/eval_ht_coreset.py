#!/usr/bin/env python3
"""Evaluate the HT-Coreset and 7 ablation baselines against the full Elliptic test set.

Reads `outputs/elliptic/coreset/{v2,B*}.npz` and computes Horvitz-Thompson
weighted P/R/F1 against the full test split's metrics. Reports per-baseline:

  |ΔP|, |ΔR|, |ΔF1|  mean and std across the K random draws (the HT-Coreset is
  deterministic so std=0)

Outputs:
  outputs/elliptic/coreset/eval_summary.csv
  outputs/elliptic/coreset/eval_summary.json (paired-bootstrap ΔF1 CIs and
  Wilcoxon p-values against the HT-Coreset, the same significance protocol as Table 2)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR  = REPO_ROOT / "data" / "elliptic" / "proc"
ML_DIR    = REPO_ROOT / "outputs" / "elliptic" / "ml"
CORE_DIR  = REPO_ROOT / "outputs" / "elliptic" / "coreset"

# The seven baselines the paper's Elliptic ablation table reports, plus
# B2t_time, an extra time-step-stratified variant kept for information and not
# part of that table.
BASELINES = ["B1_random", "B2_stratified", "B2t_time", "B3_no_iw", "B4_no_hardneg",
             "B5_fogliato", "B6_leskovec", "B7_gao"]


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     w: np.ndarray) -> dict[str, float]:
    tp = float(w[(y_true == 1) & (y_pred == 1)].sum())
    fp = float(w[(y_true == 0) & (y_pred == 1)].sum())
    fn = float(w[(y_true == 1) & (y_pred == 0)].sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return dict(precision=p, recall=r, f1=f)


def full_metrics(probs: np.ndarray, y: np.ndarray, thr: float) -> dict[str, float]:
    yhat = (probs >= thr).astype(int)
    return dict(
        precision=float(precision_score(y, yhat, zero_division=0)),
        recall   =float(recall_score   (y, yhat, zero_division=0)),
        f1       =float(f1_score       (y, yhat, zero_division=0)),
    )


def evaluate_draws(draws_idx, draws_w, probs, y, thr,
                   label: str = "") -> list[dict[str, float]]:
    """Score each draw, flagging degenerate ones instead of scoring them as 0.

    A subset holding no positive at all cannot estimate precision or recall:
    every metric collapses to 0 and the reported bias becomes the full-set
    value itself, with zero variance across draws. That is not a measurement
    of the sampler, it is a measurement of the subset being empty of the class
    under test, and it is what produced the |dF1| = 80.63 pp rows before the
    B1/B2 pool fix. Surface it loudly.
    """
    out, degenerate = [], 0
    for idx, w in zip(draws_idx, draws_w):
        idx = np.asarray(idx, dtype=int); w = np.asarray(w, dtype=float)
        if len(idx) == 0:
            out.append(dict(precision=0., recall=0., f1=0., degenerate=True))
            degenerate += 1
            continue
        n_pos = int((y[idx] == 1).sum())
        ypred = (probs[idx] >= thr).astype(int)
        m = weighted_metrics(y[idx], ypred, w)
        m["degenerate"] = n_pos == 0
        degenerate += m["degenerate"]
        out.append(m)
    if degenerate:
        print(f"  [eval] WARNING {label or 'baseline'}: {degenerate}/{len(out)} "
              "draws contain no positive node; their metrics are 0 by "
              "construction and the reported bias is the full-set value, not a "
              "property of the sampler.")
    return out


def paired_bootstrap_delta_f1(ht_d: list[dict], base_d: list[dict],
                              full_f1: float, n_boot: int = 2000,
                              seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.array([
        abs(b["f1"] - full_f1) - abs(v["f1"] - full_f1)
        for v, b in zip(ht_d, base_d)
    ])
    if len(diffs) == 0:
        return float("nan"), float("nan")
    boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                     for _ in range(n_boot)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--ml-dir",   type=Path, default=ML_DIR)
    ap.add_argument("--core-dir", type=Path, default=CORE_DIR)
    ap.add_argument("--n-boot",   type=int,  default=2000)
    args = ap.parse_args()

    test = np.load(args.proc_dir / "test.npz")
    y    = test["y"].astype(np.int32)
    v2   = np.load(args.core_dir / "v2.npz")
    probs, thr = v2["ensemble_probs"], float(v2["threshold"])

    full = full_metrics(probs, y, thr)
    print(f"[eval] full test:  P={full['precision']*100:5.2f}  "
          f"R={full['recall']*100:5.2f}  F1={full['f1']*100:5.2f}  thr={thr:.3f}")

    # The HT-Coreset: a single deterministic draw, stored as v2.npz
    ht_idx, ht_w = v2["idx"], v2["weights"]
    ht_metrics = weighted_metrics((y[ht_idx]),
                                  (probs[ht_idx] >= thr).astype(int), ht_w)
    print(f"[eval] HT-Coreset (n={len(ht_idx)}):  "
          f"|ΔP|={abs(ht_metrics['precision']-full['precision'])*100:5.2f}  "
          f"|ΔR|={abs(ht_metrics['recall']-full['recall'])*100:5.2f}  "
          f"|ΔF1|={abs(ht_metrics['f1']-full['f1'])*100:5.2f}")

    # Baselines
    rows = []
    rows.append(dict(
        method="HT-Coreset", n_subset=int(len(ht_idx)),
        delta_p_mean=abs(ht_metrics["precision"] - full["precision"]),
        delta_r_mean=abs(ht_metrics["recall"]    - full["recall"]),
        delta_f1_mean=abs(ht_metrics["f1"]       - full["f1"]),
        delta_p_std=0., delta_r_std=0., delta_f1_std=0.,
        sigma_f1=0., wilcoxon_p="-", boot_lo="-", boot_hi="-",
    ))

    ht_draws = [ht_metrics] * 50  # same draw repeated for paired comparisons

    summary = {"full": full, "threshold": thr, "n_test": int(len(y)),
               "v2": ht_metrics, "baselines": {}}

    for name in BASELINES:
        path = args.core_dir / f"{name}.npz"
        if not path.exists():
            print(f"  [eval] missing {path}; skip"); continue
        d = np.load(path, allow_pickle=True)
        draws = evaluate_draws(d["draws_idx"], d["draws_w"], probs, y, thr, label=name)
        delta_p  = np.array([abs(m["precision"] - full["precision"]) for m in draws])
        delta_r  = np.array([abs(m["recall"]    - full["recall"])    for m in draws])
        delta_f1 = np.array([abs(m["f1"]        - full["f1"])        for m in draws])

        try:
            stat, pval = wilcoxon(
                np.array([abs(m["f1"] - full["f1"]) for m in draws]),
                np.array([abs(ht_draws[i]["f1"] - full["f1"]) for i in range(len(draws))]),
                alternative="greater", zero_method="wilcox",
            )
        except Exception:
            pval = float("nan")
        lo, hi = paired_bootstrap_delta_f1(ht_draws, draws, full["f1"], args.n_boot)

        rows.append(dict(
            method=name, n_subset=int(len(d["draws_idx"][0])),
            delta_p_mean=float(delta_p.mean()),
            delta_r_mean=float(delta_r.mean()),
            delta_f1_mean=float(delta_f1.mean()),
            delta_p_std=float(delta_p.std()),
            delta_r_std=float(delta_r.std()),
            delta_f1_std=float(delta_f1.std()),
            sigma_f1=float(np.std([m["f1"] for m in draws])),
            wilcoxon_p=float(pval),
            boot_lo=float(lo), boot_hi=float(hi),
        ))
        summary["baselines"][name] = dict(
            mean_delta_p=float(delta_p.mean()),
            mean_delta_r=float(delta_r.mean()),
            mean_delta_f1=float(delta_f1.mean()),
            std_f1=float(np.std([m["f1"] for m in draws])),
            wilcoxon_p_vs_v2=float(pval) if pval == pval else None,
            boot_ci_delta_f1=[float(lo), float(hi)],
        )

    # Write CSV
    out_csv = args.core_dir / "eval_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = f"{v:.6f}"
            writer.writerow(row)
    print(f"[eval] wrote {out_csv}")
    (args.core_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote {args.core_dir/'eval_summary.json'}")

    # Pretty table
    print("\n" + "─" * 84)
    print(f"{'Method':<16} {'n':>6}  {'|ΔP|':>7}  {'|ΔR|':>7}  {'|ΔF1|':>7}  {'σ(F1)':>7}  {'p_W':>9}")
    print("─" * 84)
    for r in rows:
        print(f"{r['method']:<16} {r['n_subset']:>6d}  "
              f"{float(r['delta_p_mean'])*100:>6.2f}%  "
              f"{float(r['delta_r_mean'])*100:>6.2f}%  "
              f"{float(r['delta_f1_mean'])*100:>6.2f}%  "
              f"{float(r['sigma_f1'])*100:>6.2f}%  "
              f"{r['wilcoxon_p']!s:>9}")
    print("─" * 84)


if __name__ == "__main__":
    main()
