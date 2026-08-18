"""Naive Coreset versus HT-Coreset, scored per supervised model.

The ablation compares sampling strategies on the ensemble. This module asks the
other question: does a subset preserve each individual scorer's metrics, or only
the ensemble's? Every scorer and seed is evaluated three times, on the full
temporal test split, on the Naive Coreset subset (unweighted, because that
construction has no weights) and on the HT-Coreset (weighted and unweighted).

The HT-Coreset holds every scorer's precision, recall and F1 to the full-split
values, with one instructive exception: GCPAL+GFP flags benign edges the
ensemble is confident about, and those edges sit outside the hard-negative
stratum, so its false positives are estimated from the sampled stratum rather
than from a census.

Reads:

    <archive>/test_probs/<dataset>/         labels, typologies, scorer probabilities
    <coreset-dir>/<dataset>/ht_subset_indices.npy, ht_weights.npy
    the Naive subset, passed explicitly with --naive-subset

Writes, under ``<out>/``:

    naive_vs_ht_models.csv    one row per (dataset, scorer, seed) plus ensemble
    naive_vs_ht_models.json   the same rows
    naive_vs_ht_summary.json  mean +/- std over seeds, for the table generator

Why --naive-subset is explicit
------------------------------
The archived script discovered the Naive subset by globbing the run directory
for ``subset_<dataset>_n*.npy`` and taking the largest, falling back from an
``exhaustive_best`` file when it was absent. Under the release naming those
files are ``naive_subset_<dataset>_n<size>.npy`` and the searched minimum can
come from either search, so a glob would either miss the file or silently pick a
different subset than the one the paper reports. The path is now an argument and
the comparison simply reports "no Naive subset" when it is not given.

Which HT weights
----------------
The default is the released coreset under ``data/coreset/<dataset>/``, whose
weights are the repaired ones. The archived LI-Small weight vector double-counts
almost the whole benign population, which inflates every weighted false-positive
mass computed from it; see :mod:`amlc.coreset.ht_weights`. Point
``--coreset-dir`` at the archived vectors to reproduce the submitted CSV.

Usage:
    python -m amlc.coreset.compare \\
        --naive-subset HI-Small=/path/naive_subset_HI-Small_exhaustive_best.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .. import paths
from ..config import DATASETS, ENSEMBLE_MEMBERS, SEEDS
from .common import (
    EVALUATION_THRESHOLD_GRID,
    ensemble_soft_avg,
    ensemble_typology_vote,
    find_threshold,
    load_member_typologies,
    load_test_data,
    typology_metrics,
    unweighted_metrics,
    weighted_metrics,
)

ENSEMBLE_ROW = "Ensemble(soft-avg)"

_EMPTY_TYP = dict(typ_macro_f1=0.0, typ_accuracy=0.0, typ_n_eval=0)


def _row(dataset: str, model: str, seed, threshold: float,
         labels: np.ndarray, probs: np.ndarray,
         ht_idx: np.ndarray, ht_weights: np.ndarray,
         naive_idx: Optional[np.ndarray],
         gt_typ: Optional[np.ndarray],
         typ_pred: Optional[np.ndarray]) -> dict:
    """Score one predictor on the full split, the Naive subset and the coreset."""
    n_full = len(labels)
    preds_full = (probs >= threshold).astype(int)
    met_full = unweighted_metrics(labels, preds_full)

    naive_met = dict(precision=0.0, recall=0.0, f1=0.0)
    naive_n, naive_red = 0, 0.0
    if naive_idx is not None:
        naive_preds = (probs[naive_idx] >= threshold).astype(int)
        naive_met = unweighted_metrics(labels[naive_idx], naive_preds)
        naive_n = len(naive_idx)
        naive_red = (1 - naive_n / n_full) * 100

    ht_labels = labels[ht_idx]
    ht_preds = (probs[ht_idx] >= threshold).astype(int)
    ht_w_met = weighted_metrics(ht_labels, ht_preds, ht_weights)
    ht_uw_met = unweighted_metrics(ht_labels, ht_preds)

    typ_full, typ_ht = _EMPTY_TYP, _EMPTY_TYP
    if gt_typ is not None and typ_pred is not None:
        typ_full = typology_metrics(gt_typ, preds_full, typ_pred)
        typ_ht = typology_metrics(gt_typ[ht_idx], ht_preds, typ_pred[ht_idx])

    return dict(
        dataset=dataset, model=model, seed=seed, threshold=threshold,
        n_full=n_full,
        has_naive=naive_idx is not None,
        naive_n_subset=naive_n, naive_reduction_pct=naive_red,
        naive_precision=naive_met["precision"], naive_recall=naive_met["recall"],
        naive_f1=naive_met["f1"],
        naive_delta_p=(abs(naive_met["precision"] - met_full["precision"])
                       if naive_idx is not None else 0.0),
        naive_delta_r=(abs(naive_met["recall"] - met_full["recall"])
                       if naive_idx is not None else 0.0),
        naive_delta_f1=(abs(naive_met["f1"] - met_full["f1"])
                        if naive_idx is not None else 0.0),
        ht_n_subset=len(ht_idx),
        ht_reduction_pct=(1 - len(ht_idx) / n_full) * 100,
        full_precision=met_full["precision"], full_recall=met_full["recall"],
        full_f1=met_full["f1"],
        ht_w_precision=ht_w_met["precision"], ht_w_recall=ht_w_met["recall"],
        ht_w_f1=ht_w_met["f1"],
        ht_w_delta_p=abs(ht_w_met["precision"] - met_full["precision"]),
        ht_w_delta_r=abs(ht_w_met["recall"] - met_full["recall"]),
        ht_w_delta_f1=abs(ht_w_met["f1"] - met_full["f1"]),
        ht_uw_precision=ht_uw_met["precision"], ht_uw_recall=ht_uw_met["recall"],
        ht_uw_f1=ht_uw_met["f1"],
        ht_uw_delta_p=abs(ht_uw_met["precision"] - met_full["precision"]),
        ht_uw_delta_r=abs(ht_uw_met["recall"] - met_full["recall"]),
        ht_uw_delta_f1=abs(ht_uw_met["f1"] - met_full["f1"]),
        full_typ_macro_f1=typ_full["typ_macro_f1"],
        full_typ_accuracy=typ_full["typ_accuracy"],
        ht_typ_macro_f1=typ_ht["typ_macro_f1"],
        ht_typ_accuracy=typ_ht["typ_accuracy"],
        ht_delta_typ_macro_f1=abs(typ_ht["typ_macro_f1"]
                                  - typ_full["typ_macro_f1"]),
        ht_delta_typ_accuracy=abs(typ_ht["typ_accuracy"]
                                  - typ_full["typ_accuracy"]),
    )


def _print_row(r: dict) -> None:
    naive = (f"{r['naive_precision']*100:>5.1f}% {r['naive_recall']*100:>5.1f}% "
             f"{r['naive_f1']*100:>5.1f}%" if r["has_naive"]
             else f"{'no Naive subset':^20s}")
    naive_d = (f"{r['naive_delta_f1']*100:>5.1f}%" if r["has_naive"]
               else f"{'-':>6s}")
    print(f"  {r['model']:<22s} {str(r['seed']):>5s} | "
          f"{r['full_precision']*100:>5.1f}% {r['full_recall']*100:>5.1f}% "
          f"{r['full_f1']*100:>5.1f}% | {naive} | "
          f"{r['ht_w_precision']*100:>5.1f}% {r['ht_w_recall']*100:>5.1f}% "
          f"{r['ht_w_f1']*100:>5.1f}% | "
          f"{naive_d} {r['ht_w_delta_f1']*100:>6.1f}%")


def run(datasets: Sequence[str] = DATASETS,
        members: Sequence[str] = ENSEMBLE_MEMBERS,
        seeds: Sequence[int] = SEEDS,
        naive_subsets: Optional[dict] = None,
        archive: Optional[Path] = None,
        coreset_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None) -> list:
    """Compare the two constructions per scorer, for each dataset."""
    naive_subsets = naive_subsets or {}
    out_dir = paths.ensure(Path(out_dir) if out_dir
                           else paths.results() / "coreset")

    print(f"\n{'='*80}")
    print("  Naive Coreset versus HT-Coreset, per model")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    all_results = []

    for dataset in datasets:
        print(f"\n{'='*110}\n  {dataset}\n{'='*110}")

        data = load_test_data(dataset, members, seeds, archive=archive)
        labels = data["labels"]
        gt_typ = data["typologies"]
        n_full = data["n"]
        model_probs = data["probs"]
        member_typ = load_member_typologies(dataset, members, seeds,
                                            archive=archive)

        cd = Path(coreset_dir) / dataset if coreset_dir else paths.coreset(dataset)
        ht_idx = np.load(cd / "ht_subset_indices.npy")
        ht_w = np.load(cd / "ht_weights.npy")
        n_ill_full = int(labels.sum())
        n_ht_ill = int(labels[ht_idx].sum())
        print(f"  HT-Coreset: {len(ht_idx):,} edges ({n_ht_ill:,} illicit, "
              f"{n_ht_ill/len(ht_idx)*100:.3f}%), "
              f"{(1 - len(ht_idx)/n_full)*100:.1f}% reduction, "
              f"weights sum to {ht_w.sum():,.0f} of {n_full:,}")

        naive_idx = None
        naive_path = naive_subsets.get(dataset)
        if naive_path:
            naive_idx = np.load(naive_path)
            n_naive_ill = int(labels[naive_idx].sum())
            print(f"  Naive Coreset: {len(naive_idx):,} edges "
                  f"({n_naive_ill:,} illicit, "
                  f"{n_naive_ill/len(naive_idx)*100:.3f}%), "
                  f"{(1 - len(naive_idx)/n_full)*100:.1f}% reduction "
                  f"[{naive_path}]")
        else:
            print("  Naive Coreset: not given, pass --naive-subset "
                  f"{dataset}=<path> to include it")
        print(f"  Full test split: {n_full:,} edges "
              f"({n_ill_full:,} illicit, {n_ill_full/n_full*100:.3f}%)")

        print(f"\n  {'Model':<22s} {'Seed':>5s} | {'--- full split ---':^20s} | "
              f"{'--- Naive ---':^20s} | {'--- HT weighted ---':^20s} | "
              f"{'dF1(N)':>6s} {'dF1(HT)':>7s}")
        print(f"  {'-'*104}")

        ds_rows = []
        for member in members:
            member_rows = []
            for seed in seeds:
                if (member, seed) not in model_probs:
                    continue
                probs = model_probs[(member, seed)]
                threshold, _ = find_threshold(labels, probs,
                                              grid=EVALUATION_THRESHOLD_GRID)
                r = _row(dataset, member, seed, threshold, labels, probs,
                         ht_idx, ht_w, naive_idx, gt_typ,
                         member_typ.get((member, seed)))
                member_rows.append(r)
                ds_rows.append(r)
                _print_row(r)
            if len(member_rows) > 1:
                f1s = [r["ht_w_delta_f1"] for r in member_rows]
                print(f"  {'  mean |dF1| (HT)':<22s} {'':>5s} | "
                      f"{np.mean(f1s)*100:.2f}% +/- {np.std(f1s)*100:.2f}%")

        ens_probs = ensemble_soft_avg(data)
        ens_threshold, _ = find_threshold(labels, ens_probs,
                                          grid=EVALUATION_THRESHOLD_GRID)
        ens_typ = (ensemble_typology_vote(list(member_typ.values()), n_full)
                   if len(member_typ) >= 2 else None)
        r_ens = _row(dataset, ENSEMBLE_ROW, "all", ens_threshold, labels,
                     ens_probs, ht_idx, ht_w, naive_idx, gt_typ, ens_typ)
        ds_rows.append(r_ens)
        _print_row(r_ens)

        all_results.extend(ds_rows)

    _save(all_results, datasets, members, out_dir)
    return all_results


# ─────────────────────────────────────────────────────────────────────────
#  Result files
# ─────────────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "dataset", "model", "seed", "threshold", "n_full", "has_naive",
    "naive_n_subset", "naive_reduction_pct",
    "naive_precision", "naive_recall", "naive_f1",
    "naive_delta_p", "naive_delta_r", "naive_delta_f1",
    "ht_n_subset", "ht_reduction_pct",
    "full_precision", "full_recall", "full_f1",
    "ht_w_precision", "ht_w_recall", "ht_w_f1",
    "ht_w_delta_p", "ht_w_delta_r", "ht_w_delta_f1",
    "ht_uw_precision", "ht_uw_recall", "ht_uw_f1",
    "ht_uw_delta_p", "ht_uw_delta_r", "ht_uw_delta_f1",
    "full_typ_macro_f1", "full_typ_accuracy",
    "ht_typ_macro_f1", "ht_typ_accuracy",
    "ht_delta_typ_macro_f1", "ht_delta_typ_accuracy",
]

_SUMMARY_KEYS = [
    "full_precision", "full_recall", "full_f1",
    "ht_w_precision", "ht_w_recall", "ht_w_f1", "ht_w_delta_f1", "ht_uw_f1",
    "full_typ_macro_f1", "full_typ_accuracy",
    "ht_typ_macro_f1", "ht_typ_accuracy",
    "ht_delta_typ_macro_f1", "ht_delta_typ_accuracy",
]


def _save(rows: list, datasets: Sequence[str], members: Sequence[str],
          out_dir: Path) -> None:
    csv_path = out_dir / "naive_vs_ht_models.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved CSV : {csv_path}")

    json_path = out_dir / "naive_vs_ht_models.json"
    clean = [{k: (int(v) if isinstance(v, np.integer)
                  else float(v) if isinstance(v, np.floating) else v)
              for k, v in r.items()} for r in rows]
    with open(json_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  Saved JSON: {json_path}")

    def fmt(values):
        return f"{np.mean(values)*100:.1f}+/-{np.std(values)*100:.1f}"

    summary_rows = []
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        has_naive = any(r["has_naive"] for r in ds_rows)
        for model in list(members) + [ENSEMBLE_ROW]:
            m_rows = [r for r in ds_rows if r["model"] == model]
            if not m_rows:
                continue
            row = {"model": model, "dataset": dataset, "n_seeds": len(m_rows),
                   "ht_n_subset": m_rows[0]["ht_n_subset"],
                   "ht_reduction_pct": f"{m_rows[0]['ht_reduction_pct']:.1f}"}
            row.update({k: fmt([r[k] for r in m_rows]) for k in _SUMMARY_KEYS})
            if has_naive:
                row["naive_n_subset"] = m_rows[0]["naive_n_subset"]
                row["naive_reduction_pct"] = (
                    f"{m_rows[0]['naive_reduction_pct']:.1f}")
                row["naive_f1"] = fmt([r["naive_f1"] for r in m_rows])
                row["naive_delta_f1"] = fmt([r["naive_delta_f1"] for r in m_rows])
            summary_rows.append(row)

    summary_path = out_dir / "naive_vs_ht_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"  Saved summary JSON: {summary_path}")


def _parse_mapping(items: Optional[Sequence[str]]) -> dict:
    """Parse repeated ``DATASET=PATH`` arguments into a dict."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(
                f"--naive-subset expects DATASET=PATH, got {item!r}")
        dataset, path = item.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"--naive-subset file not found: {p}")
        out[dataset] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-model comparison of the Naive Coreset and the HT-Coreset")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--members", nargs="+", default=list(ENSEMBLE_MEMBERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--naive-subset", action="append", metavar="DATASET=PATH",
                    help="Naive Coreset index file for one dataset; repeatable. "
                         "Omit to report the HT-Coreset columns only")
    ap.add_argument("--archive", type=Path, default=None,
                    help="run directory outputs/; defaults to AMLC_ARCHIVE")
    ap.add_argument("--coreset-dir", type=Path, default=None,
                    help="directory holding <dataset>/ht_subset_indices.npy and "
                         "ht_weights.npy; defaults to the released data/coreset")
    ap.add_argument("--out", type=Path, default=paths.results() / "coreset")
    args = ap.parse_args()

    run(datasets=args.datasets, members=args.members, seeds=args.seeds,
        naive_subsets=_parse_mapping(args.naive_subset),
        archive=args.archive, coreset_dir=args.coreset_dir, out_dir=args.out)


if __name__ == "__main__":
    main()
