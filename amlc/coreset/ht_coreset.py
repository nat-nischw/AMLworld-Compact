"""HT-Coreset construction: the benign-budget sweep and the released draw.

Stage 1 keeps every illicit edge, so recall is exact. Stage 2 samples benign
edges from four strata (hard negatives, then difficulty terciles over the rest).
Stage 3 attaches Horvitz-Thompson weights, giving unbiased full-split confusion
count estimates for fixed predictions and known positive inclusion probabilities.
Precision and F1 are ratio estimates and can have finite-sample bias.

This module drives that construction. It sweeps benign budgets on a geometric
grid, evaluates each budget over ``k`` random draws, reports HT-weighted and
unweighted metrics against the full split, and keeps the smallest budget whose
weighted deltas are all within tolerance. That budget is the released coreset:
3,753 edges on HI-Small and 2,268 on LI-Small.

Reads (only with the archive configured):

    <archive>/test_probs/<dataset>/            labels, typologies, scorer probabilities

Writes, under ``<out>/<dataset>/``:

    ensemble_probs_<dataset>.npy      soft-average scorer probability, full split
    ht_subset_indices.npy             the chosen draw, indices into the full split
    ht_weights.npy                    its HT weights
    ht_coreset_<dataset>_sizes.csv    one row per benign budget, mean and std
    ht_coreset_<dataset>_draws.csv    one row per (budget, draw)
    ht_coreset_summary_<dataset>.json everything above plus the run's parameters
    naive_vs_ht_<dataset>.csv         written when the Naive Coreset summary is
                                      present in the same directory

The benign sampling calls :func:`amlc.coreset.sampler.sample_benign_stratified`
rather than carrying its own copy. That is the one behavioural change against the
archived generator: weights are assigned from realised stratum membership at the
end of the draw, so the spare-fill block no longer adds a second copy of the
benign population to the weight total. See the sampler's docstring for what the
defect did to LI-Small.

The archived summary also recorded a KL typology threshold that this
construction never applied as a constraint. It is dropped rather than carried as
a knob that does nothing.

Usage:
    python -m amlc.coreset.ht_coreset --datasets HI-Small LI-Small
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .. import paths
from ..config import (
    BENIGN_MULTIPLIER,
    DATASETS,
    CONSTRUCTION_MEMBERS,
    HARD_NEG_RATIO,
    HARD_NEG_THRESHOLD_FRAC,
    SEEDS,
)
from . import common
from .common import (
    DEFAULT_WORKERS,
    TYPOLOGY_NAMES,
    build_response_matrix,
    check_irt,
    compute_difficulty,
    ensemble_soft_avg,
    estimate_context_size,
    find_threshold,
    irt_difficulty_1pl,
    irt_discrimination,
    kl_div,
    load_test_data,
    token_stats,
    typology_dist,
    unweighted_metrics,
    weighted_metrics,
)
from .sampler import sample_benign_stratified

#: Smallest benign budget the sweep considers, whatever the illicit count is.
MIN_BENIGN_BUDGET = 500


# ─────────────────────────────────────────────────────────────────────────
#  One draw
# ─────────────────────────────────────────────────────────────────────────

def evaluate_subset(data: dict, labels: np.ndarray, typologies: np.ndarray,
                    difficulty: np.ndarray, ens_probs: np.ndarray,
                    threshold: float, full_metrics: dict,
                    ill_idx: np.ndarray, ben_idx: np.ndarray,
                    ben_weights: np.ndarray,
                    ref_typ_dist: np.ndarray,
                    tok_stats: dict) -> dict:
    """Score one realised draw (all illicit plus the sampled benign edges).

    Reports the HT-weighted metrics, which are the estimators the paper uses,
    beside the unweighted ones, which show how far the 1:2 subset ratio moves
    precision when the weights are ignored.
    """
    subset_idx = np.sort(np.concatenate([ill_idx, ben_idx]))
    n_sub = len(subset_idx)

    sub_labels = labels[subset_idx]
    sub_preds = (ens_probs[subset_idx] >= threshold).astype(int)

    weights = np.ones(n_sub, dtype=float)
    ben_positions = np.searchsorted(subset_idx, ben_idx)
    weights[ben_positions] = ben_weights

    w_met = weighted_metrics(sub_labels, sub_preds, weights)
    uw_met = unweighted_metrics(sub_labels, sub_preds)

    sub_typs = typologies[subset_idx]
    ill_typs = sub_typs[sub_labels == 1]
    n_typs = len(set(int(t) for t in ill_typs if t >= 0))
    kl_typ = kl_div(typology_dist(sub_typs, sub_labels), ref_typ_dist)

    typ_counts = Counter(int(t) for t in ill_typs if t >= 0)
    typ_dict = {f"typ_{i}": typ_counts.get(i, 0) for i in TYPOLOGY_NAMES}

    irt = check_irt(data, difficulty, subset_idx[sub_labels == 1])
    ctx = estimate_context_size(n_sub, tok_stats, fmt="edge_list")

    return dict(
        n_total=n_sub, n_illicit=len(ill_idx), n_benign=len(ben_idx),
        w_precision=w_met["precision"], w_recall=w_met["recall"],
        w_f1=w_met["f1"],
        delta_p_w=abs(w_met["precision"] - full_metrics["precision"]),
        delta_r_w=abs(w_met["recall"] - full_metrics["recall"]),
        delta_f1_w=abs(w_met["f1"] - full_metrics["f1"]),
        uw_precision=uw_met["precision"], uw_recall=uw_met["recall"],
        uw_f1=uw_met["f1"],
        delta_p_uw=abs(uw_met["precision"] - full_metrics["precision"]),
        delta_r_uw=abs(uw_met["recall"] - full_metrics["recall"]),
        delta_f1_uw=abs(uw_met["f1"] - full_metrics["f1"]),
        n_typology_classes=n_typs, kl_typology=float(kl_typ),
        **typ_dict,
        irt_level=irt["level"], irt_diff_std=float(irt["diff_std"]),
        context=ctx,
    )


def stability_check(data, labels, typologies, difficulty, ens_probs,
                    threshold, full_metrics, ill_idx, ben_pool,
                    n_ben_select, ref_typ_dist, tok_stats,
                    hard_neg_ratio, k_repeats, seed_base) -> dict:
    """Repeat one benign budget ``k_repeats`` times; aggregate mean, std, range.

    Returns the aggregate together with the raw per-draw rows, which the draws
    CSV needs for the variance columns in the ablation table.
    """
    results, breakdowns = [], []
    for i in range(k_repeats):
        rng = np.random.default_rng(seed_base + i * 7919)
        ben_sel, ben_w, bkdn = sample_benign_stratified(
            rng, ben_pool, ens_probs, threshold, n_ben_select,
            hard_neg_ratio=hard_neg_ratio, difficulty=difficulty)
        r = evaluate_subset(data, labels, typologies, difficulty,
                            ens_probs, threshold, full_metrics,
                            ill_idx, ben_sel, ben_w, ref_typ_dist, tok_stats)
        r["draw_id"] = i
        results.append(r)
        breakdowns.append(bkdn)

    keys = ["w_precision", "w_recall", "w_f1",
            "delta_p_w", "delta_r_w", "delta_f1_w",
            "uw_precision", "uw_recall", "uw_f1",
            "delta_p_uw", "delta_r_uw", "delta_f1_uw"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
        agg[f"{k}_min"] = float(np.min(vals))
        agg[f"{k}_max"] = float(np.max(vals))

    agg["k_repeats"] = k_repeats
    for k in ("n_total", "n_illicit", "n_benign", "irt_level",
              "n_typology_classes"):
        agg[k] = results[0][k]
    agg["kl_typology_mean"] = float(np.mean([r["kl_typology"] for r in results]))
    agg["kl_typology_std"] = float(np.std([r["kl_typology"] for r in results]))
    # Illicit counts are deterministic: Stage 1 keeps every illicit edge.
    for i in TYPOLOGY_NAMES:
        agg[f"typ_{i}"] = results[0].get(f"typ_{i}", 0)
    agg["context"] = results[0]["context"]
    agg["ben_breakdown"] = breakdowns[0]
    agg["raw_draws"] = results
    return agg


# ─────────────────────────────────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────────────────────────────────

_AGG_CSV_COLS = [
    "dataset", "n_benign", "n_total", "n_illicit",
    "illicit_rate_pct", "full_illicit_rate_pct", "benign_iw",
    "reduction_pct", "feasible",
    "w_precision_mean", "w_precision_std",
    "w_recall_mean", "w_recall_std",
    "w_f1_mean", "w_f1_std",
    "delta_p_w_mean", "delta_p_w_std",
    "delta_r_w_mean", "delta_r_w_std",
    "delta_f1_w_mean", "delta_f1_w_std",
    "uw_precision_mean", "uw_precision_std",
    "uw_recall_mean", "uw_recall_std",
    "uw_f1_mean", "uw_f1_std",
    "delta_p_uw_mean", "delta_p_uw_std",
    "delta_r_uw_mean", "delta_r_uw_std",
    "delta_f1_uw_mean", "delta_f1_uw_std",
    "kl_typology_mean", "kl_typology_std",
    "n_typology_classes", "irt_level",
    "typ_0", "typ_1", "typ_2", "typ_3",
    "typ_4", "typ_5", "typ_6", "typ_7",
    "tok_edge_list", "tok_json",
    "ben_n_hard", "ben_n_fill",
]

_DRAW_CSV_COLS = [
    "dataset", "n_benign", "draw_id",
    "w_precision", "w_recall", "w_f1",
    "delta_p_w", "delta_r_w", "delta_f1_w",
    "uw_precision", "uw_recall", "uw_f1",
    "delta_p_uw", "delta_r_uw", "delta_f1_uw",
    "kl_typology", "n_typology_classes", "irt_level",
    "typ_0", "typ_1", "typ_2", "typ_3",
    "typ_4", "typ_5", "typ_6", "typ_7",
]


def save_aggregate_csv(path: Path, dataset: str, results: list,
                       full_met: dict, n_full: int) -> None:
    """One row per benign budget, prefixed by the full-split reference row."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_AGG_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerow(dict(
            dataset=dataset, n_benign=n_full, n_total=n_full,
            n_illicit="(full)", reduction_pct=0.0, feasible="ref",
            w_precision_mean=full_met["precision"],
            w_recall_mean=full_met["recall"],
            w_f1_mean=full_met["f1"],
        ))
        for r in results:
            row = dict(dataset=dataset, **r)
            bk = r.get("ben_breakdown") or {}
            row["ben_n_hard"] = bk.get("hard_neg", "")
            row["ben_n_fill"] = bk.get("n_fill", "")
            w.writerow(row)


def save_draws_csv(path: Path, dataset: str, all_sizes_agg: list) -> None:
    """One row per (benign budget, draw), for the variance columns."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_DRAW_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for size_agg in all_sizes_agg:
            for draw in size_agg.get("raw_draws", []):
                # n_benign is already on the draw row, from evaluate_subset.
                w.writerow(dict(dataset=dataset, **draw))


def save_comparison_csv(path: Path, dataset: str, n_full: int,
                        full_met: dict, ht_best: Optional[dict],
                        naive_summary: Optional[dict]) -> None:
    """Naive Coreset versus HT-Coreset, one row per method, for the appendix table."""
    rows = [dict(
        dataset=dataset, method="full_set",
        n_total=n_full, reduction_pct=0.0,
        precision=full_met["precision"], recall=full_met["recall"],
        f1=full_met["f1"],
        delta_p=0.0, delta_r=0.0, delta_f1=0.0,
        tracks_pr="yes",
    )]

    if naive_summary:
        for sz in naive_summary.get("sizes", []):
            rows.append(dict(
                dataset=dataset,
                method=f"naive_geometric_{sz['size']}",
                n_total=sz["size"],
                reduction_pct=(1 - sz["size"] / n_full) * 100,
                f1=sz.get("subset_f1", 0),
                delta_f1=sz.get("delta_f1", 0),
                kl_typology=sz.get("kl_typology", 0),
                kl_difficulty=sz.get("kl_difficulty", 0),
                irt_level=sz.get("irt_level", ""),
                tracks_pr="no",
            ))
        # The Naive search ran exhaustively on HI-Small and under Optuna on
        # LI-Small; whichever is present is that dataset's Naive result.
        naive_best = naive_summary.get("exhaustive") or naive_summary.get("optuna")
        if naive_best:
            rows.append(dict(
                dataset=dataset, method="naive_best",
                n_total=naive_best.get("n_total", 0),
                reduction_pct=(1 - naive_best.get("n_total", n_full) / n_full) * 100,
                f1=naive_best.get("subset_f1", 0),
                delta_f1=naive_best.get("delta_f1", 0),
                tracks_pr="no",
            ))

    if ht_best:
        rows.append(dict(
            dataset=dataset, method="ht_coreset",
            n_total=ht_best["n_total"],
            reduction_pct=ht_best["reduction_pct"],
            precision=ht_best["w_precision_mean"],
            recall=ht_best["w_recall_mean"],
            f1=ht_best["w_f1_mean"],
            delta_p=ht_best["delta_p_w_mean"],
            delta_r=ht_best["delta_r_w_mean"],
            delta_f1=ht_best["delta_f1_w_mean"],
            precision_std=ht_best.get("w_precision_std", 0),
            recall_std=ht_best.get("w_recall_std", 0),
            f1_std=ht_best.get("w_f1_std", 0),
            tracks_pr="yes",
        ))

    all_keys = sorted(set(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────
#  Construction driver
# ─────────────────────────────────────────────────────────────────────────

def build_ht_coreset(dataset: str, out_dir: Path,
                     members: Sequence[str] = CONSTRUCTION_MEMBERS,
                     seeds: Sequence[int] = SEEDS,
                     archive: Optional[Path] = None,
                     seed_rng: int = 0,
                     workers: int = DEFAULT_WORKERS,
                     n_ben_sizes: int = 12,
                     hard_neg_ratio: float = HARD_NEG_RATIO,
                     metric_tol: float = 0.05,
                     k_repeats: int = 50) -> dict:
    """Sweep benign budgets for one dataset and save the smallest feasible draw.

    A budget is feasible when the mean HT-weighted |dP|, |dR| and |dF1| against
    the full split are all at or below ``metric_tol``. The sweep is ascending, so
    the first feasible budget is the smallest one.
    """
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*75}")
    print(f"  HT-Coreset construction - {dataset}")
    print(f"  Started: {ts}")
    print(f"{'='*75}")

    out_dir = paths.ensure(Path(out_dir))

    data = load_test_data(dataset, members, seeds, archive=archive,
                          workers=workers)
    labels = data["labels"]
    typologies = data["typologies"]
    n = data["n"]

    n_ill = int(labels.sum())
    n_ben = n - n_ill
    ill_rate = n_ill / n

    print(f"  Full test set : {n:,} edges  "
          f"({n_ill:,} illicit {ill_rate*100:.3f}%,  {n_ben:,} benign)")

    full_typs = Counter(int(t) for t in typologies[labels == 1] if t >= 0)
    print(f"  Typed illicit : {sum(full_typs.values()):,} "
          f"({len(full_typs)} typology classes)")
    for t, cnt in sorted(full_typs.items()):
        print(f"    [{t}] {TYPOLOGY_NAMES.get(t, '?'):<20s} {cnt:4d}")

    print("\n  Computing ensemble soft-average ...")
    ens_probs = ensemble_soft_avg(data)
    full_thresh, _ = find_threshold(labels, ens_probs)
    full_preds = (ens_probs >= full_thresh).astype(int)
    full_met = unweighted_metrics(labels, full_preds)
    print(f"  Ensemble (full test): threshold={full_thresh:.2f}  "
          f"P={full_met['precision']*100:.2f}%  "
          f"R={full_met['recall']*100:.2f}%  F1={full_met['f1']*100:.2f}%")

    np.save(out_dir / f"ensemble_probs_{dataset}.npy", ens_probs)

    difficulty = compute_difficulty(data)
    diff_ill = difficulty[labels == 1]
    print(f"  Difficulty (illicit): mean={diff_ill.mean():.3f}  "
          f"std={diff_ill.std():.3f}")

    ref_typ_dist = typology_dist(typologies, labels)
    tok_stats = token_stats()

    ill_idx = np.where(labels == 1)[0]      # Stage 1: the whole minority class
    ben_pool = np.where(labels == 0)[0]

    hard_cut = full_thresh * HARD_NEG_THRESHOLD_FRAC
    n_hard = int((ens_probs[ben_pool] >= hard_cut).sum())
    print(f"\n  Benign pool: {len(ben_pool):,} edges")
    print(f"  Hard negatives (prob >= {hard_cut:.2f}): {n_hard:,}  "
          f"({n_hard/len(ben_pool)*100:.2f}%)")
    print(f"  Strategy: {hard_neg_ratio*100:.0f}% of the budget to hard "
          "negatives, the rest difficulty-stratified")

    ben_min = max(MIN_BENIGN_BUDGET, int(n_ill * BENIGN_MULTIPLIER))
    ben_sizes = np.unique(
        np.round(np.geomspace(ben_min, n_ben, n_ben_sizes)).astype(int))

    print(f"\n  Evaluating {len(ben_sizes)} benign budgets "
          f"(k={k_repeats} draws each, HT-weighted metrics)")
    print(f"  All illicit edges kept: {n_ill:,}")
    print(f"  Tolerance: |dP|, |dR|, |dF1| <= {metric_tol*100:.0f}%")
    print(f"\n{'-'*104}")
    print(f"  {'N-Ben':>8} {'Total':>9} {'Reduce':>7} | "
          f"{'wP':>6} {'dP':>6} {'wR':>6} {'dR':>6} {'wF1':>6} {'dF1':>6} | "
          f"{'uwP':>6} {'uwR':>6} {'uwF1':>6} | {'KL-t':>6} {'Tok(EL)':>9}")
    print(f"{'-'*104}")

    results, all_aggs, best_feasible = [], [], None

    for n_ben_target in ben_sizes:
        t_step = time.time()
        agg = stability_check(
            data, labels, typologies, difficulty, ens_probs,
            full_thresh, full_met, ill_idx, ben_pool,
            int(n_ben_target), ref_typ_dist, tok_stats,
            hard_neg_ratio, k_repeats, seed_rng)
        elapsed = time.time() - t_step

        total = agg["n_total"]
        reduction = (1.0 - total / n) * 100
        tok_el = agg["context"]["all_fmt_tokens"]["edge_list"]

        dp, dr, df = (agg["delta_p_w_mean"], agg["delta_r_w_mean"],
                      agg["delta_f1_w_mean"])
        feasible = (dp <= metric_tol and dr <= metric_tol and df <= metric_tol)
        bk = agg.get("ben_breakdown", {})

        print(f" {'*' if feasible else ' '}{n_ben_target:>7,} {total:>9,} "
              f"{reduction:>6.1f}% | "
              f"{agg['w_precision_mean']*100:>5.1f}% {dp*100:>5.1f}% "
              f"{agg['w_recall_mean']*100:>5.1f}% {dr*100:>5.1f}% "
              f"{agg['w_f1_mean']*100:>5.1f}% {df*100:>5.1f}% | "
              f"{agg['uw_precision_mean']*100:>5.1f}% "
              f"{agg['uw_recall_mean']*100:>5.1f}% "
              f"{agg['uw_f1_mean']*100:>5.1f}% | "
              f"{agg['kl_typology_mean']:>5.3f} {tok_el/1e6:>7.1f}M"
              f"  [{elapsed:.1f}s]")
        print(f"    benign: {bk.get('hard_neg', 0):,} hard-neg + "
              f"{sum(bk.get('easy', [])):,} tercile + "
              f"{bk.get('n_fill', 0):,} fill = {int(n_ben_target):,}"
              f"  | illicit share {n_ill / total * 100:.3f}%"
              f"  | benign weight x{n_ben / int(n_ben_target):.1f}")

        skip = {"context", "raw_draws", "ben_breakdown",
                "n_total", "n_benign", "n_illicit"}
        row = dict(
            n_benign=int(n_ben_target), n_total=total,
            n_illicit=agg.get("n_illicit", n_ill),
            illicit_rate_pct=n_ill / total * 100,
            full_illicit_rate_pct=ill_rate * 100,
            benign_iw=n_ben / int(n_ben_target),
            reduction_pct=reduction, feasible=feasible,
            ben_breakdown=bk,
            **{k: agg[k] for k in agg if k not in skip},
            tok_edge_list=tok_el,
            tok_json=agg["context"]["all_fmt_tokens"]["json"],
        )
        results.append(row)
        all_aggs.append(dict(n_benign=int(n_ben_target),
                             raw_draws=agg.get("raw_draws", [])))

        if feasible and best_feasible is None:
            best_feasible = row

    print(f"{'-'*104}")
    print(f"  Full-set reference: P={full_met['precision']*100:.2f}%  "
          f"R={full_met['recall']*100:.2f}%  F1={full_met['f1']*100:.2f}%")

    if best_feasible and best_feasible.get("w_precision_std", 0) < 1e-6:
        print("\n  NOTE: the weighted metrics have near-zero variance because "
              "every illicit edge is kept, so TP and FN are deterministic; "
              f"all {n_hard:,} hard negatives fit inside the budget, so FP is "
              "deterministic too; and the easy benign edges are all predicted "
              "legit, so they move nothing. A scorer with a different threshold "
              "will show variance, and the weights keep the estimate unbiased.")

    if best_feasible is not None:
        b = best_feasible
        print(f"\n  Smallest feasible subset (|dP|,|dR|,|dF1| <= "
              f"{metric_tol*100:.0f}%)")
        print(f"    Size        = {b['n_total']:,}  "
              f"(illicit {b['n_illicit']:,} + benign {b['n_benign']:,})")
        print(f"    Reduction   = {b['reduction_pct']:.1f}%  "
              f"({n:,} -> {b['n_total']:,})")
        print(f"    Weighted P  = {b['w_precision_mean']*100:.2f}% +/- "
              f"{b['w_precision_std']*100:.2f}%  "
              f"(dP = {b['delta_p_w_mean']*100:.2f}%)")
        print(f"    Weighted R  = {b['w_recall_mean']*100:.2f}% +/- "
              f"{b['w_recall_std']*100:.2f}%  "
              f"(dR = {b['delta_r_w_mean']*100:.2f}%)")
        print(f"    Weighted F1 = {b['w_f1_mean']*100:.2f}% +/- "
              f"{b['w_f1_std']*100:.2f}%  "
              f"(dF1 = {b['delta_f1_w_mean']*100:.2f}%)")
        print(f"    Tokens      = {b['tok_edge_list']/1e6:.1f}M edge_list, "
              f"{b['tok_json']/1e6:.1f}M json")
        print("    Illicit by typology:")
        for i in TYPOLOGY_NAMES:
            cnt = b.get(f"typ_{i}", 0)
            full_cnt = full_typs.get(i, 0)
            pct = (cnt / full_cnt * 100) if full_cnt > 0 else 0.0
            print(f"      [{i}] {TYPOLOGY_NAMES[i]:<20s} "
                  f"{cnt:4d} / {full_cnt:4d}  ({pct:5.1f}%)")

        # The canonical draw: the same RNG as draw 0 of the stability check.
        rng = np.random.default_rng(seed_rng)
        ben_sel, ben_w, _ = sample_benign_stratified(
            rng, ben_pool, ens_probs, full_thresh, b["n_benign"],
            hard_neg_ratio=hard_neg_ratio, difficulty=difficulty)
        best_idx = np.sort(np.concatenate([ill_idx, ben_sel]))
        best_weights = np.ones(len(best_idx), dtype=float)
        best_weights[np.searchsorted(best_idx, ben_sel)] = ben_w

        np.save(out_dir / "ht_subset_indices.npy", best_idx)
        np.save(out_dir / "ht_weights.npy", best_weights)
        print(f"    Saved indices : {out_dir / 'ht_subset_indices.npy'}")
        print(f"    Saved weights : {out_dir / 'ht_weights.npy'}  "
              f"(sum {best_weights.sum():,.0f}, population {n:,})")
    else:
        print("\n  No feasible subset within tolerance. Raise --metric-tol or "
              "--n-ben-sizes.")

    naive_path = out_dir / f"naive_coreset_summary_{dataset}.json"
    naive_summary = None
    if naive_path.exists():
        with open(naive_path) as f:
            naive_summary = json.load(f)
        naive_best = naive_summary.get("exhaustive") or naive_summary.get("optuna")
        if naive_best and best_feasible:
            print(f"\n  Naive Coreset: {naive_best.get('n_total', 0):,} edges  "
                  f"F1={naive_best.get('subset_f1', 0)*100:.2f}% (tracks F1 only)")
            print(f"  HT-Coreset   : {best_feasible['n_total']:,} edges  "
                  f"F1={best_feasible['w_f1_mean']*100:.2f}% "
                  f"P={best_feasible['w_precision_mean']*100:.2f}% "
                  f"R={best_feasible['w_recall_mean']*100:.2f}% "
                  "(tracks P, R and F1)")

    print("\n  Building the IRT response matrix (full test set) ...")
    R_full = build_response_matrix(data, ill_idx)
    beta = irt_difficulty_1pl(R_full)
    alpha = irt_discrimination(R_full)
    print(f"  IRT beta: {beta.mean():.3f} +/- {beta.std():.3f}   "
          f"rater discrimination alpha: {alpha.mean():.3f} +/- {alpha.std():.3f}")

    agg_csv_path = out_dir / f"ht_coreset_{dataset}_sizes.csv"
    save_aggregate_csv(agg_csv_path, dataset, results, full_met, n)
    draws_csv_path = out_dir / f"ht_coreset_{dataset}_draws.csv"
    save_draws_csv(draws_csv_path, dataset, all_aggs)
    comp_csv_path = out_dir / f"naive_vs_ht_{dataset}.csv"
    save_comparison_csv(comp_csv_path, dataset, n, full_met,
                        best_feasible, naive_summary)

    elapsed_total = time.time() - t0
    summary = dict(
        method="HT-Coreset",
        dataset=dataset,
        n_total=n, n_illicit=n_ill, n_benign=n_ben,
        illicit_rate=ill_rate,
        n_raters=data["n_raters"],
        members=list(members), seeds=list(seeds),
        ensemble_threshold=full_thresh,
        full_metrics=full_met,
        hard_neg_ratio=hard_neg_ratio,
        metric_tol=metric_tol,
        k_repeats=k_repeats,
        serializer=dict(
            avg_edges_per_case=common.AVG_EDGES_PER_CASE,
            tokens_per_case={fmt: s["n_tokens"] for fmt, s in tok_stats.items()},
        ),
        irt_full=dict(
            beta_mean=float(beta.mean()), beta_std=float(beta.std()),
            alpha_mean=float(alpha.mean()), alpha_std=float(alpha.std()),
        ),
        best_feasible=best_feasible,
        all_sizes=results,
        timing=dict(started=ts,
                    finished=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    elapsed_seconds=round(elapsed_total, 1)),
        output_files=dict(
            aggregate_csv=str(agg_csv_path),
            draws_csv=str(draws_csv_path),
            comparison_csv=str(comp_csv_path),
        ),
    )

    json_path = out_dir / f"ht_coreset_summary_{dataset}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary : {json_path}")
    print(f"  Elapsed: {elapsed_total:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the HT-Coreset and sweep the benign budget")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--members", nargs="+", default=list(CONSTRUCTION_MEMBERS),
                    help="ensemble members, by their paper names")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--archive", type=Path, default=None,
                    help="run directory outputs/; defaults to AMLC_ARCHIVE")
    ap.add_argument("--out", type=Path, default=paths.results() / "coreset")
    ap.add_argument("--seed-rng", type=int, default=0)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--n-ben-sizes", type=int, default=12,
                    help="benign budgets on the geometric grid")
    ap.add_argument("--hard-neg-ratio", type=float, default=HARD_NEG_RATIO)
    ap.add_argument("--metric-tol", type=float, default=0.05,
                    help="max |dP|, |dR|, |dF1| for a budget to count as feasible")
    ap.add_argument("--k-repeats", type=int, default=50,
                    help="random draws per budget")
    args = ap.parse_args()

    summaries = {}
    for ds in args.datasets:
        summaries[ds] = build_ht_coreset(
            dataset=ds, out_dir=Path(args.out) / ds,
            members=args.members, seeds=args.seeds, archive=args.archive,
            seed_rng=args.seed_rng, workers=args.workers,
            n_ben_sizes=args.n_ben_sizes, hard_neg_ratio=args.hard_neg_ratio,
            metric_tol=args.metric_tol, k_repeats=args.k_repeats)

    print(f"\n{'='*75}")
    print("  HT-Coreset summary")
    print(f"{'='*75}")
    for ds, s in summaries.items():
        fm = s["full_metrics"]
        b = s.get("best_feasible")
        print(f"\n  {ds}: full {s['n_total']:,} edges  "
              f"P={fm['precision']*100:.2f}% R={fm['recall']*100:.2f}% "
              f"F1={fm['f1']*100:.2f}%")
        if b:
            print(f"    HT-Coreset {b['n_total']:,} edges "
                  f"(-{b['reduction_pct']:.1f}%)  "
                  f"wP={b['w_precision_mean']*100:.2f}% "
                  f"wR={b['w_recall_mean']*100:.2f}% "
                  f"wF1={b['w_f1_mean']*100:.2f}%  "
                  f"tokens {b['tok_edge_list']/1e6:.1f}M edge_list")
        else:
            print("    no feasible subset")

    out_root = paths.ensure(Path(args.out))
    combined = out_root / "ht_coreset_summary_all.json"
    with open(combined, "w") as f:
        json.dump(dict(datasets=list(summaries), summaries=summaries), f, indent=2)
    print(f"\n  Saved combined : {combined}")


if __name__ == "__main__":
    main()
