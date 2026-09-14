"""Naive Coreset: proportional stratified subsampling, searched for a minimum.

This is the baseline the HT-Coreset is measured against, and the paper's
Naive-versus-HT appendix table is read straight out of the summary JSON this
module writes. It samples illicit edges stratified by typology and difficulty,
adds benign edges in the population ratio, and searches for the smallest subset
that still satisfies four constraints at once:

    IRT eligibility GOOD          enough illicit items, raters and spread
    KL typology  <= 0.20          typology mix close to the full split
    KL difficulty <= 0.15         difficulty mix close to the full split
    |dF1| <= 0.05                 ensemble F1 close to the full split
    every typology class present

There is no importance weighting here, so only F1 is tracked; precision and
recall stay biased by the subset's class ratio. That is the point of the
comparison: the minimum feasible Naive subset is 641,389 edges on HI-Small
(1.6x reduction) against 3,753 for the HT-Coreset.

Two searches, both load-bearing. HI-Small's published row comes from the
exhaustive search (binary search, then a dense ascending scan, 6M evaluations);
LI-Small's comes from the Optuna search (1,000 trials). Whichever ran is the
key present under ``exhaustive`` or ``optuna`` in the summary, and every
consumer reads them in that order.

Reads (only with the archive configured):

    <archive>/test_probs/<dataset>/     labels, typologies, scorer probabilities

Writes, under ``<out>/<dataset>/``:

    ensemble_probs_<dataset>.npy            soft-average scorer probability
    naive_subset_<dataset>_n<size>.npy      one draw per geometric size
    naive_subset_<dataset>_optuna_best.npy  the Optuna minimum, when run
    naive_subset_<dataset>_exhaustive_best.npy  the exhaustive minimum, when run
    naive_coreset_summary_<dataset>.json    the sweep, both searches, IRT summary

Usage:
    # the released HI-Small run
    python -m amlc.coreset.naive_coreset --datasets HI-Small \\
        --search-mode exhaustive --k-per-point 1000000
    # the released LI-Small run
    python -m amlc.coreset.naive_coreset --datasets LI-Small \\
        --search-mode optuna --optuna-trials 1000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score

from .. import paths
from ..config import DATASETS, CONSTRUCTION_MEMBERS, SEEDS
from .common import (
    AVG_EDGES_PER_CASE,
    DEFAULT_WORKERS,
    IRT_GOOD_ILLICIT,
    IRT_MIN_ILLICIT,
    TYPOLOGY_NAMES,
    build_response_matrix,
    check_irt,
    compute_difficulty,
    difficulty_dist,
    ensemble_soft_avg,
    estimate_context_size,
    find_threshold,
    irt_difficulty_1pl,
    irt_discrimination,
    kl_div,
    load_test_data,
    token_stats,
    typology_dist,
)

#: Serialiser format quoted in the printed tables and in the token headline.
DISPLAY_FMT = "edge_list"


def subset_f1(labels: np.ndarray, ens_probs: np.ndarray,
              idx: np.ndarray, threshold: float) -> float:
    """Ensemble F1 on a subset, scored at the full-split threshold."""
    preds = (ens_probs[idx] >= threshold).astype(int)
    return float(f1_score(labels[idx], preds, zero_division=0))


# ─────────────────────────────────────────────────────────────────────────
#  Proportional stratified sampling
# ─────────────────────────────────────────────────────────────────────────

def stratify_illicit(data: dict, difficulty: np.ndarray,
                     n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n_target`` illicit edges, stratified by typology x difficulty.

    Strata are (typology, difficulty tercile) pairs; each gets an equal base
    allocation and the remainder is filled from the edges no stratum drew.
    Returns global test-split indices.
    """
    labels = data["labels"]
    typologies = data["typologies"]
    ill_idx = np.where(labels == 1)[0]

    if n_target >= len(ill_idx):
        return ill_idx

    diff_ill = difficulty[ill_idx]
    q33, q67 = np.percentile(diff_ill, [33, 67])
    dbins = np.where(diff_ill < q33, 0, np.where(diff_ill < q67, 1, 2))
    typs = typologies[ill_idx]

    strata_keys = typs.astype(np.int32) * 3 + dbins.astype(np.int32)
    unique_keys = np.unique(strata_keys)
    base_per = max(1, n_target // len(unique_keys))

    selected_parts, spare_parts = [], []
    for key in unique_keys:
        members = np.where(strata_keys == key)[0]
        n_take = min(base_per, len(members))
        chosen = rng.choice(members, n_take, replace=False)
        selected_parts.append(chosen)
        if len(members) > n_take:
            mask = np.ones(len(members), dtype=bool)
            mask[np.searchsorted(members, chosen)] = False
            spare_parts.append(members[mask])

    selected_local = (np.concatenate(selected_parts) if selected_parts
                      else np.array([], dtype=int))

    still_need = n_target - len(selected_local)
    if still_need > 0 and spare_parts:
        spare_pool = np.concatenate(spare_parts)
        extra = rng.choice(spare_pool, min(still_need, len(spare_pool)),
                           replace=False)
        selected_local = np.concatenate([selected_local, extra])

    return ill_idx[selected_local[:n_target].astype(int)]


def sample_subset(data: dict, difficulty: np.ndarray,
                  n_illicit: int, rng: np.random.Generator) -> np.ndarray:
    """One Naive draw: stratified illicit plus benign at the population ratio.

    The benign side is stratified by difficulty terciles only. Because the
    illicit rate is about 0.1%, holding the ratio means a subset of the
    HT-Coreset's size would carry roughly four illicit edges, which is why this
    construction cannot go small.
    """
    labels = data["labels"]
    n_total = data["n"]
    n_ill_total = int(labels.sum())
    n_ben_total = n_total - n_ill_total
    illicit_rate = n_ill_total / n_total

    ill_sel = stratify_illicit(data, difficulty, n_illicit, rng)
    n_ill_actual = len(ill_sel)

    n_ben_target = min(int(n_ill_actual / illicit_rate) - n_ill_actual,
                       n_ben_total)
    ben_pool = np.where(labels == 0)[0]

    diff_ben = difficulty[ben_pool]
    q33, q67 = np.percentile(diff_ben, [33, 67])
    ben_parts = []
    for lo, hi in [(0.0, q33), (q33, q67), (q67, 1.1)]:
        mask = (diff_ben >= lo) & (diff_ben < hi)
        pool_b = ben_pool[mask]
        n_b = int(n_ben_target * mask.mean())
        if len(pool_b) > 0 and n_b > 0:
            ben_parts.append(rng.choice(pool_b, min(n_b, len(pool_b)),
                                        replace=False))

    ben_sel = (np.concatenate(ben_parts) if ben_parts
               else np.array([], dtype=int))

    still_need = n_ben_target - len(ben_sel)
    if still_need > 0:
        used_mask = np.zeros(n_total, dtype=bool)
        if len(ben_sel) > 0:
            used_mask[ben_sel] = True
        spare = ben_pool[~used_mask[ben_pool]]
        if len(spare) > 0:
            extra = rng.choice(spare, min(still_need, len(spare)),
                               replace=False)
            ben_sel = np.concatenate([ben_sel, extra])

    return np.sort(np.concatenate([ill_sel, ben_sel.astype(int)]).astype(int))


# ─────────────────────────────────────────────────────────────────────────
#  Optuna constrained minimum search
# ─────────────────────────────────────────────────────────────────────────

def optuna_search(
    dataset: str,
    data: dict,
    labels: np.ndarray,
    typologies: np.ndarray,
    difficulty: np.ndarray,
    ens_probs: np.ndarray,
    full_f1: float,
    full_thresh: float,
    tok_stats: dict,
    ref_typ_dist: np.ndarray,
    ref_diff_dist: np.ndarray,
    out_dir: Path,
    seed_rng: int = 0,
    n_trials: int = 100,
    kl_typ_thresh: float = 0.20,
    kl_dif_thresh: float = 0.15,
    f1_tol: float = 0.05,
) -> Optional[dict]:
    """Minimise subset size under the five constraints with a TPE sampler.

    The search space is the illicit count, log-uniform between the IRT-GOOD
    floor and the full illicit population. The winning trial's subset is redrawn
    with that trial's RNG, so the saved indices are the ones whose constraints
    were checked rather than a fresh sample that might not satisfy them.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_ill = int(labels.sum())
    n_typs_full = len(set(int(t) for t in typologies[labels == 1] if t >= 0))
    ill_min = min(IRT_GOOD_ILLICIT, n_ill)

    def objective(trial) -> float:
        n_ill_target = trial.suggest_int("n_illicit", ill_min, n_ill, log=True)
        trial_rng = np.random.default_rng(seed_rng + trial.number * 31337)
        subset_idx = sample_subset(data, difficulty, n_ill_target, trial_rng)

        sub_labels = labels[subset_idx]
        sub_typs = typologies[subset_idx]
        sub_diff = difficulty[subset_idx]
        n_sub = len(subset_idx)

        irt = check_irt(data, difficulty, subset_idx[sub_labels == 1])
        kl_typ = kl_div(typology_dist(sub_typs, sub_labels), ref_typ_dist)
        kl_dif = kl_div(difficulty_dist(sub_diff, sub_labels), ref_diff_dist)
        n_typs = len(set(int(t) for t in sub_typs[sub_labels == 1] if t >= 0))
        sub_f1 = subset_f1(labels, ens_probs, subset_idx, full_thresh)
        delta_f1 = abs(sub_f1 - full_f1)

        trial.set_user_attr("n_total", n_sub)
        trial.set_user_attr("n_illicit", int(sub_labels.sum()))
        trial.set_user_attr("irt_level", irt["level"])
        trial.set_user_attr("kl_typ", float(kl_typ))
        trial.set_user_attr("kl_dif", float(kl_dif))
        trial.set_user_attr("n_typs", n_typs)
        trial.set_user_attr("sub_f1", float(sub_f1))
        trial.set_user_attr("delta_f1", float(delta_f1))
        # Constraint values, satisfied when <= 0.
        trial.set_user_attr("constraints", [
            kl_typ - kl_typ_thresh,
            kl_dif - kl_dif_thresh,
            delta_f1 - f1_tol,
            0.0 if irt["level"] == "GOOD" else 1.0,
            float(n_typs_full - n_typs),
        ])
        return float(n_sub)

    sampler = optuna.samplers.TPESampler(
        seed=seed_rng,
        constraints_func=lambda t: t.user_attrs.get("constraints", [1.0]),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\n  Optuna search ({n_trials} trials)")
    print(f"    Constraints : IRT=GOOD | KL-typ <= {kl_typ_thresh} | "
          f"KL-dif <= {kl_dif_thresh} | |dF1| <= {f1_tol*100:.0f}% | "
          f"all {n_typs_full} typologies present")
    print(f"    Search space: n_illicit in [{ill_min}, {n_ill}] (log-uniform)")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    feasible = [t for t in study.trials
                if t.state.name == "COMPLETE"
                and all(c <= 0.0 for c in t.user_attrs.get("constraints", [1.0]))]
    if not feasible:
        print(f"  No feasible solution in {n_trials} trials. Relax "
              f"--kl-typ-thresh ({kl_typ_thresh}), --kl-dif-thresh "
              f"({kl_dif_thresh}) or --f1-tol ({f1_tol}).")
        return None

    best = min(feasible, key=lambda t: t.values[0])
    best_rng = np.random.default_rng(seed_rng + best.number * 31337)
    best_idx = sample_subset(data, difficulty, best.params["n_illicit"], best_rng)

    sub_labels = labels[best_idx]
    sub_typs = typologies[best_idx]
    n_sub = len(best_idx)
    irt = check_irt(data, difficulty, best_idx[sub_labels == 1])
    kl_typ = kl_div(typology_dist(sub_typs, sub_labels), ref_typ_dist)
    kl_dif = kl_div(difficulty_dist(difficulty[best_idx], sub_labels),
                    ref_diff_dist)
    n_typs = len(set(int(t) for t in sub_typs[sub_labels == 1] if t >= 0))
    sub_f1 = subset_f1(labels, ens_probs, best_idx, full_thresh)
    delta_f1 = abs(sub_f1 - full_f1)
    ctx = estimate_context_size(n_sub, tok_stats, fmt=DISPLAY_FMT)

    idx_path = out_dir / f"naive_subset_{dataset}_optuna_best.npy"
    np.save(idx_path, best_idx)

    print(f"  Optuna minimum ({len(feasible)} feasible of {n_trials} trials): "
          f"size={n_sub:,} illicit={int(sub_labels.sum())} "
          f"typs={n_typs}/{n_typs_full} IRT={irt['level']} "
          f"KL-typ={kl_typ:.4f} KL-dif={kl_dif:.4f} "
          f"F1={sub_f1*100:.2f}% dF1={delta_f1*100:.2f}%")
    print(f"    Saved index : {idx_path}")

    return dict(
        n_total=n_sub,
        n_illicit=int(sub_labels.sum()),
        n_typology_classes=n_typs,
        irt_level=irt["level"],
        irt_diff_std=float(irt["diff_std"]),
        kl_typology=float(kl_typ),
        kl_difficulty=float(kl_dif),
        subset_f1=float(sub_f1),
        delta_f1=float(delta_f1),
        n_trials=n_trials,
        n_feasible=len(feasible),
        kl_typ_thresh=kl_typ_thresh,
        kl_dif_thresh=kl_dif_thresh,
        f1_tol=f1_tol,
        context=ctx,
        best_trial_number=best.number,
        indices_path=str(idx_path),
    )


# ─────────────────────────────────────────────────────────────────────────
#  Exhaustive minimum search
# ─────────────────────────────────────────────────────────────────────────

# Populated by the parent before forking. Forked children inherit it through
# copy-on-write, so the 1M-edge arrays are never pickled onto the IPC pipe.
_exhaust_shared: dict = {}


def _exhaust_worker(args: tuple):
    """Evaluate one (illicit count, seed offset) in a forked child.

    Returns ``(seed_offset, subset_size, metrics)`` when every constraint holds
    and ``None`` otherwise. Constraints are checked cheapest first so an
    infeasible candidate exits before the F1 sweep.
    """
    n_target, seed_offset = args
    s = _exhaust_shared

    rng = np.random.default_rng(
        s["seed_rng"] + n_target * 100003 + seed_offset * 31337)
    idx = sample_subset(s["data"], s["difficulty"], n_target, rng)
    sub_labels = s["labels"][idx]
    sub_typs = s["typologies"][idx]
    sub_diff = s["difficulty"][idx]

    irt = check_irt(s["data"], s["difficulty"], idx[sub_labels == 1])
    if irt["level"] != "GOOD":
        return None

    kl_typ = kl_div(typology_dist(sub_typs, sub_labels), s["ref_typ_dist"])
    if kl_typ > s["kl_typ_thresh"]:
        return None

    kl_dif = kl_div(difficulty_dist(sub_diff, sub_labels), s["ref_diff_dist"])
    if kl_dif > s["kl_dif_thresh"]:
        return None

    n_typs = len(set(int(t) for t in sub_typs[sub_labels == 1] if t >= 0))
    if n_typs < s["n_typs_full"]:
        return None

    sub_f1 = subset_f1(s["labels"], s["ens_probs"], idx, s["full_thresh"])
    delta_f1 = abs(sub_f1 - s["full_f1"])
    if delta_f1 > s["f1_tol"]:
        return None

    return (seed_offset, len(idx), dict(
        kl_typ=float(kl_typ), kl_dif=float(kl_dif),
        sub_f1=float(sub_f1), delta_f1=float(delta_f1),
        n_total=len(idx), n_illicit=int(sub_labels.sum()),
        irt_level=irt["level"], irt_diff_std=float(irt["diff_std"]),
    ))


def exhaustive_search(
    dataset: str,
    data: dict,
    labels: np.ndarray,
    typologies: np.ndarray,
    difficulty: np.ndarray,
    ens_probs: np.ndarray,
    full_f1: float,
    full_thresh: float,
    tok_stats: dict,
    ref_typ_dist: np.ndarray,
    ref_diff_dist: np.ndarray,
    out_dir: Path,
    seed_rng: int = 0,
    k_per_point: int = 500,
    n_workers: Optional[int] = None,
    kl_typ_thresh: float = 0.20,
    kl_dif_thresh: float = 0.15,
    f1_tol: float = 0.05,
) -> Optional[dict]:
    """Three-phase search for the smallest feasible illicit count.

    Phase 1, geometric bisection at k=min(50, k_per_point) draws per probe,
    narrows the feasibility boundary in a logarithmic number of steps. Phase 2
    scans every integer in the narrowed range in ascending order at
    ``k_per_point`` draws and stops at the first feasible value, which is
    therefore the minimum. Phase 3 re-evaluates that value at five times the
    draws to pick the smallest feasible subset and to estimate how often a draw
    at the minimum is feasible at all (1.5e-05 on HI-Small).

    Parallelism is a fork pool rather than threads because the per-draw work is
    pure Python and numpy indexing under the GIL.
    """
    import multiprocessing as mp

    n_ill = int(labels.sum())
    n_typs_full = len(set(int(t) for t in typologies[labels == 1] if t >= 0))
    ill_min = min(IRT_GOOD_ILLICIT, n_ill)
    n_workers = n_workers or DEFAULT_WORKERS

    _exhaust_shared.update(dict(
        data=data, difficulty=difficulty, labels=labels,
        typologies=typologies, ens_probs=ens_probs,
        full_f1=full_f1, full_thresh=full_thresh,
        ref_typ_dist=ref_typ_dist, ref_diff_dist=ref_diff_dist,
        n_typs_full=n_typs_full, seed_rng=seed_rng,
        kl_typ_thresh=kl_typ_thresh, kl_dif_thresh=kl_dif_thresh,
        f1_tol=f1_tol,
    ))

    pool = mp.get_context("fork").Pool(n_workers)

    def _probe(n_target: int, k: int, base_offset: int = 0):
        tasks = [(n_target, base_offset + i) for i in range(k)]
        chunksize = max(1, k // (n_workers * 4))
        results = pool.map(_exhaust_worker, tasks, chunksize=chunksize)

        best_seed, best_size, best_metrics, n_feasible = None, float("inf"), {}, 0
        for res in results:
            if res is not None:
                seed_off, sz, metrics = res
                n_feasible += 1
                if sz < best_size:
                    best_seed, best_size, best_metrics = seed_off, sz, metrics
        return n_feasible, best_seed, best_metrics

    print(f"\n  Exhaustive search (k={k_per_point:,} draws/point, "
          f"{n_workers} fork workers)")
    print(f"    Constraints : IRT=GOOD | KL-typ <= {kl_typ_thresh} | "
          f"KL-dif <= {kl_dif_thresh} | |dF1| <= {f1_tol*100:.0f}% | "
          f"all {n_typs_full} typologies")
    print(f"    Search space: n_illicit in [{ill_min}, {n_ill}]")

    k1 = min(50, k_per_point)
    print(f"\n  Phase 1 - bisection (k={k1}/point)")
    lo, hi = ill_min, n_ill
    phase1_calls = 0
    while True:
        if hi - lo <= max(2, int((n_ill - ill_min) * 0.01)):
            break
        mid = max(lo + 1, min(hi - 1, int(np.sqrt(float(lo) * float(hi)))))
        n_feas, _, _ = _probe(mid, k1)
        phase1_calls += k1
        if n_feas > 0:
            print(f"    n_ill={mid:5d}: {n_feas:>2}/{k1} feasible  -> hi={mid}")
            hi = mid
        else:
            print(f"    n_ill={mid:5d}:  0/{k1} infeasible -> lo={mid}")
            lo = mid

    lo_scan = max(ill_min, int(lo * 0.95))
    hi_scan = min(n_ill, int(hi * 1.05) + 1)
    print(f"\n  Phase 2 - ascending scan over [{lo_scan}, {hi_scan}] "
          f"(k={k_per_point:,}/point); the first feasible value is the minimum")

    minimum_n_ill, minimum_seed, minimum_metrics = None, None, {}
    phase2_calls = 0
    for n_target in range(lo_scan, hi_scan + 1):
        n_feas, best_seed, best_metrics = _probe(
            n_target, k_per_point, base_offset=k1 + 1000)
        phase2_calls += k_per_point
        print(f"    n_ill={n_target:5d}: {n_feas:>{len(str(k_per_point))}}"
              f"/{k_per_point} feasible", flush=True)
        if n_feas > 0:
            minimum_n_ill, minimum_seed, minimum_metrics = (
                n_target, best_seed, best_metrics)
            break

    if minimum_n_ill is None:
        pool.close()
        pool.join()
        print("  No feasible solution in the dense scan. Relax the constraints "
              "or widen the range.")
        return None

    k3 = k_per_point * 5
    print(f"\n  Phase 3 - confirmation at n_ill={minimum_n_ill} (k={k3:,})")
    n_feas3, best_seed3, final_metrics = _probe(
        minimum_n_ill, k3, base_offset=k1 + k_per_point * 2 + 9999)
    p_feas3 = n_feas3 / k3
    print(f"    {n_feas3:,}/{k3:,} feasible (P(feasible) = {p_feas3*100:.3f}%)")
    if best_seed3 is not None:
        minimum_seed, minimum_metrics = best_seed3, final_metrics

    pool.close()
    pool.join()

    best_rng = np.random.default_rng(
        seed_rng + minimum_n_ill * 100003 + minimum_seed * 31337)
    minimum_idx = sample_subset(data, difficulty, minimum_n_ill, best_rng)

    n_sub = minimum_metrics.get("n_total", len(minimum_idx))
    n_typs_sub = len(set(
        int(t) for t in typologies[minimum_idx][labels[minimum_idx] == 1]
        if t >= 0))
    ctx = estimate_context_size(n_sub, tok_stats, fmt=DISPLAY_FMT)

    idx_path = out_dir / f"naive_subset_{dataset}_exhaustive_best.npy"
    np.save(idx_path, minimum_idx)

    total_calls = phase1_calls + phase2_calls + k3
    print(f"\n  Exhaustive minimum ({total_calls:,} evaluations): "
          f"n_illicit={minimum_n_ill} size={n_sub:,} "
          f"typs={n_typs_sub}/{n_typs_full} "
          f"IRT={minimum_metrics.get('irt_level', '?')} "
          f"F1={minimum_metrics.get('sub_f1', float('nan'))*100:.2f}% "
          f"dF1={minimum_metrics.get('delta_f1', float('nan'))*100:.2f}%")
    print(f"    Saved index : {idx_path}")

    return dict(
        n_total=n_sub,
        n_illicit=minimum_metrics.get("n_illicit",
                                      int(labels[minimum_idx].sum())),
        minimum_n_illicit=minimum_n_ill,
        n_typology_classes=n_typs_sub,
        irt_level=minimum_metrics.get("irt_level", "?"),
        irt_diff_std=float(minimum_metrics.get("irt_diff_std", float("nan"))),
        kl_typology=float(minimum_metrics.get("kl_typ", float("nan"))),
        kl_difficulty=float(minimum_metrics.get("kl_dif", float("nan"))),
        subset_f1=float(minimum_metrics.get("sub_f1", float("nan"))),
        delta_f1=float(minimum_metrics.get("delta_f1", float("nan"))),
        p_feasible=float(p_feas3),
        k_per_point=k_per_point,
        k_confirm=k3,
        total_evals=total_calls,
        kl_typ_thresh=kl_typ_thresh,
        kl_dif_thresh=kl_dif_thresh,
        f1_tol=f1_tol,
        context=ctx,
        indices_path=str(idx_path),
    )


# ─────────────────────────────────────────────────────────────────────────
#  Construction driver
# ─────────────────────────────────────────────────────────────────────────

def build_naive_coreset(dataset: str, out_dir: Path,
                        members: Sequence[str] = CONSTRUCTION_MEMBERS,
                        seeds: Sequence[int] = SEEDS,
                        archive: Optional[Path] = None,
                        n_sizes: int = 8,
                        seed_rng: int = 0,
                        workers: int = DEFAULT_WORKERS,
                        optuna_trials: int = 0,
                        kl_typ_thresh: float = 0.20,
                        kl_dif_thresh: float = 0.15,
                        f1_tol: float = 0.05,
                        search_mode: str = "optuna",
                        k_per_point: int = 500) -> dict:
    """Sweep geometric sizes for one dataset, then search for the minimum."""
    print(f"\n{'='*65}")
    print(f"  Naive Coreset - {dataset}")
    print(f"{'='*65}")

    out_dir = paths.ensure(Path(out_dir))

    data = load_test_data(dataset, members, seeds, archive=archive,
                          workers=workers)
    labels = data["labels"]
    typologies = data["typologies"]
    n = data["n"]

    n_ill = int(labels.sum())
    ill_rate = n_ill / n
    print(f"  Full test set : {n:,} edges  "
          f"({n_ill:,} illicit {ill_rate*100:.3f}%,  {n - n_ill:,} benign)")

    full_typs = Counter(int(t) for t in typologies[labels == 1] if t >= 0)
    print(f"  Typed illicit : {sum(full_typs.values()):,} "
          f"({len(full_typs)} typology classes)")
    for t, cnt in sorted(full_typs.items()):
        print(f"    [{t}] {TYPOLOGY_NAMES.get(t, '?'):<20s} {cnt:4d}")

    ens_probs = ensemble_soft_avg(data)
    full_thresh, full_f1 = find_threshold(labels, ens_probs)
    print(f"  Ensemble F1 (full test): {full_f1*100:.2f}% "
          f"at threshold {full_thresh:.2f}")
    np.save(out_dir / f"ensemble_probs_{dataset}.npy", ens_probs)

    difficulty = compute_difficulty(data)
    ref_typ_dist = typology_dist(typologies, labels)
    ref_diff_dist = difficulty_dist(difficulty, labels)
    tok_stats = token_stats()

    ill_min = max(10, min(50, n_ill))
    ill_sizes = np.unique(
        np.round(np.geomspace(ill_min, n_ill, n_sizes)).astype(int))
    rng = np.random.default_rng(seed_rng)

    print(f"\n{'-'*94}")
    print(f"  {'Size':>10} {'Illicit':>8} {'Typs':>5} {'IRT':>6} "
          f"{'KL-typ':>7} {'KL-dif':>7} {'SubF1':>7} {'dF1':>7} "
          f"{'Tok(EL)':>9} {'Tok(JSON)':>10}")
    print(f"{'-'*94}")

    results = []
    best_min_idx = None   # first size that is at least WEAK on IRT
    best_f1_idx = None    # first size within the F1 tolerance

    for idx_s, n_ill_target in enumerate(ill_sizes):
        subset_idx = sample_subset(data, difficulty, int(n_ill_target), rng)
        sub_labels = labels[subset_idx]
        sub_typs = typologies[subset_idx]
        n_sub = len(subset_idx)
        n_sub_ill = int(sub_labels.sum())

        irt = check_irt(data, difficulty, subset_idx[sub_labels == 1])
        kl_typ = kl_div(typology_dist(sub_typs, sub_labels), ref_typ_dist)
        kl_diff = kl_div(difficulty_dist(difficulty[subset_idx], sub_labels),
                         ref_diff_dist)
        n_typs_pres = len(set(int(t) for t in sub_typs[sub_labels == 1]
                              if t >= 0))
        sub_f1 = subset_f1(labels, ens_probs, subset_idx, full_thresh)
        delta_f1 = abs(sub_f1 - full_f1)
        ctx = estimate_context_size(n_sub, tok_stats, fmt=DISPLAY_FMT)
        f1_ok = delta_f1 < f1_tol

        if best_min_idx is None and irt["level"] in ("WEAK", "GOOD"):
            best_min_idx = idx_s
        if best_f1_idx is None and f1_ok and n_sub_ill >= IRT_MIN_ILLICIT:
            best_f1_idx = idx_s

        results.append(dict(
            size=n_sub, n_illicit=n_sub_ill,
            n_typology_classes=n_typs_pres,
            illicit_rate=float(sub_labels.mean()),
            kl_typology=kl_typ, kl_difficulty=kl_diff,
            subset_f1=sub_f1, delta_f1=delta_f1, f1_ok=f1_ok,
            irt=irt, context=ctx,
        ))

        print(f"  {n_sub:>10,} {n_sub_ill:>8,} {n_typs_pres:>5} "
              f"{irt['level']:>6} {kl_typ:>7.4f} {kl_diff:>7.4f} "
              f"{sub_f1*100:>6.1f}% {delta_f1*100:>6.1f}% "
              f"{ctx['all_fmt_tokens']['edge_list']/1e6:>7.1f}M "
              f"{ctx['all_fmt_tokens']['json']/1e6:>9.1f}M")

        np.save(out_dir / f"naive_subset_{dataset}_n{n_sub:09d}.npy", subset_idx)

    print(f"{'-'*94}")
    print(f"  Full-set ensemble F1 = {full_f1*100:.2f}%  "
          f"(tolerance +/-{f1_tol*100:.0f}%)")
    for r in results:
        issues = ", ".join(r["irt"]["issues"]) or "OK"
        print(f"  {r['size']:>10,}  IRT {r['irt']['level']:>4}  {issues}")

    optuna_result = None
    exhaustive_result = None
    search_kwargs = dict(
        dataset=dataset, data=data, labels=labels,
        typologies=typologies, difficulty=difficulty,
        ens_probs=ens_probs, full_f1=full_f1, full_thresh=full_thresh,
        tok_stats=tok_stats, ref_typ_dist=ref_typ_dist,
        ref_diff_dist=ref_diff_dist, out_dir=out_dir, seed_rng=seed_rng,
        kl_typ_thresh=kl_typ_thresh, kl_dif_thresh=kl_dif_thresh,
        f1_tol=f1_tol,
    )

    if search_mode in ("optuna", "both") and optuna_trials > 0:
        optuna_result = optuna_search(**search_kwargs, n_trials=optuna_trials)
    if search_mode in ("exhaustive", "both"):
        exhaustive_result = exhaustive_search(
            **search_kwargs, k_per_point=k_per_point, n_workers=workers)

    print("\n  Building the IRT response matrix (full test set) ...")
    R_full = build_response_matrix(data, np.where(labels == 1)[0])
    beta = irt_difficulty_1pl(R_full)
    alpha = irt_discrimination(R_full)
    print(f"  IRT beta: {beta.mean():.3f} +/- {beta.std():.3f}   "
          f"rater discrimination alpha: {alpha.mean():.3f} +/- {alpha.std():.3f}")

    summary = dict(
        method="Naive Coreset",
        dataset=dataset, n_total=n, n_illicit=n_ill,
        illicit_rate=ill_rate, n_raters=data["n_raters"],
        members=list(members), seeds=list(seeds),
        ensemble_f1_full=full_f1,
        ensemble_threshold=full_thresh,
        serializer=dict(
            avg_edges_per_case=AVG_EDGES_PER_CASE,
            display_format=DISPLAY_FMT,
            tokens_per_case={f: s["n_tokens"] for f, s in tok_stats.items()},
            chars_per_case={f: s["n_chars"] for f, s in tok_stats.items()},
        ),
        irt_full=dict(
            n_illicit=n_ill,
            beta_mean=float(beta.mean()), beta_std=float(beta.std()),
            alpha_mean=float(alpha.mean()), alpha_std=float(alpha.std()),
        ),
        best_f1_size=(results[best_f1_idx]["size"]
                      if best_f1_idx is not None else None),
        best_irt_size=(results[best_min_idx]["size"]
                       if best_min_idx is not None else None),
        optuna=optuna_result,
        exhaustive=exhaustive_result,
        sizes=[
            dict(size=r["size"], n_illicit=r["n_illicit"],
                 n_typology_classes=r["n_typology_classes"],
                 illicit_rate=r["illicit_rate"],
                 kl_typology=r["kl_typology"], kl_difficulty=r["kl_difficulty"],
                 subset_f1=r["subset_f1"], delta_f1=r["delta_f1"],
                 f1_ok=r["f1_ok"],
                 irt_level=r["irt"]["level"], irt_diff_std=r["irt"]["diff_std"],
                 irt_issues=r["irt"]["issues"],
                 n_context_graphs=r["context"]["n_context_graphs"],
                 tokens_by_format=r["context"]["all_fmt_tokens"])
            for r in results
        ],
    )

    json_path = out_dir / f"naive_coreset_summary_{dataset}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary : {json_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the Naive Coreset and search for its minimum size")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--members", nargs="+", default=list(CONSTRUCTION_MEMBERS),
                    help="ensemble members, by their paper names")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--archive", type=Path, default=None,
                    help="run directory outputs/; defaults to AMLC_ARCHIVE")
    ap.add_argument("--out", type=Path, default=paths.results() / "coreset")
    ap.add_argument("--n-sizes", type=int, default=8,
                    help="geometrically spaced sizes in the sweep")
    ap.add_argument("--seed-rng", type=int, default=0)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--search-mode", choices=["optuna", "exhaustive", "both"],
                    default="optuna",
                    help="HI-Small's published row used exhaustive, "
                         "LI-Small's used optuna")
    ap.add_argument("--optuna-trials", type=int, default=0,
                    help="0 disables the Optuna search; the released LI-Small "
                         "run used 1000")
    ap.add_argument("--k-per-point", type=int, default=500,
                    help="draws per candidate in the exhaustive scan; the "
                         "released HI-Small run used 1000000")
    ap.add_argument("--kl-typ-thresh", type=float, default=0.20)
    ap.add_argument("--kl-dif-thresh", type=float, default=0.15)
    ap.add_argument("--f1-tol", type=float, default=0.05)
    args = ap.parse_args()

    kwargs = [
        dict(dataset=ds, out_dir=Path(args.out) / ds,
             members=args.members, seeds=args.seeds, archive=args.archive,
             n_sizes=args.n_sizes, seed_rng=args.seed_rng,
             workers=args.workers, optuna_trials=args.optuna_trials,
             kl_typ_thresh=args.kl_typ_thresh,
             kl_dif_thresh=args.kl_dif_thresh, f1_tol=args.f1_tol,
             search_mode=args.search_mode, k_per_point=args.k_per_point)
        for ds in args.datasets
    ]

    summaries = {}
    # The exhaustive search already owns every core through its fork pool, so
    # datasets run one at a time in that mode to avoid nesting process pools.
    parallel = (len(args.datasets) > 1
                and args.search_mode not in ("exhaustive", "both"))
    if parallel:
        with ProcessPoolExecutor(max_workers=len(args.datasets)) as ex:
            futures = {ex.submit(build_naive_coreset, **kw): kw["dataset"]
                       for kw in kwargs}
            for fut in as_completed(futures):
                summaries[futures[fut]] = fut.result()
    else:
        for kw in kwargs:
            summaries[kw["dataset"]] = build_naive_coreset(**kw)

    print(f"\n{'='*75}")
    print("  Naive Coreset summary")
    print(f"{'='*75}")
    for ds, s in summaries.items():
        best = s.get("exhaustive") or s.get("optuna")
        line = (f"{best['n_total']:,} edges, "
                f"{best['n_illicit']:,} illicit, "
                f"dF1={best['delta_f1']*100:.1f}%, "
                f"{best['context']['est_tokens_total']/1e6:.0f}M tokens"
                if best else "no minimum found (no search was run)")
        print(f"  {ds}: full {s['n_total']:,} edges -> {line}")

    out_root = paths.ensure(Path(args.out))
    combined = out_root / "naive_coreset_summary_all.json"
    with open(combined, "w") as f:
        json.dump(dict(datasets=list(summaries), summaries=summaries), f,
                  indent=2)
    print(f"\n  Saved combined : {combined}")


if __name__ == "__main__":
    main()
