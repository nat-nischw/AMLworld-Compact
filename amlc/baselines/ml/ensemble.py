"""Combine the three supervised members into the ensemble the LLMs are measured against.

Reads the per-seed test probabilities each member wrote into the archived run
directory, combines them three ways, and writes the aggregate rows into
``results/summary_all.csv`` and ``results/summary_all.json`` beside the
per-model and LLM rows.

    hard majority vote      threshold each member at its own optimum, take >= 2
    soft average            mean the probabilities, then threshold
    weighted soft average   mean weighted by each member's oracle F1

Soft average is the ensemble behind the paper's supervised detection row. Per
seed, over the three members, it gives the 70.8 +/- 0.1 and 29.8 +/- 0.2 rows in
``results/summary_all.csv``. The 70.9 and 29.6 figures quoted for the full
temporal split come from the same rule applied one level up: the mean over all
three members and all five seeds, which is the ``ens_probs_<dataset>.npy`` the
coreset stage stratifies on, thresholded at the operating points in
:data:`amlc.config.ML_THRESHOLDS`. Both are the soft vote; they differ in
whether the seed average happens before or after scoring.

A two-member boosted-tree-only ensemble is also scored, because its rows are in
the published CSV.

Reads
    ``test_labels.npy``, ``test_typologies.npy`` and, per member and seed,
    ``seed_<seed>.npy`` and ``seed_<seed>_typ.npy`` from the archive.
Writes
    ``results/summary_all.csv`` and ``results/summary_all.json``, replacing any
    row with the same method and dataset and leaving every other row alone.

Usage::

    python -m amlc.baselines.ml.ensemble --datasets HI-Small
    python -m amlc.baselines.ml.ensemble --datasets HI-Small LI-Small
    python -m amlc.baselines.ml.ensemble --members LightGBM+GFP XGBoost+GFP

The pre-release docstring documented a ``--dataset`` flag; the parser has always
spelled it ``--datasets``, and the examples above are what actually runs.

Thresholds are selected by sweeping the full test labels. The released
operating points are stored in :data:`amlc.config.ML_THRESHOLDS`.

Dropped: the fallbacks to ``test_labels_pna.npy`` and
``test_typologies_pna.npy``. PNA used a different data loader and so a different
test set, and it appears in no table in the paper.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ... import paths
from ...config import DATASETS, ENSEMBLE_MEMBERS, SEEDS
from ...archive import legacy_path, member_dir
from ..metrics import find_optimal_threshold

#: Written into cells the supervised path does not produce. This exact
#: character, U+2014, is in the published summary_all.csv, so it is data rather
#: than prose and is spelled as an escape to keep it out of the source text.
EMPTY_CELL = "\u2014"


def load_probs(archive: Path, dataset: str, member: str,
               seeds: List[int]) -> Dict[int, np.ndarray]:
    """Per-seed test probabilities for one member, skipping seeds not on disk."""
    out = {}
    for seed in seeds:
        f = legacy_path(archive, "member_probs", dataset,
                        member=member_dir(member), seed=seed)
        if f.exists():
            out[seed] = np.load(f)
    return out


def load_typ_preds(archive: Path, dataset: str, member: str,
                   seeds: List[int]) -> Dict[int, np.ndarray]:
    """Per-seed typology predictions for one member, skipping seeds not on disk."""
    out = {}
    for seed in seeds:
        f = legacy_path(archive, "member_typology", dataset,
                        member=member_dir(member), seed=seed)
        if f.exists():
            out[seed] = np.load(f)
    return out


def load_labels(archive: Path, dataset: str) -> np.ndarray:
    """Ground-truth labels for the temporal test split."""
    f = legacy_path(archive, "test_labels", dataset)
    if not f.exists():
        raise FileNotFoundError(f"No test labels at {f}")
    return np.load(f)


def load_typologies(archive: Path, dataset: str) -> Optional[np.ndarray]:
    """Ground-truth typology indices, -1 for legitimate. None when absent."""
    f = legacy_path(archive, "test_typologies", dataset)
    return np.load(f) if f.exists() else None


def _fmt(vals) -> str:
    """Percentage mean and standard deviation, the format the CSV carries."""
    return f"{np.mean(vals)*100:.1f}±{np.std(vals)*100:.1f}"


def _score_ensemble(labels: np.ndarray, ensemble_preds: np.ndarray) -> dict:
    """Detection metrics for one seed's combined predictions."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    return {
        "f1": f1_score(labels, ensemble_preds, pos_label=1, zero_division=0),
        "prec": precision_score(labels, ensemble_preds, pos_label=1, zero_division=0),
        "rec": recall_score(labels, ensemble_preds, pos_label=1, zero_division=0),
        "acc": accuracy_score(labels, ensemble_preds),
    }


def _score_typology(gt_typ: np.ndarray, ensemble_preds: np.ndarray,
                    member_typ_preds: List[np.ndarray]) -> dict:
    """Typology metrics over the edges the ensemble flagged and truth types.

    The typology prediction is a plurality vote across members. Ties go to
    whichever class ``Counter.most_common`` reports first, which is insertion
    order, which is member order.
    """
    from sklearn.metrics import accuracy_score, f1_score

    ill_mask = ensemble_preds == 1
    gt_mask = gt_typ >= 0
    eval_mask = ill_mask & gt_mask

    if eval_mask.sum() == 0:
        return {"typ_f1": 0.0, "typ_acc": 0.0}

    typ_stack = np.stack(member_typ_preds)[:, eval_mask]
    n_eval = eval_mask.sum()
    typ_votes = np.zeros(n_eval, dtype=int)
    for i in range(n_eval):
        counter = Counter(typ_stack[:, i].tolist())
        typ_votes[i] = counter.most_common(1)[0][0]

    gt = gt_typ[eval_mask]
    return {
        "typ_f1": f1_score(gt, typ_votes, average="macro", zero_division=0),
        "typ_acc": accuracy_score(gt, typ_votes),
    }


def ensemble_score(dataset: str, members: List[str], seeds: List[int],
                   archive: Path) -> List[dict]:
    """Score every combination rule for one dataset. Returns summary rows."""
    print(f"\n{'='*60}")
    print(f"  Ensemble scoring --- {dataset}")
    print(f"  Members: {members}")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    labels = load_labels(archive, dataset)
    n_test = len(labels)
    n_illicit = labels.sum()
    print(f"  Test: {n_test:,} edges ({n_illicit:,} illicit, "
          f"{n_illicit/n_test*100:.2f}%)")

    test_gt_typ = load_typologies(archive, dataset)
    if test_gt_typ is not None:
        print(f"  Typology ground truth: {(test_gt_typ >= 0).sum():,} typed edges")
    else:
        print("  Typology ground truth: not found, typology metrics skipped")

    all_member_probs = {}
    for member in members:
        probs = load_probs(archive, dataset, member, seeds)
        if probs:
            all_member_probs[member] = probs
            print(f"  Loaded {member}: {len(probs)} seeds, "
                  f"{len(next(iter(probs.values()))):,} predictions")
        else:
            print(f"  {member}: no saved probabilities found, skipping")

    # A member whose arrays are a different length is not scoring the same
    # edges and cannot be averaged with the others. Rebuilt rather than deleted
    # in place, which is what the pre-release loop did while iterating.
    all_member_probs = {
        m: probs for m, probs in all_member_probs.items()
        if all(len(p) == n_test for p in probs.values())
    }
    for member in members:
        if member in all_member_probs:
            continue
        print(f"  {member}: probability length does not match {n_test:,} labels, "
              "dropped")

    if len(all_member_probs) < 2:
        print("  Need at least 2 members for an ensemble, skipping.")
        return []

    available = list(all_member_probs.keys())
    n_members = len(available)
    print(f"\n  Using {n_members} members: {available}")

    all_member_typ = {}
    for member in available:
        typ = load_typ_preds(archive, dataset, member, seeds)
        if typ:
            all_member_typ[member] = typ
            print(f"  Loaded {member} typology: {len(typ)} seeds")
    has_typ = test_gt_typ is not None and len(all_member_typ) >= 2

    common_seeds = set(seeds)
    for member in available:
        common_seeds &= set(all_member_probs[member].keys())
    common_seeds = sorted(common_seeds)
    print(f"  Common seeds: {common_seeds}")

    if not common_seeds:
        print("  No common seeds, cannot ensemble.")
        return []

    # ── Individual members, at their own oracle thresholds ──
    print("\n  --- Individual member baselines ---")
    from sklearn.metrics import f1_score

    member_oracle_f1s = {}
    for member in available:
        seed_f1s = []
        seed_oracle_f1s = []
        for seed in common_seeds:
            probs = all_member_probs[member][seed]
            oracle_t, oracle_f1 = find_optimal_threshold(labels, probs)
            seed_oracle_f1s.append(oracle_f1)
            preds = (probs >= oracle_t).astype(int)
            seed_f1s.append(f1_score(labels, preds, pos_label=1, zero_division=0))
        member_oracle_f1s[member] = np.mean(seed_oracle_f1s)
        print(f"  {member:20s} oracle F1={np.mean(seed_f1s)*100:.1f}"
              f"±{np.std(seed_f1s)*100:.1f}%")

    def _build_row(method_label: str, seed_metrics: List[dict]) -> dict:
        typ_f1s = [m.get("typ_f1", 0.0) for m in seed_metrics]
        typ_accs = [m.get("typ_acc", 0.0) for m in seed_metrics]
        return {
            "method": method_label,
            "model": "non-llm",
            "dataset": dataset,
            "n_seeds": len(common_seeds),
            "detection_f1": _fmt([m["f1"] for m in seed_metrics]),
            "detection_precision": _fmt([m["prec"] for m in seed_metrics]),
            "detection_recall": _fmt([m["rec"] for m in seed_metrics]),
            "detection_accuracy": _fmt([m["acc"] for m in seed_metrics]),
            "typology_macro_f1": _fmt(typ_f1s) if has_typ else EMPTY_CELL,
            "typology_accuracy": _fmt(typ_accs) if has_typ else EMPTY_CELL,
            "verifier_pass_rate": EMPTY_CELL,
            "evidence_precision": EMPTY_CELL,
            "evidence_recall": EMPTY_CELL,
            "evidence_f1": EMPTY_CELL,
            "unsupported_claim_rate": EMPTY_CELL,
        }

    def _combine(rule: str, subset: List[str], seed: int) -> np.ndarray:
        """One seed's combined binary predictions under one rule."""
        probs_list = [all_member_probs[m][seed] for m in subset]

        if rule in ("majority", "both"):
            preds_list = []
            for probs in probs_list:
                t, _ = find_optimal_threshold(labels, probs)
                preds_list.append((probs >= t).astype(int))
            votes = np.sum(preds_list, axis=0)
            needed = 2 if rule == "both" else (len(subset) + 1) // 2
            return (votes >= needed).astype(int)

        if rule == "soft_avg":
            avg_probs = np.mean(probs_list, axis=0)
        elif rule == "weighted_soft":
            weights = np.array([member_oracle_f1s[m] for m in subset])
            weights = weights / weights.sum()
            avg_probs = np.zeros(n_test)
            for w, probs in zip(weights, probs_list):
                avg_probs += w * probs
        else:
            raise ValueError(f"unknown rule {rule!r}")

        best_t, _ = find_optimal_threshold(labels, avg_probs)
        return (avg_probs >= best_t).astype(int)

    def _score_rule(label: str, rule: str, subset: List[str]) -> dict:
        seed_metrics = []
        for seed in common_seeds:
            ensemble_preds = _combine(rule, subset, seed)
            metrics = _score_ensemble(labels, ensemble_preds)
            if has_typ:
                typ_members = [m for m in subset
                               if m in all_member_typ and seed in all_member_typ[m]]
                if len(typ_members) >= 2:
                    typ_list = [all_member_typ[m][seed] for m in typ_members]
                    metrics.update(
                        _score_typology(test_gt_typ, ensemble_preds, typ_list))
            seed_metrics.append(metrics)

        f1s = [m["f1"] for m in seed_metrics]
        typ_str = ""
        if has_typ:
            typ_f1s = [m.get("typ_f1", 0.0) for m in seed_metrics]
            typ_str = (f" Typ-F1={np.mean(typ_f1s)*100:.1f}"
                       f"±{np.std(typ_f1s)*100:.1f}%")
        print(f"  {label:30s} F1={np.mean(f1s)*100:.1f}"
              f"±{np.std(f1s)*100:.1f}%{typ_str}")
        return _build_row(label, seed_metrics)

    # Method labels are published values in results/summary_all.csv. Do not
    # reword them; a table would stop matching its source row.
    print("\n  --- Ensemble results ---")
    summary_rows = [
        _score_rule("Ensemble (Hard Majority Vote)", "majority", available),
        _score_rule("Ensemble (Soft Average)", "soft_avg", available),
        _score_rule("Ensemble (Weighted Soft (by oracle))", "weighted_soft",
                    available),
    ]

    if n_members > 2:
        gbt = [m for m in available if m in ("LightGBM+GFP", "XGBoost+GFP")]
        if len(gbt) == 2:
            print(f"\n  --- Boosted-tree ensemble ({' + '.join(gbt)}) ---")
            summary_rows.append(_score_rule(
                "Ensemble-GBT (Hard Vote (2 models → both agree))",
                "both", gbt))
            summary_rows.append(_score_rule(
                "Ensemble-GBT (Soft Average)", "soft_avg", gbt))

    return summary_rows


def write_summary(rows: List[dict], results_dir: Path) -> None:
    """Merge the ensemble rows into summary_all.csv and summary_all.json.

    Rows with the same method and dataset are replaced; every other row,
    including the LLM rows, is left as it is.
    """
    import pandas as pd

    results_dir = paths.ensure(Path(results_dir))
    csv_fn = results_dir / "summary_all.csv"
    json_fn = results_dir / "summary_all.json"

    df_ens = pd.DataFrame(rows)
    if csv_fn.exists():
        df_old = pd.read_csv(csv_fn)
        new_keys = set(zip(df_ens["method"], df_ens["dataset"]))
        df_keep = df_old[~df_old.apply(
            lambda r: (r["method"], r["dataset"]) in new_keys, axis=1)]
        df = pd.concat([df_keep, df_ens], ignore_index=True)
    else:
        df = df_ens

    df = df.sort_values(["method", "dataset"]).reset_index(drop=True)
    df.to_csv(csv_fn, index=False, encoding="utf-8")
    with open(json_fn, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"\n  {'='*60}")
    print("  Results summary, with ensembles")
    print(f"  {'='*60}")
    print(df.to_string(index=False))
    print(f"\n  Saved: {csv_fn}")
    print(f"  Saved: {json_fn}")


def main():
    parser = argparse.ArgumentParser(
        description="Soft and hard vote over the supervised ensemble members")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--members", nargs="+", default=list(ENSEMBLE_MEMBERS),
                        help="Paper member names; archive resolves each to "
                             "its archive directory")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--archive", type=str, default=None,
                        help="Run directory. Default: $AMLC_ARCHIVE")
    parser.add_argument("--results", type=str, default=None,
                        help="Where summary_all.* live. Default: results/")
    args = parser.parse_args()

    archive = Path(args.archive) if args.archive else paths.archive()
    results_dir = Path(args.results) if args.results else paths.results()

    print("=" * 60)
    print("  Ensemble scoring")
    print("=" * 60)

    all_rows = []
    for dataset in args.datasets:
        all_rows.extend(ensemble_score(dataset, args.members, args.seeds, archive))

    if all_rows:
        write_summary(all_rows, results_dir)

    print(f"\n{'='*60}")
    print("  Done")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
