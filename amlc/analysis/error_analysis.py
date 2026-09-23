#!/usr/bin/env python3
"""Error analysis: the supervised ensemble against the LLMs on the HT-Coreset.

Compares per-edge predictions between the ensemble and each LLM: error
transitions, per-typology breakdown, and the failure patterns behind them.

Usage:  make analysis          (scripts/20_error_analysis.py)
Output: results/error_analysis/  CSVs, and figures under results/figures/
"""

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .. import archive, config, paths
from ..baselines.ml.ensemble import load_evaluation_probabilities
from ..case_ids import CASE_ID_WIDTH, LEGACY_CASE_ID_PREFIX, case_id
from ..config import DATASETS, N_TEST_FULL
from ..config import LLM_MODELS as CONFIG_LLM_MODELS
from ..config import ML_THRESHOLDS as THRESHOLDS
from ..coreset.ht_weights import repair_archived_weights
from ..figures._paper_style import (
    COLORS,
    FIG_TEXT,
    FIG_TEXT_TALL,
    LLM_PALETTE,
    TYPOLOGY_PALETTE,
    apply_style,
    short_model,
)

# The typology encoding lives in amlc.typology. The permutation that used to be
# hardcoded in this module agreed with the ground-truth string column on 0.00%
# of typed rows; see that module for the evidence.
from ..typology import (
    TYPOLOGY_CLASSES as _CANONICAL_TYPOLOGY_CLASSES,
)
from ..typology import (
    TYPOLOGY_INT_MAP as _CANONICAL_TYPOLOGY_MAP,
)

apply_style()


# ── Unicode-aware typology extraction ────────────────────────────────────
# GPT-OSS emits the Unicode non-breaking hyphen (U+2011) in compound
# typology names like 'gather‑scatter', which the original ASCII-only
# parser missed → typology=null for ~70% of GPT-OSS illicit predictions.
# We recover by normalising hyphens before matching, and falling back to
# regex on raw_response when the structured field is empty.
_UNICODE_HYPHENS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2043\u00ad"
_CANONICAL_TYPOLOGIES = (
    "scatter-gather", "gather-scatter", "fan-out", "fan-in",
    "bipartite", "cycle", "random", "stack", "legitimate",
)
_RAW_TYPOLOGY_PATTERNS = [
    re.compile(r"observed\s+pattern\s*:?\s*\**\s*([a-z][a-z\- ]{2,30})", re.IGNORECASE),
    re.compile(r"final\s+pattern\s*:?\s*\**\s*([a-z][a-z\- ]{2,30})", re.IGNORECASE),
    re.compile(r"typology\s*:?\s*\**\s*([a-z][a-z\- ]{2,30})", re.IGNORECASE),
]


def _ascii_normalize(s: str) -> str:
    if not s:
        return s
    return s.translate({ord(c): "-" for c in _UNICODE_HYPHENS})


def _normalize_label(s: str) -> str:
    s = _ascii_normalize(str(s)).strip().lower().rstrip(".,;:*")
    for sep in ("(", "[", "{", " and ", " or ", "/"):
        if sep in s:
            s = s.split(sep, 1)[0].strip()
    if s in _CANONICAL_TYPOLOGIES:
        return s
    if "scatter" in s and "gather" in s:
        return "scatter-gather" if s.find("scatter") < s.find("gather") else "gather-scatter"
    if "fan" in s and "out" in s: return "fan-out"
    if "fan" in s and "in" in s:  return "fan-in"
    if "cycl" in s:               return "cycle"
    if "random" in s:             return "random"
    if "bipart" in s:             return "bipartite"
    if "stack" in s:              return "stack"
    if "legit" in s or "normal" in s: return "legitimate"
    return ""


def _extract_from_raw(raw: str) -> str:
    if not raw:
        return ""
    text = _ascii_normalize(raw)
    for pat in _RAW_TYPOLOGY_PATTERNS:
        m = pat.search(text)
        if m:
            n = _normalize_label(m.group(1))
            if n:
                return n
    text_low = text.lower()
    for canon in _CANONICAL_TYPOLOGIES:
        if canon in text_low:
            return canon
    return ""


def _patched_typology(pred: dict) -> str:
    """Recover predicted typology with Unicode-aware extraction.

    Priority order (strict, matching the pre-release
    ``repatch_predictions.py``, which is not part of this release): sourced
    from raw_response, NOT the structured
    `typology` field, because the original baselines parser had a greedy
    'layer'→'stack' fallback that produced canonical-looking but wrong
    values like 'stack' for 'distribution layer' outputs):

      1. Pre-computed `typology_patched` field (written by that pre-release
         script) if present.
      2. Re-extract from `raw_response` with Unicode normalization.
      3. Fall back to the structured `typology` field (legacy path).
    """
    if pred.get("typology_patched"):
        norm = _normalize_label(pred["typology_patched"])
        if norm:
            return norm
    if pred.get("illicit"):
        rec = _extract_from_raw(pred.get("raw_response", ""))
        if rec:
            return rec
    raw = pred.get("typology")
    if raw:
        norm = _normalize_label(raw)
        if norm:
            return norm
    return ""

# ── Paths ──
# Paths come from the environment so a clone is not tied to one machine, and so
# regenerating never overwrites the paper's committed figures by accident.
def _archive_root() -> Path:
    """Parent of the archived ``outputs/``, resolved the way the rest of the
    package resolves it.

    This module used to compute the root itself and fall back to ``.`` when
    neither ``AMLC_ARCHIVE_ROOT`` nor an ``outputs``-suffixed ``AMLC_ARCHIVE``
    was set, which turned a missing archive into a silent read of the current
    directory. :func:`amlc.paths.archive` raises instead, and accepts both the
    outputs directory and its parent. Resolved lazily so importing this module
    never needs an archive.
    """
    return paths.archive().parent


OUT_DIR = Path(os.environ.get("AMLC_ERROR_ANALYSIS_OUT", "results/error_analysis"))
FIG_DIR = Path(os.environ.get("AMLC_FIGURE_OUT", "figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Created here too: the stage writes nine figures and used to assume the
# directory already existed, so `make analysis` crashed on a fresh clone.
FIG_DIR.mkdir(parents=True, exist_ok=True)


# The permutation this file used to hardcode. Retained behind an env switch so
# the published CSVs can be reproduced and the two corrections (coreset draw,
# typology encoding) can be attributed separately.
_LEGACY_TYPOLOGY_MAP = {
    -1: "legit", 0: "cycle", 1: "fan-out", 2: "stack", 3: "fan-in",
    4: "bipartite", 5: "scatter-gather", 6: "random", 7: "gather-scatter",
}
_USE_LEGACY_MAP = os.environ.get("AMLC_LEGACY_TYPOLOGY_MAP") == "1"
TYPOLOGY_MAP = _LEGACY_TYPOLOGY_MAP if _USE_LEGACY_MAP else _CANONICAL_TYPOLOGY_MAP
TYPOLOGY_CLASSES = ([TYPOLOGY_MAP[i] for i in range(8)] if _USE_LEGACY_MAP
                    else _CANONICAL_TYPOLOGY_CLASSES)

# The evaluated models, grouped by family rather than by size. This order is
# not cosmetic: it is the column order of the shipped
# results/error_analysis/typology_recall.csv, so changing it changes that file.
# config.LLM_MODELS is the same seven in the paper's by-size order, and the
# assertion below is what stops the two from drifting: add a model to
# config.yaml and this file raises instead of quietly leaving it out.
LLM_MODELS = [
    "GPT-OSS-120B", "GPT-OSS-20B",
    "Nemotron-3-Super-120B", "Nemotron-3-Nano-30B",
    "Qwen3.5-397B-A17B", "Qwen3.5-35B-A3B", "Qwen3.5-27B",
]
if set(LLM_MODELS) != set(CONFIG_LLM_MODELS):
    raise RuntimeError(
        "LLM_MODELS here must be a reordering of config.LLM_MODELS, not a "
        f"different set. Only here: {sorted(set(LLM_MODELS) - set(CONFIG_LLM_MODELS))}; "
        f"only in config: {sorted(set(CONFIG_LLM_MODELS) - set(LLM_MODELS))}"
    )
LLM_METHOD = archive.prompting_dir("ICL-FS")

# The evaluation reference averages the two GFP-feature boosted trees and all
# five seeds. The original three-member construction scores remain attached to
# the frozen coreset design; they do not supply the predictions compared here.
ENSEMBLE_MEMBERS    = [archive.member_dir(m) for m in config.ENSEMBLE_MEMBERS]
ML_BASELINES_EXTRA  = []
ML_MODELS           = ENSEMBLE_MEMBERS + ML_BASELINES_EXTRA
SEEDS = list(config.SEEDS)

# Fallback for a dataset with no tuned operating point. Both evaluated splits
# have one in THRESHOLDS, so this is only reached by a caller that adds a third.
# It is deliberately not 0.80 or 0.48: at a 0.05-0.12% illicit rate a threshold
# tuned for one split is not a sensible default for another.
THRESHOLD  = 0.5


# ══════════════════════════════════════════════════════════════════════
# 1. Load ground truth, coreset indices and weights
# ══════════════════════════════════════════════════════════════════════
def _archive_case_id(position: int) -> str:
    """The pre-release spelling of an identifier, for joining against archived
    prediction files. Only :mod:`amlc.archive` and this helper know it."""
    return f"{LEGACY_CASE_ID_PREFIX}{position:0{CASE_ID_WIDTH}d}"


def load_ground_truth(dataset):
    base = _archive_root() / "outputs" / "test_probs" / dataset
    labels = np.load(base / "test_labels.npy")
    typologies = np.load(base / "test_typologies.npy")
    return labels, typologies


def load_coreset_subset(dataset):
    base = _archive_root() / "outputs" / "eval_subsets" / dataset
    # The RELEASED draw. The pre-release code loaded the ablation re-draw
    # (V2_full), which shares under half its edges with the draw the LLM
    # predictions are indexed against; see amlc.triage.doubt_triage.
    _stem = archive.draw_stem("ablation-redraw"
                              if os.environ.get("AMLC_LEGACY_DRAW") == "1"
                              else "ht-coreset")
    indices = np.load(base / f"subset_{dataset}_{_stem}.npy")
    if os.environ.get("AMLC_LEGACY_DRAW") == "1":
        # Reproducing the archived numbers, archived weights and all.
        return indices, np.load(base / f"weights_{dataset}_{_stem}.npy")
    # The archived LI-Small vector double-counts the benign population and sums
    # to 2,767,353 against a test split of 1,384,810. Every weighted quantity
    # below reads this column, so loading it raw silently scaled them; the
    # shipped comparison_LI-Small.csv carried exactly that vector. Repair here,
    # from the one function the rest of the package repairs with.
    repair = repair_archived_weights(paths.archive(), dataset, "ht-coreset")
    if not np.array_equal(repair["subset_idx"], indices):
        raise SystemExit(
            f"{dataset}: repaired weights are indexed against a different draw "
            f"than subset_{dataset}_{_stem}.npy"
        )
    weights = repair["fixed"]
    if not np.isclose(weights.sum(), N_TEST_FULL[dataset], rtol=0, atol=1e-6):
        raise SystemExit(
            f"{dataset}: weights sum to {weights.sum():,.1f}, expected "
            f"{N_TEST_FULL[dataset]:,}"
        )
    return indices, weights


# ══════════════════════════════════════════════════════════════════════
# 2. Load ML predictions (ensemble over 5 seeds)
# ══════════════════════════════════════════════════════════════════════
def load_ml_predictions(dataset, v2_indices):
    """Load all evaluation members/seeds on the frozen coreset.

    Typology uses the mode across seeds within each member, then the mode
    across members. Ties resolve to the lowest canonical typology index,
    preserving the existing diagnostic and triage convention.
    """
    from scipy.stats import mode

    results = {}
    typ_results = {}
    for model in ML_MODELS:
        probs_list = []
        typ_list = []
        for seed in SEEDS:
            p = _archive_root() / "outputs" / "test_probs" / dataset / model / f"seed_{seed}.npy"
            pt = _archive_root() / "outputs" / "test_probs" / dataset / model / f"seed_{seed}_typ.npy"
            probs = np.load(p)
            typ = np.load(pt)
            expected_shape = (N_TEST_FULL[dataset],)
            if probs.shape != expected_shape or typ.shape != expected_shape:
                raise ValueError(f"{dataset}/{model}/seed_{seed}: wrong full-test shape")
            if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
                raise ValueError(f"{p}: invalid probability values")
            if not np.isin(typ, np.arange(len(TYPOLOGY_CLASSES))).all():
                raise ValueError(f"{pt}: invalid canonical typology indices")
            probs_list.append(probs[v2_indices])
            typ_list.append(typ[v2_indices])
        results[model] = np.mean(probs_list, axis=0)
        typ_results[model] = mode(np.stack(typ_list), axis=0, keepdims=False).mode

    results["Ensemble (Soft)"] = load_evaluation_probabilities(
        paths.archive(), dataset, seeds=SEEDS)[v2_indices]
    member_typ = [typ_results[m] for m in ENSEMBLE_MEMBERS]
    typ_results["Ensemble (Soft)"] = mode(
        np.stack(member_typ), axis=0, keepdims=False).mode

    return results, typ_results


# ══════════════════════════════════════════════════════════════════════
# 3. Load LLM predictions
# ══════════════════════════════════════════════════════════════════════
def load_llm_predictions(dataset, n_v2):
    """Load the available seed-42 traces used for the diagnostic analysis."""
    results = {}
    for model in LLM_MODELS:
        preds_per_seed = []
        for seed in [42]:
            p = _archive_root() / "outputs" / model / LLM_METHOD / dataset / f"seed_{seed}.json"
            with open(p) as f:
                data = json.load(f)

            # Build case_id → prediction map
            pred_map = {}
            for pred in data.get("predictions", []):
                cid = pred.get("case_id", "")
                if cid in pred_map:
                    raise ValueError(f"{p}: duplicate case_id {cid!r}")
                # Patched typology extraction: GPT-OSS emits Unicode
                # non-breaking hyphen (U+2011) which was missed by the
                # original parser. Recover from raw_response when needed.
                typ = _patched_typology(pred)
                pred_map[cid] = {
                    "illicit": pred.get("illicit", False),
                    "typology": typ,
                    "confidence": pred.get("confidence", 0.5),
                }
            expected = {_archive_case_id(i) for i in range(n_v2)}
            if set(pred_map) != expected:
                raise ValueError(f"{p}: seed-42 predictions do not cover the frozen coreset")
            preds_per_seed.append(pred_map)

        if preds_per_seed:
            # Five-seed task evaluation is separate from this diagnostic slice.
            results[model] = preds_per_seed[0]
    return results


# ══════════════════════════════════════════════════════════════════════
# 4. Build per-edge comparison DataFrame
# ══════════════════════════════════════════════════════════════════════
def build_comparison_df(dataset):
    labels, typologies = load_ground_truth(dataset)
    v2_idx, weights = load_coreset_subset(dataset)

    gt_labels = labels[v2_idx]
    gt_typo = np.array([TYPOLOGY_MAP.get(t, "legit") for t in typologies[v2_idx]])
    # The -1 typology sentinel also occurs on illicit edges without a typed
    # annotation. Their detection label remains illicit, not benign.
    if not _USE_LEGACY_MAP:
        gt_typo = gt_typo.astype(object)
        gt_typo[(gt_labels == 1) & (typologies[v2_idx] < 0)] = "unknown"
    n_v2 = len(v2_idx)

    df = pd.DataFrame({
        # Paper vocabulary on the way out. The archived prediction files key on
        # the pre-release spelling, so the join below derives that from the row
        # position rather than reading it back off this column; regenerating
        # this stage used to emit v2_ identifiers into a release that claims a
        # single vocabulary.
        "case_id": [case_id(i) for i in range(n_v2)],
        "dataset": dataset,
        "gt_label": gt_labels,
        "gt_typology": gt_typo,
        "weight": weights,
    })

    # ML predictions
    ml_preds, ml_typs = load_ml_predictions(dataset, v2_idx)
    for model, probs in ml_preds.items():
        col = model.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
        df[f"{col}_prob"] = probs
        thr = THRESHOLDS.get(dataset, THRESHOLD)
        df[f"{col}_pred"] = (probs >= thr).astype(int)
        if model in ml_typs:
            df[f"{col}_typo"] = [TYPOLOGY_MAP.get(int(t), "legit") for t in ml_typs[model]]

    # LLM predictions
    llm_preds = load_llm_predictions(dataset, n_v2)
    for model, pred_map in llm_preds.items():
        col = model.replace("-", "_").replace(".", "_")
        preds = []
        typos = []
        confs = []
        for i in range(len(df)):
            p = pred_map.get(_archive_case_id(i), {})
            preds.append(1 if p.get("illicit", False) else 0)
            typos.append(p.get("typology", ""))
            confs.append(p.get("confidence", 0.5))
        df[f"{col}_pred"] = preds
        df[f"{col}_typo"] = typos
        df[f"{col}_conf"] = confs

    return df


# ══════════════════════════════════════════════════════════════════════
# 5. Error transition analysis
# ══════════════════════════════════════════════════════════════════════
def error_transition(df, ml_col, llm_col):
    """
    4 categories:
    - Both correct (BC)
    - ML correct, LLM wrong (ML+)
    - LLM correct, ML wrong (LLM+)
    - Both wrong (BW)
    """
    ml_correct = (df[ml_col] == df["gt_label"])
    llm_correct = (df[llm_col] == df["gt_label"])
    return pd.DataFrame({
        "both_correct": ml_correct & llm_correct,
        "ml_only": ml_correct & ~llm_correct,
        "llm_only": ~ml_correct & llm_correct,
        "both_wrong": ~ml_correct & ~llm_correct,
    })


def compute_transitions(df):
    """Compute transitions for all ML-LLM pairs."""
    ml_col = "Ensemble_Soft_pred"
    rows = []
    for col in df.columns:
        if col.endswith("_pred") and col != ml_col and "GFP" not in col and "GCPAL" not in col and not col.startswith("Ensemble"):
            model_name = col.replace("_pred", "").replace("_", "-")
            trans = error_transition(df, ml_col, col)

            # Weighted counts
            for cat in ["both_correct", "ml_only", "llm_only", "both_wrong"]:
                w_count = df.loc[trans[cat], "weight"].sum()
                uw_count = trans[cat].sum()
                rows.append({
                    "dataset": df["dataset"].iloc[0],
                    "llm_model": model_name,
                    "category": cat,
                    "weighted_count": w_count,
                    "unweighted_count": uw_count,
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 6. Per-typology breakdown
# ══════════════════════════════════════════════════════════════════════
def typology_breakdown(df):
    """Per-typology accuracy for ML vs each LLM."""
    ml_col = "Ensemble_Soft_pred"
    illicit = df[df["gt_label"] == 1].copy()
    rows = []

    for typo in sorted(illicit["gt_typology"].unique()):
        sub = illicit[illicit["gt_typology"] == typo]
        n = len(sub)
        # ML recall on this typology
        ml_recall = (sub[ml_col] == 1).mean() if n > 0 else 0

        row = {
            "dataset": df["dataset"].iloc[0],
            "typology": typo,
            "n_edges": n,
            "ML_ensemble_recall": ml_recall,
        }

        # LLM recalls
        for col in df.columns:
            if col.endswith("_pred") and col != ml_col and "GFP" not in col and "GCPAL" not in col and not col.startswith("Ensemble"):
                model_name = col.replace("_pred", "").replace("_", "-")
                llm_recall = (sub[col] == 1).mean() if n > 0 else 0
                row[f"{model_name}_recall"] = llm_recall

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 7. Error type analysis (FP/FN breakdown)
# ══════════════════════════════════════════════════════════════════════
def error_type_summary(df):
    """FP and FN rates for each model."""
    rows = []
    pred_cols = [c for c in df.columns if c.endswith("_pred")]

    for col in pred_cols:
        model_name = col.replace("_pred", "").replace("_", "-")
        tp = ((df[col] == 1) & (df["gt_label"] == 1)).sum()
        fp = ((df[col] == 1) & (df["gt_label"] == 0)).sum()
        fn = ((df[col] == 0) & (df["gt_label"] == 1)).sum()
        tn = ((df[col] == 0) & (df["gt_label"] == 0)).sum()

        # Weighted versions
        w_tp = df.loc[(df[col] == 1) & (df["gt_label"] == 1), "weight"].sum()
        w_fp = df.loc[(df[col] == 1) & (df["gt_label"] == 0), "weight"].sum()
        w_fn = df.loc[(df[col] == 0) & (df["gt_label"] == 1), "weight"].sum()

        w_prec = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0
        w_rec = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0
        w_f1 = 2 * w_prec * w_rec / (w_prec + w_rec) if (w_prec + w_rec) > 0 else 0

        is_ml = "GFP" in col or "Ensemble" in col or "GCPAL" in col
        rows.append({
            "dataset": df["dataset"].iloc[0],
            "model": model_name,
            "type": "ML" if is_ml else "LLM",
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "w_precision": round(w_prec, 4),
            "w_recall": round(w_rec, 4),
            "w_f1": round(w_f1, 4),
            "uw_precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0,
            "uw_recall": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
            "FP_rate_subset": round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 8. Typology F1 analysis
# ══════════════════════════════════════════════════════════════════════
def typology_f1_analysis(df):
    """Compute typology classification F1 (per-class + macro) for all models.

    Only evaluated on TRUE POSITIVE detections (predicted illicit AND ground truth illicit).
    This matches the paper's definition: typology classification conditioned on correct detection.
    """
    illicit_mask = df["gt_label"] == 1

    rows = []
    typo_cols = [c for c in df.columns if c.endswith("_typo")]

    for col in typo_cols:
        model_name = col.replace("_typo", "").replace("_", "-")

        # Determine which pred col corresponds to this typo col
        pred_col = col.replace("_typo", "_pred")
        if pred_col not in df.columns:
            # ML models: check with different naming
            pred_col_alt = col.replace("_typo", "_pred")
            if pred_col_alt not in df.columns:
                continue
            pred_col = pred_col_alt

        # True positives only: predicted illicit AND actually illicit
        tp_mask = illicit_mask & (df[pred_col] == 1)
        if tp_mask.sum() == 0:
            continue

        gt_tp = df.loc[tp_mask, "gt_typology"].values
        pred_tp = df.loc[tp_mask, col].values

        # Normalize LLM typology strings (lowercase, strip)
        pred_tp_norm = []
        for p in pred_tp:
            p_str = str(p).lower().strip()
            # Map common aliases
            if p_str in ("fan_out", "fanout"):
                p_str = "fan-out"
            elif p_str in ("fan_in", "fanin"):
                p_str = "fan-in"
            elif p_str in ("scatter_gather", "scattergather"):
                p_str = "scatter-gather"
            elif p_str in ("gather_scatter", "gatherscatter"):
                p_str = "gather-scatter"
            pred_tp_norm.append(p_str)
        pred_tp_norm = np.array(pred_tp_norm)

        # Per-class F1
        per_class = {}
        for cls in TYPOLOGY_CLASSES:
            gt_bin = (gt_tp == cls).astype(int)
            pred_bin = (pred_tp_norm == cls).astype(int)
            if gt_bin.sum() > 0:
                per_class[cls] = f1_score(gt_bin, pred_bin, zero_division=0)
            else:
                per_class[cls] = np.nan

        # Macro F1
        valid_f1s = [v for v in per_class.values() if not np.isnan(v)]
        macro_f1 = np.mean(valid_f1s) if valid_f1s else 0

        # Accuracy
        acc = (gt_tp == pred_tp_norm).mean()

        is_ml = "GFP" in col or "Ensemble" in col or "GCPAL" in col
        row = {
            "dataset": df["dataset"].iloc[0],
            "model": model_name,
            "type": "ML" if is_ml else "LLM",
            "n_tp": int(tp_mask.sum()),
            "typology_macro_f1": round(macro_f1, 4),
            "typology_accuracy": round(acc, 4),
        }
        for cls in TYPOLOGY_CLASSES:
            row[f"f1_{cls}"] = round(per_class.get(cls, 0), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_typology_f1(typo_f1_df, dataset):
    """Grouped bar chart of per-class typology F1."""
    sub = typo_f1_df[typo_f1_df["dataset"] == dataset].copy()
    if sub.empty:
        return

    # Display order: ML first (lightest to heaviest by F1), then the LLMs by
    # active parameter count. Model names here are the spelling used inside the
    # results CSVs, where the separator is a hyphen ("Qwen3-5") rather than the
    # paper's dot; _paper_style.pretty_model maps them back for display.
    ml_order  = ["LightGBM-GFP", "XGBoost-GFP", "Ensemble-Soft"]
    llm_order = ["GPT-OSS-20B", "Nemotron-3-Nano-30B", "GPT-OSS-120B",
                 "Nemotron-3-Super-120B", "Qwen3-5-397B-A17B"]
    desired   = ml_order + llm_order
    rank      = {m: i for i, m in enumerate(desired)}
    sub       = sub.assign(_rank=sub["model"].map(rank).fillna(99))
    sub       = sub.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

    # Textwidth (figure*) — 9 methods × 8 typologies needs the wider canvas
    fig, ax = plt.subplots(figsize=(FIG_TEXT_TALL[0], FIG_TEXT_TALL[1] + 0.4))

    f1_cols = [c for c in sub.columns if c.startswith("f1_")]
    class_names = [c.replace("f1_", "") for c in f1_cols]
    x = np.arange(len(class_names))
    n_models = len(sub)
    w = 0.8 / max(n_models, 1)

    # ML in greys/blues, LLMs in warm palette → easy ML/LLM separation
    ml_colors  = ["#1f4e79", "#2e75b6", "#5b9bd5", "#9dc3e6"]      # blue family
    llm_colors = ["#c55a11", "#ed7d31", "#f4b183", "#996633", "#7030a0"]  # warm/purple

    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [row[c] for c in f1_cols]
        is_ml = row["type"] == "ML"
        col_idx = i if is_ml else (i - len(ml_order))
        color = (ml_colors[col_idx % len(ml_colors)] if is_ml
                 else llm_colors[col_idx % len(llm_colors)])
        label = f"{short_model(row['model'])} ({row['typology_macro_f1']:.1%})"
        ax.bar(x + i * w, [v * 100 for v in vals], w, label=label,
               color=color, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylabel("Typology F1 (%)")
    ax.set_title(f"Per-Class Typology F1: ML vs LLMs — {dataset}")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False, columnspacing=1.0, handletextpad=0.5)
    ax.set_ylim(0, 110)
    plt.subplots_adjust(bottom=0.32)

    out = FIG_DIR / f"typology_f1_{dataset.lower().replace('-','_')}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_typology_f1_llm_only(typo_f1_df, dataset):
    """Separate chart for LLM-only typology F1 (ML dominates scale)."""
    sub = typo_f1_df[(typo_f1_df["dataset"] == dataset) & (typo_f1_df["type"] == "LLM")].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    f1_cols = [c for c in sub.columns if c.startswith("f1_")]
    class_names = [c.replace("f1_", "") for c in f1_cols]
    x = np.arange(len(class_names))
    n_models = len(sub)
    w = 0.8 / max(n_models, 1)

    colors_list = LLM_PALETTE

    for i, (_, row) in enumerate(sub.iterrows()):
        vals = [row[c] * 100 for c in f1_cols]
        label = f"{short_model(row['model'])} ({row['typology_macro_f1']:.1%})"
        ax.bar(x + i * w, vals, w, label=label,
               color=colors_list[i % len(colors_list)], edgecolor="white", linewidth=0.3)

    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(class_names, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("Typology F1 (%)")
    ax.set_title(f"Per-Class Typology F1 (LLMs only) — {dataset}", fontsize=14)
    ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False, columnspacing=1.0, handletextpad=0.5)
    ax.set_ylim(0, max(20, sub[f1_cols].max().max() * 110))
    plt.subplots_adjust(bottom=0.30)
    plt.tight_layout()

    out = FIG_DIR / f"typology_f1_llm_{dataset.lower().replace('-','_')}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
# 9b. Prediction distribution analysis
# ══════════════════════════════════════════════════════════════════════
def prediction_count_summary(df):
    """Count predictions per model: illicit/legit predicted."""
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    rows = []
    for col in pred_cols:
        model_name = col.replace("_pred", "").replace("_", "-")
        is_ml = "GFP" in col or "Ensemble" in col or "GCPAL" in col
        n_pred_ill = (df[col] == 1).sum()
        n_pred_leg = (df[col] == 0).sum()
        n_gt_ill = (df["gt_label"] == 1).sum()
        rows.append({
            "dataset": df["dataset"].iloc[0],
            "model": model_name,
            "type": "ML" if is_ml else "LLM",
            "pred_illicit": n_pred_ill,
            "pred_legit": n_pred_leg,
            "gt_illicit": n_gt_ill,
            "gt_legit": len(df) - n_gt_ill,
            "over_prediction_ratio": round(n_pred_ill / n_gt_ill, 2) if n_gt_ill > 0 else 0,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 9c. Error transition deep analysis: WHY ML/LLM fail
# ══════════════════════════════════════════════════════════════════════
def error_transition_deep_analysis(df, dataset):
    """Analyze characteristics of ML-only-correct vs LLM-only-correct edges.

    For each LLM model:
    - ML-correct-only: what typology, difficulty score distribution
    - LLM-correct-only: what typology, difficulty score distribution
    """
    ml_pred_col = "Ensemble_Soft_pred"
    ml_prob_col = "Ensemble_Soft_prob"

    all_rows = []
    all_distribution = []

    for col in df.columns:
        if not (col.endswith("_pred") and col != ml_pred_col and "GFP" not in col and "GCPAL" not in col and not col.startswith("Ensemble")):
            continue
        model_name = col.replace("_pred", "").replace("_", "-")

        ml_correct = (df[ml_pred_col] == df["gt_label"])
        llm_correct = (df[col] == df["gt_label"])

        groups = {
            "both_correct": ml_correct & llm_correct,
            "ml_only_correct": ml_correct & ~llm_correct,
            "llm_only_correct": ~ml_correct & llm_correct,
            "both_wrong": ~ml_correct & ~llm_correct,
        }

        for gname, gmask in groups.items():
            sub = df[gmask]
            if len(sub) == 0:
                continue

            # Typology distribution
            typo_dist = sub["gt_typology"].value_counts().to_dict()

            # ML probability distribution
            probs = sub[ml_prob_col].values if ml_prob_col in sub.columns else []

            # GT label distribution
            n_ill = (sub["gt_label"] == 1).sum()
            n_leg = (sub["gt_label"] == 0).sum()

            row = {
                "dataset": dataset,
                "llm_model": model_name,
                "group": gname,
                "n_edges": len(sub),
                "n_illicit": int(n_ill),
                "n_legit": int(n_leg),
                "ml_prob_mean": round(float(np.mean(probs)), 4) if len(probs) > 0 else None,
                "ml_prob_median": round(float(np.median(probs)), 4) if len(probs) > 0 else None,
                "ml_prob_std": round(float(np.std(probs)), 4) if len(probs) > 0 else None,
            }
            # Add typology counts
            # Do not overwrite n_legit: it counts ground-truth benign edges,
            # while missing typology annotations may occur on illicit edges.
            for typo in TYPOLOGY_CLASSES + ["unknown"]:
                row[f"n_{typo}"] = typo_dist.get(typo, 0)
            all_rows.append(row)

            # Per-edge distribution for histogram
            for _, edge in sub.iterrows():
                all_distribution.append({
                    "dataset": dataset,
                    "llm_model": model_name,
                    "group": gname,
                    "gt_label": edge["gt_label"],
                    "gt_typology": edge["gt_typology"],
                    "ml_prob": edge.get(ml_prob_col, None),
                })

    return pd.DataFrame(all_rows), pd.DataFrame(all_distribution)


def plot_error_transition_distributions(dist_df, dataset):
    """Plot ML probability distributions for each error transition group."""
    llm_models = dist_df["llm_model"].unique()

    for model in llm_models:
        msub = dist_df[dist_df["llm_model"] == model]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

        groups = ["both_correct", "ml_only_correct", "llm_only_correct", "both_wrong"]
        # This panel is drawn in a single colour. A per-group palette was built
        # here and never passed to the plot; it is not reinstated, because doing
        # so would change a figure the paper already prints.
        titles = {"both_correct": "Both Correct", "ml_only_correct": "ML Only Correct",
                  "llm_only_correct": "LLM Only Correct", "both_wrong": "Both Wrong"}

        for ax, gname in zip(axes, groups):
            gsub = msub[msub["group"] == gname]
            if gsub.empty:
                ax.set_title(titles[gname] + "\n(n=0)", fontsize=12)
                continue

            # Split by gt_label
            ill = gsub[gsub["gt_label"] == 1]["ml_prob"].dropna()
            leg = gsub[gsub["gt_label"] == 0]["ml_prob"].dropna()

            bins = np.linspace(0, 1, 25)
            if len(leg) > 0:
                ax.hist(leg, bins=bins, alpha=0.6, color=COLORS["both_correct"], label=f"legit (n={len(leg)})", density=True)
            if len(ill) > 0:
                ax.hist(ill, bins=bins, alpha=0.7, color=COLORS["fail"], label=f"illicit (n={len(ill)})", density=True)

            ax.set_title(f"{titles[gname]}\n(n={len(gsub)})", fontsize=12)
            ax.set_xlabel("ML prob s(eᵢ)", fontsize=11)
            ax.legend(fontsize=10)
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

        axes[0].set_ylabel("Density")
        model_short = model.replace("Nemotron-3-", "Nem-")
        fig.suptitle(f"ML Ensemble Score Distribution by Error Group — {model_short} — {dataset}",
                     fontsize=12, y=1.02)
        plt.tight_layout()

        out = FIG_DIR / f"error_dist_{model.lower().replace('-','_').replace('.','_')}_{dataset.lower().replace('-','_')}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}")


def plot_error_transition_typology(summary_df, dataset):
    """Stacked bar: typology breakdown per error group, for one representative LLM."""
    # Pick the best LLM (Qwen3.5 or first available)
    sub = summary_df[summary_df["dataset"] == dataset]
    models = sub["llm_model"].unique()
    target = [m for m in models if "Qwen" in m]
    model = target[0] if target else models[0]

    msub = sub[sub["llm_model"] == model].copy()
    if msub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    groups = ["both_correct", "ml_only_correct", "llm_only_correct", "both_wrong"]
    group_labels = ["Both\ncorrect", "ML only\ncorrect", "LLM only\ncorrect", "Both\nwrong"]
    typo_cols = [f"n_{t}" for t in TYPOLOGY_CLASSES + ["legit"]]
    typo_labels = TYPOLOGY_CLASSES + ["legit"]
    colors_t = TYPOLOGY_PALETTE + [COLORS["ref_line"]]

    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))

    for i, (tc, tl) in enumerate(zip(typo_cols, typo_labels)):
        vals = []
        for g in groups:
            row = msub[msub["group"] == g]
            vals.append(row[tc].values[0] if len(row) > 0 and tc in row.columns else 0)
        ax.bar(x, vals, bottom=bottom, label=tl, color=colors_t[i % len(colors_t)],
               edgecolor="white", linewidth=0.5)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=13)
    ax.set_ylabel("Number of edges")
    model_short = model.replace("Qwen3-5-397B-A17B", "Qwen3.5-397B")
    ax.set_title(f"Typology Breakdown per Error Group — Ensemble vs {model_short} — {dataset}",
                 fontsize=13)
    ax.legend(fontsize=11, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout()

    out = FIG_DIR / f"error_typology_breakdown_{dataset.lower().replace('-','_')}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
# 10. Plotting (detection)
# ══════════════════════════════════════════════════════════════════════
def plot_error_transitions(trans_df, dataset):
    """Stacked bar chart of error transitions."""
    # Textwidth (figure*) — 7 LLMs in horizontal bar chart
    fig, ax = plt.subplots(figsize=FIG_TEXT)

    sub = trans_df[trans_df["dataset"] == dataset].copy()
    models = sub["llm_model"].unique()

    # Pivot to wide
    pivot = sub.pivot(index="llm_model", columns="category", values="unweighted_count").fillna(0)
    pivot = pivot.reindex(models)

    colors = {"both_correct": COLORS["both_correct"], "ml_only": COLORS["ml_only"],
              "llm_only": COLORS["llm_only"], "both_wrong": COLORS["both_wrong"]}
    labels = {"both_correct": "Both correct", "ml_only": "ML only correct",
              "llm_only": "LLM only correct", "both_wrong": "Both wrong"}

    bottom = np.zeros(len(models))
    for cat in ["both_correct", "ml_only", "llm_only", "both_wrong"]:
        if cat in pivot.columns:
            vals = pivot[cat].values
            ax.barh(range(len(models)), vals, left=bottom, color=colors[cat],
                    label=labels[cat], edgecolor="white", linewidth=0.5)
            bottom += vals

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_model(m) for m in models])
    ax.set_xlabel("Number of edges (HT-Coreset)")
    ax.set_title(f"Error Transitions: ML Ensemble vs LLMs — {dataset}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
              frameon=False)
    ax.invert_yaxis()
    plt.subplots_adjust(bottom=0.22)

    out = FIG_DIR / f"error_transitions_{dataset.lower().replace('-','_')}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")
    return out


def plot_typology_recall(typo_df, dataset):
    """Grouped bar chart of per-typology recall."""
    sub = typo_df[typo_df["dataset"] == dataset].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))

    recall_cols = [c for c in sub.columns if c.endswith("_recall")]
    x = np.arange(len(sub))
    n_models = len(recall_cols)
    w = 0.8 / n_models

    colors_list = LLM_PALETTE

    for i, col in enumerate(recall_cols):
        model_name = short_model(col.replace("_recall", ""))
        color = colors_list[i % len(colors_list)]
        ax.bar(x + i * w, sub[col].values * 100, w, label=model_name,
               color=color, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(sub["typology"].values, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("Recall (%)")
    ax.set_title(f"Per-Typology Recall: ML vs LLMs — {dataset}", fontsize=14)
    ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False, columnspacing=1.0, handletextpad=0.5)
    ax.set_ylim(0, 110)
    plt.subplots_adjust(bottom=0.30)

    out = FIG_DIR / f"typology_recall_{dataset.lower().replace('-','_')}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_fp_fn_comparison(err_df, dataset):
    """Side-by-side FP vs FN counts."""
    sub = err_df[err_df["dataset"] == dataset].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    models = [short_model(m) for m in sub["model"].values]
    y = np.arange(len(models))

    # Color by type
    colors = [COLORS["ml"] if t == "ML" else COLORS["dt"] for t in sub["type"]]

    ax1.barh(y, sub["FP"].values, color=colors, edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(models, fontsize=12)
    ax1.set_xlabel("False Positives (HT-Coreset)")
    ax1.set_title("False Positives")
    ax1.invert_yaxis()

    ax2.barh(y, sub["FN"].values, color=colors, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(models, fontsize=12)
    ax2.set_xlabel("False Negatives (HT-Coreset)")
    ax2.set_title("False Negatives")
    ax2.invert_yaxis()

    fig.suptitle(f"FP vs FN: ML and LLM — {dataset}", fontsize=14)
    plt.tight_layout()

    out = FIG_DIR / f"fp_fn_comparison_{dataset.lower().replace('-','_')}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    all_trans = []
    all_typo = []
    all_err = []
    all_typo_f1 = []
    all_pred_counts = []
    all_deep_summary = []
    all_deep_dist = []

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"  {dataset}")
        print(f"{'='*60}")

        # Build comparison DF
        print("Loading predictions...")
        df = build_comparison_df(dataset)
        df.to_csv(OUT_DIR / f"comparison_{dataset}.csv", index=False)
        print(f"  Saved comparison_{dataset}.csv ({len(df)} rows, {len(df.columns)} cols)")

        # Prediction counts
        print("\nPrediction counts:")
        pred_counts = prediction_count_summary(df)
        all_pred_counts.append(pred_counts)
        print(pred_counts[["model", "type", "pred_illicit", "pred_legit",
                           "gt_illicit", "gt_legit", "over_prediction_ratio"]].to_string())

        # Error types
        print("\nComputing error types...")
        err = error_type_summary(df)
        all_err.append(err)
        print(err[["model", "type", "TP", "FP", "FN", "TN", "w_precision", "w_recall", "w_f1"]].to_string())

        # Error transitions
        print("\nComputing error transitions (ML Ensemble vs LLMs)...")
        trans = compute_transitions(df)
        all_trans.append(trans)

        # Deep error transition analysis
        print("\nDeep error transition analysis (WHY ML/LLM fail)...")
        deep_summary, deep_dist = error_transition_deep_analysis(df, dataset)
        all_deep_summary.append(deep_summary)
        all_deep_dist.append(deep_dist)
        # Print summary for one model
        for model in deep_summary["llm_model"].unique()[:1]:
            ms = deep_summary[deep_summary["llm_model"] == model]
            print(f"\n  {model}:")
            print(ms[["group", "n_edges", "n_illicit", "n_legit",
                       "ml_prob_mean", "ml_prob_median", "ml_prob_std"]].to_string())

        # Typology breakdown (detection recall per typology)
        print("\nComputing per-typology recall...")
        typo = typology_breakdown(df)
        all_typo.append(typo)

        # Typology F1 analysis
        print("\nComputing typology classification F1...")
        typo_f1 = typology_f1_analysis(df)
        all_typo_f1.append(typo_f1)
        if not typo_f1.empty:
            print(typo_f1[["model", "type", "n_tp", "typology_macro_f1", "typology_accuracy"]].to_string())

        # Plots
        print("\nGenerating figures...")
        plot_error_transitions(trans, dataset)
        plot_typology_recall(typo, dataset)
        plot_fp_fn_comparison(err, dataset)
        plot_typology_f1(typo_f1, dataset)
        plot_typology_f1_llm_only(typo_f1, dataset)
        plot_error_transition_distributions(deep_dist, dataset)
        plot_error_transition_typology(deep_summary, dataset)

    # Save aggregated results
    trans_all = pd.concat(all_trans, ignore_index=True)
    trans_all.to_csv(OUT_DIR / "error_transitions.csv", index=False)

    typo_all = pd.concat(all_typo, ignore_index=True)
    typo_all.to_csv(OUT_DIR / "typology_recall.csv", index=False)

    err_all = pd.concat(all_err, ignore_index=True)
    err_all.to_csv(OUT_DIR / "error_summary.csv", index=False)

    typo_f1_all = pd.concat(all_typo_f1, ignore_index=True)
    typo_f1_all.to_csv(OUT_DIR / "typology_f1.csv", index=False)

    pred_all = pd.concat(all_pred_counts, ignore_index=True)
    pred_all.to_csv(OUT_DIR / "prediction_counts.csv", index=False)

    deep_all = pd.concat(all_deep_summary, ignore_index=True)
    deep_all.to_csv(OUT_DIR / "error_transition_deep.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  All results saved to: {OUT_DIR}")
    print(f"  Figures saved to: {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
