"""Doubt Triage (DT) — selective ML+LLM deferral on the HT-Coreset.

Horvitz-Thompson importance weights partition the coreset into a *census
stratum* (weight = 1: every illicit edge plus the retained hard-negative
benign edges) and a *sampled stratum* (weight >> 1: easy benign). The
rule consults the LLM only inside the census stratum, where each additional
false positive carries weight one. Evaluation uses the two temporal-trained
boosters; the original construction scorer still defines the frozen strata and
weights. This label-informed rule is a diagnostic use case, not a deployment
policy.

Reference:
  Horvitz & Thompson (1952). JASA 47(260): 663-685.
  Mozannar & Sontag (2020). "Consistent Estimators for Learning to Defer", ICML.

Usage:
    from amlc.triage.doubt_triage import DoubtTriage, load_coreset
    data = load_coreset("HI-Small")                     # downloaded and cached
    dt = DoubtTriage(data["ml_probs"], data["weights"], data["ml_threshold"])
    preds, routing = dt.predict(llm_preds)
    result = dt.evaluate(data["labels"], llm_preds, ...)

Two loaders. :func:`load_coreset` pulls the released HT-Coreset from the
published dataset, which is what a fresh clone uses.
:func:`load_coreset_from_archive` rebuilds it from the original run directory
and is only needed to regenerate results. Every pre-release identifier it has
to touch lives in :mod:`amlc.archive`, not here.

Provenance note (2026-08-10)
----------------------------
This module supersedes ``src/hybrid.py`` in the pre-release tree, which
hardcoded the ablation re-draw while the serialised prompts, ``case_id``
values, labels and weights that the LLMs were evaluated on all come from the
released HT-Coreset draw. The two draws are the same size but share only 47.3%
(HI-Small) / 49.4% (LI-Small) of their edges and match positionally on just
3.0% / 8.9%, so 25.6% / 32.4% of array positions paired an LLM verdict about
one edge with the ground-truth label and ensemble probability of a different
edge.

Fixes here:
  1. The draw is an explicit parameter defaulting to ``"ht-coreset"``. Pass
     ``draw="ablation-redraw"`` only to reproduce the ARR-submission numbers.
  2. :func:`verify_alignment` cross-checks labels and weights against the
     serialised subset the prompts were built from and raises rather than
     silently scoring a misaligned pairing.
  3. Archive loads repair the HT weights by default, because the archived
     LI-Small vector double-counts almost the whole benign population
     (see :mod:`amlc.coreset.ht_weights`).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..case_ids import is_case_id, position as case_position
from ..typology import TYPOLOGY_CLASSES, TYPOLOGY_INT_MAP
from ..archive import canonical_strategy, legacy_path, member_dir
from ..config import (CORESET_DRAWS, CONSTRUCTION_THRESHOLDS, ENSEMBLE_MEMBERS,
                      ML_THRESHOLDS)


# ── Unicode-aware typology extraction ────────────────────────────────────
# GPT-OSS emits a Unicode non-breaking hyphen (U+2011) inside compound
# typology names (e.g. 'gather-scatter'); a parser keyed on ASCII '-' misses
# them, which nulls typology for ~70% of GPT-OSS illicit predictions and
# under-counts typology macro-F1. Recover by normalising hyphens and falling
# back to a raw_response regex.
_UNICODE_HYPHENS = "‐‑‒–—―−⁃­"
_CANONICAL_TYPOLOGIES = (
    "scatter-gather", "gather-scatter", "fan-out", "fan-in",
    "bipartite", "cycle", "random", "stack", "legitimate",
)
_RAW_TYPOLOGY_PATTERNS = [
    re.compile(r"observed\s+pattern\s*:?\s*\**\s*([a-z][a-z\- ]{2,30})", re.IGNORECASE),
    re.compile(r"final\s+pattern\s*:?\s*\**\s*([a-z][a-z\- ]{2,30})", re.IGNORECASE),
    re.compile(r"typology\s*:?\s*\**\s*([a-z][a-z\- ]{2,30})", re.IGNORECASE),
]


def _ascii_normalize_hyphens(s: str) -> str:
    if not s:
        return s
    return s.translate({ord(c): "-" for c in _UNICODE_HYPHENS})


def _normalize_typology_label(s: str) -> str:
    s = _ascii_normalize_hyphens(str(s)).strip().lower().rstrip(".,;:*")
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


def _extract_typology_from_raw(raw: str) -> str:
    if not raw:
        return ""
    text = _ascii_normalize_hyphens(raw)
    for pat in _RAW_TYPOLOGY_PATTERNS:
        m = pat.search(text)
        if m:
            n = _normalize_typology_label(m.group(1))
            if n:
                return n
    text_low = text.lower()
    for canon in _CANONICAL_TYPOLOGIES:
        if canon in text_low:
            return canon
    return ""


# ENSEMBLE_MEMBERS and SEEDS are imported above rather than restated here.
# archive owns the mapping from a member's paper name to the directory name used
# inside the archived tree.


# ═══════════════════════════════════════════════════════════════════════
# Core Doubt Triage logic
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DoubtTriageResult:
    """Output of a single Doubt Triage evaluation."""
    preds: np.ndarray               # (n,) int   hybrid binary predictions
    routing: np.ndarray             # (n,) str   ml_illicit / ml_legit / llm_*
    n_consult: int                  # edges deferred to the LLM
    consult_ratio: float            # n_consult / n_total

    # HT-weighted detection metrics
    w_precision: float = 0.0
    w_recall: float = 0.0
    w_f1: float = 0.0

    # HT-weighted typology metrics (if typology provided)
    w_typ_macro_f1: float = 0.0
    w_typ_accuracy: float = 0.0

    typ_per_class: dict = field(default_factory=dict)


# "dt" is the paper's Doubt Triage rule. The pre-release alias is accepted on
# input via archive.canonical_strategy so older configs keep working.
STRATEGIES = ("dt", "confidence", "disagree", "union")


class DoubtTriage:
    """Selective deferral of ML predictions to an LLM inside the census stratum.

    Four deferral rules, all restricted to the census stratum (weight = 1):

    - ``dt`` (default, the paper's rule) defer to the LLM when the ensemble
      predicts legit. Targets the ensemble's known false-negative bias on
      hard-negative benign edges.
    - ``confidence`` defer when the ensemble is uncertain near its threshold
      (|prob - threshold| < ``confidence_delta``).
    - ``disagree`` defer only when ML and LLM disagree, trust ML otherwise.
    - ``union`` defer when either ``dt`` or ``confidence`` fires.

    Parameters
    ----------
    ml_probs : ndarray (n,)
        Ensemble probability per edge, averaged over members and seeds.
    weights : ndarray (n,)
        HT importance weights: 1.0 in the census stratum, > 1 in the sampled
        stratum.
    ml_threshold : float
        Evaluation ensemble decision threshold from ``config.ML_THRESHOLDS``.
    weight_threshold : float
        Edges with weight <= this are census stratum. Default 1.01 for float
        tolerance.
    strategy : str
        One of ``STRATEGIES``.
    confidence_delta : float
        Band around ``ml_threshold`` used by ``confidence`` and ``union``.
    """

    def __init__(
        self,
        ml_probs: np.ndarray,
        weights: np.ndarray,
        ml_threshold: float = ML_THRESHOLDS["HI-Small"],
        weight_threshold: float = 1.01,
        strategy: str = "dt",
        confidence_delta: float = 0.10,
    ):
        strategy = canonical_strategy(strategy)
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

        self.ml_probs = ml_probs
        self.weights = weights
        self.ml_threshold = ml_threshold
        self.strategy = strategy
        self.confidence_delta = confidence_delta
        self.n = len(ml_probs)

        # Partition
        self.ml_preds = (ml_probs >= ml_threshold).astype(int)
        self.census_mask = weights <= weight_threshold      # weight = 1 stratum
        self._dt_mask = self.census_mask & (self.ml_preds == 0)
        self._conf_mask = self.census_mask & (
            np.abs(ml_probs - ml_threshold) < confidence_delta
        )
        # The consult mask for dt/confidence/union does not depend on the LLM;
        # disagree does, and is resolved in predict().
        if strategy == "dt":
            self.consult_mask = self._dt_mask
        elif strategy == "confidence":
            self.consult_mask = self._conf_mask
        elif strategy == "union":
            self.consult_mask = self._dt_mask | self._conf_mask
        else:  # disagree
            self.consult_mask = self.census_mask.copy()

    def _consult_mask_for(self, llm_preds: np.ndarray) -> np.ndarray:
        if self.strategy != "disagree":
            return self.consult_mask
        return self.census_mask & (self.ml_preds != llm_preds.astype(int))

    def predict(self, llm_preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Route each edge, then return (preds, routing) of length n.

        Outside the consult mask keep the ML prediction; inside take the LLM
        prediction. Edges with weight > 1 always trust ML.
        """
        if len(llm_preds) != self.n:
            raise ValueError(
                f"llm_preds has length {len(llm_preds)} but the coreset has {self.n} "
                "edges. This is the misalignment that produced the pre-2026-08-10 "
                "Doubt Triage numbers; check that the LLM predictions and the "
                "coreset draw come from the same subset."
            )
        consult_mask = self._consult_mask_for(llm_preds)
        preds = self.ml_preds.copy()
        routing = np.where(self.ml_preds == 1, "ml_illicit", "ml_legit").astype(object)

        preds[consult_mask] = llm_preds[consult_mask]
        routing[consult_mask] = np.where(
            llm_preds[consult_mask] == 1, "llm_flipped", "llm_legit"
        )
        # Refresh so downstream code sees the mask actually used (matters for
        # the disagree strategy).
        self.consult_mask = consult_mask
        return preds, routing

    def evaluate(
        self,
        labels: np.ndarray,
        llm_preds: np.ndarray,
        llm_typologies: Optional[np.ndarray] = None,
        ml_typologies: Optional[np.ndarray] = None,
        gt_typologies: Optional[np.ndarray] = None,
    ) -> DoubtTriageResult:
        """Full evaluation with HT-weighted metrics."""
        preds, routing = self.predict(llm_preds)
        n_consult = int(self.consult_mask.sum())

        wp, wr, wf1 = ht_weighted_prf(preds, labels, self.weights)

        wtf1, wtacc, tpc = 0.0, 0.0, {}
        if gt_typologies is not None:
            tp_mask = (preds == 1) & (labels == 1)
            hybrid_typ = np.full(self.n, "unknown", dtype=object)

            # ML-contributed true positives use the ML typology head
            if ml_typologies is not None:
                ml_tp = tp_mask & (self.ml_preds == 1)
                hybrid_typ[ml_tp] = ml_typologies[ml_tp]

            # LLM-contributed true positives use the LLM typology
            if llm_typologies is not None:
                llm_tp = tp_mask & self.consult_mask
                hybrid_typ[llm_tp] = llm_typologies[llm_tp]

            wtf1, wtacc, tpc = ht_weighted_typology(
                hybrid_typ, gt_typologies, tp_mask, self.weights
            )

        return DoubtTriageResult(
            preds=preds,
            routing=routing,
            n_consult=n_consult,
            consult_ratio=n_consult / self.n,
            w_precision=wp,
            w_recall=wr,
            w_f1=wf1,
            w_typ_macro_f1=wtf1,
            w_typ_accuracy=wtacc,
            typ_per_class=tpc,
        )


# ═══════════════════════════════════════════════════════════════════════
# HT-weighted metric functions
# ═══════════════════════════════════════════════════════════════════════

def ht_weighted_prf(
    preds: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    """Horvitz-Thompson weighted precision, recall, F1."""
    wtp = weights[(preds == 1) & (labels == 1)].sum()
    wfp = weights[(preds == 1) & (labels == 0)].sum()
    wfn = weights[(preds == 0) & (labels == 1)].sum()
    p = wtp / (wtp + wfp) if (wtp + wfp) > 0 else 0.0
    r = wtp / (wtp + wfn) if (wtp + wfn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def ht_weighted_typology(
    pred_typ: np.ndarray,
    gt_typ: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, dict]:
    """HT-weighted typology macro-F1 and accuracy over true-positive edges."""
    if mask.sum() == 0:
        return 0.0, 0.0, {}

    g = gt_typ[mask]
    p = pred_typ[mask]
    w = weights[mask]
    valid = (p != "unknown") & (p != "")
    if valid.sum() < 3:
        return 0.0, 0.0, {}

    g, p, w = g[valid], p[valid], w[valid]
    acc = w[g == p].sum() / w.sum() if w.sum() > 0 else 0.0

    per_class = {}
    f1_list = []
    for cls in TYPOLOGY_CLASSES:
        tp = w[(g == cls) & (p == cls)].sum()
        fp = w[(g != cls) & (p == cls)].sum()
        fn = w[(g == cls) & (p != cls)].sum()
        cp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cf = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0
        per_class[cls] = cf
        if (tp + fn) > 0:
            f1_list.append(cf)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    return macro_f1, acc, per_class


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def resolve_archive(archive: Optional[Path] = None) -> Path:
    """Locate the archived experiment tree used to regenerate results.

    Order: explicit argument, then ``$AMLC_ARCHIVE``. Only the
    regeneration paths need this; a plain clone does not.
    """
    if archive is not None:
        archive = Path(archive)
        return archive if archive.name == "outputs" else archive / "outputs"
    env = os.environ.get("AMLC_ARCHIVE")
    if env:
        return Path(env)
    raise FileNotFoundError(
        "No archived experiment tree configured. Set AMLC_ARCHIVE to the "
        "run directory, or load the shipped coreset with load_coreset()."
    )


def load_coreset(dataset: str = "HI-Small", **kw) -> dict:
    """Load the released HT-Coreset. Thin wrapper over :mod:`amlc.hub`.

    The coreset is not vendored in this repository; it is downloaded from the
    published dataset and cached. See :mod:`amlc.hub` for how to point at
    a fork or a local build instead.
    """
    from ..hub import load_coreset as _load
    return _load(dataset, **kw)


def verify_alignment(
    archive: Path, dataset: str, draw: str = "ht-coreset", strict: bool = True
) -> dict:
    """Check that a coreset draw matches the subset the prompts were built from.

    The serialised case directory holds the labels and weights for the exact
    edge ordering that generated the frozen ``case_id`` values. A draw whose
    labels disagree with those cannot be scored positionally against the LLM
    predictions.

    Returns a report dict; raises ValueError when ``strict`` and misaligned.
    """
    archive = Path(archive)
    idx = np.load(legacy_path(archive, "subset", dataset, draw=draw))
    w = np.load(legacy_path(archive, "weights", dataset, draw=draw))
    labels = np.load(legacy_path(archive, "test_labels", dataset))[idx]

    ser = legacy_path(archive, "serialised", dataset)
    report = {"dataset": dataset, "draw": draw, "n": int(len(idx))}
    if not (ser / "labels.npy").exists():
        report["checked"] = False
        return report

    ref_labels = np.load(ser / "labels.npy")
    ref_w = np.load(ser / "weights.npy")
    report["checked"] = True
    report["labels_match"] = bool(np.array_equal(labels, ref_labels))
    report["weights_match"] = bool(np.allclose(w, ref_w))
    report["label_mismatch_frac"] = float((labels != ref_labels).mean())

    if strict and not (report["labels_match"] and report["weights_match"]):
        raise ValueError(
            f"{dataset}: draw {draw!r} does not match the serialised subset the "
            f"prompts were built from. {report['label_mismatch_frac']:.1%} of "
            "positions carry a different ground-truth label. Use 'ht-coreset'."
        )
    return report


def load_coreset_from_archive(
    archive: Optional[Path] = None,
    dataset: str = "HI-Small",
    draw: str = "ht-coreset",
    verify: bool = True,
    repair_weights: bool = True,
) -> dict:
    """Rebuild the coreset from the archived experiment tree.

    Only needed to regenerate results or to reproduce the ARR submission.
    A plain clone should call :func:`load_coreset` instead.

    Parameters
    ----------
    draw : {"ht-coreset", "ablation-redraw"}
        ``ht-coreset`` is the released draw and the default. ``ablation-redraw``
        is the separate same-size re-draw behind the ARR-submission Doubt
        Triage numbers; it does not line up with the LLM predictions, so
        ``verify`` must be off to use it.
    repair_weights : bool
        Recompute the HT weights from the realised sample. The archived
        LI-Small vectors double-count almost the whole benign population
        because of a spare-fill defect in the generator; see
        ``coreset/ht_weights.py``. Leave on unless you are reproducing the
        submitted numbers exactly.
    """
    if draw not in CORESET_DRAWS:
        raise ValueError(f"draw must be one of {CORESET_DRAWS}, got {draw!r}")
    archive = resolve_archive(archive)

    if verify:
        verify_alignment(archive, dataset, draw, strict=True)

    idx = np.load(legacy_path(archive, "subset", dataset, draw=draw))
    weights = np.load(legacy_path(archive, "weights", dataset, draw=draw))
    labels = np.load(legacy_path(archive, "test_labels", dataset))[idx]
    typo_int = np.load(legacy_path(archive, "test_typologies", dataset))[idx]

    if repair_weights:
        from ..coreset.ht_weights import recompute_weights
        full_labels = np.load(legacy_path(archive, "test_labels", dataset))
        construction_scores = np.load(legacy_path(archive, "construction_probs", dataset))
        weights, _ = recompute_weights(
            idx, full_labels, construction_scores, CONSTRUCTION_THRESHOLDS[dataset])

    # Local import avoids the shared metrics module importing this module back.
    from ..baselines.ml.ensemble import load_evaluation_probabilities

    # Evaluation changes the scorer, never the frozen draw or its design weights.
    ens = load_evaluation_probabilities(archive, dataset)[idx]

    return {
        "source": "archive",
        "evaluation_members": list(ENSEMBLE_MEMBERS),
        "draw": draw,
        "weights_repaired": repair_weights,
        "subset_idx": idx,
        "labels": labels,
        "typologies": typo_int,
        "gt_typo_str": np.array([TYPOLOGY_INT_MAP.get(int(t), "legit") for t in typo_int]),
        "weights": weights,
        "ensemble_probs": ens,
        "ml_probs": ens,
        "ml_threshold": ML_THRESHOLDS[dataset],
        "n": len(idx),
    }


def load_ensemble_typology(
    archive: Optional[Path] = None,
    dataset: str = "HI-Small",
    subset_idx: np.ndarray = None,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Vote over evaluation heads; ties use the lowest canonical class index.

    With two members, a disagreement therefore uses the smaller stored index
    from ``amlc.typology``. Both heads are required so missing files cannot
    silently change the ensemble.
    """
    archive = resolve_archive(archive)
    preds = []
    for member in ENSEMBLE_MEMBERS:
        p = legacy_path(archive, "member_typology", dataset,
                        member=member_dir(member), seed=seed)
        if not p.exists():
            raise FileNotFoundError(f"missing evaluation typology head: {p}")
        values = np.load(p)
        if subset_idx is not None:
            values = values[subset_idx]
        if values.ndim != 1 or not np.isin(values, np.arange(len(TYPOLOGY_CLASSES))).all():
            raise ValueError(f"invalid canonical typology indices: {p}")
        preds.append(values)
    stacked = np.stack(preds)
    counts = np.stack([(stacked == i).sum(axis=0)
                       for i in range(len(TYPOLOGY_CLASSES))])
    # argmax chooses the first (smallest canonical) class in a tie.
    voted = counts.argmax(axis=0)
    return np.array([TYPOLOGY_INT_MAP[int(t)] for t in voted])


def load_llm_predictions(
    archive: Optional[Path] = None,
    dataset: str = "HI-Small",
    model: str = "GPT-OSS-120B",
    prompting: str = "ICL-FS",
    seed: int = 42,
    n: int = 0,
    *,
    strict: bool = False,
    runs_dir: Optional[Path] = None,
) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
    """Read one seed's predictions into arrays of length n.

    ``prompting`` takes a paper name (ICL-FS, ICL-ZS, ICL-V); the archived
    directory name is resolved by ``archive``.

    Prediction files carry the pre-release ``case_id`` prefix, because they are
    recorded output and are never rewritten. Either prefix parses; the numeric
    part is the row position in the released draw. See
    :mod:`amlc.case_ids`.

    ``strict=True`` requires a file with exactly one boolean or integer 0/1
    verdict for every row. This is used for fresh evaluations so a missing
    prediction cannot silently become a benign verdict. The default preserves
    the archived loader's permissive behavior.

    ``runs_dir`` reads that directory directly, as written by the runner's
    ``--out`` option, without the legacy archive ``outputs/`` resolution.
    """
    if runs_dir is not None:
        if archive is not None:
            raise ValueError("use either runs_dir or archive, not both")
        archive = Path(runs_dir)
    else:
        archive = resolve_archive(archive)
    p = legacy_path(archive, "predictions", dataset,
                    model=model, prompting=prompting, seed=seed)
    if not p.exists():
        if strict:
            raise FileNotFoundError(f"missing requested predictions: {p}")
        return None

    with open(p) as f:
        data = json.load(f)

    if strict:
        for key, expected in (("dataset", dataset), ("model", model),
                              ("method", prompting), ("seed", seed)):
            if key in data and data[key] != expected:
                raise ValueError(f"{p}: {key}={data[key]!r}, expected {expected!r}")
        records = data.get("predictions")
        if not isinstance(records, list):
            raise ValueError(f"{p}: predictions must be a list")
        positions = set()
        for pred in records:
            cid = pred.get("case_id", "") if isinstance(pred, dict) else ""
            if not isinstance(cid, str) or not is_case_id(cid):
                raise ValueError(f"{p}: invalid case_id {cid!r}")
            i = case_position(cid)
            if i >= n or i in positions:
                raise ValueError(f"{p}: duplicate or out-of-range case_id {cid!r}")
            value = pred.get("illicit")
            if type(value) not in (bool, int) or value not in (0, 1):
                raise ValueError(f"{p}: {cid} illicit must be boolean or integer 0/1")
            positions.add(i)
        if len(positions) != n:
            raise ValueError(f"{p}: expected {n} unique cases, found {len(positions)}")

    llm_bin = np.zeros(n, dtype=int)
    llm_typ = np.full(n, "unknown", dtype=object)
    seen = 0
    for pred in data.get("predictions", []):
        cid = pred.get("case_id", "")
        if is_case_id(cid):
            i = case_position(cid)
            if i < n:
                seen += 1
                is_illicit = bool(pred.get("illicit", False))
                llm_bin[i] = 1 if is_illicit else 0
                t = _normalize_typology_label(pred.get("typology", ""))
                if not t and is_illicit:
                    t = _extract_typology_from_raw(pred.get("raw_response", ""))
                if t:
                    llm_typ[i] = t

    tok = dict(data.get("token_stats", {}))
    tok["n_cases_matched"] = seen
    return llm_bin, llm_typ, tok
