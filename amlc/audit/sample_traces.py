"""Draw the 1,000-trace audit slice and score it with the regex annotator.

This is the first stage of the four-step rubric audit. It samples reasoning
traces stratified by (model, outcome), applies the Parse / Recall / Match /
Conclude rubric with a keyword heuristic, and writes the table that every later
stage joins against: the three API judges in :mod:`amlc.audit.judges`, the
agreement statistics in :mod:`amlc.audit.iaa`, and the pass-rate analysis
in :mod:`amlc.audit.analyze_rubric`.

The rubric, in the same words the judges are given in
``prompts/four_step_rubric_system.j2``:

1. Parse, the trace names at least two structural features of the subgraph.
2. Recall, the trace enumerates at least two candidate typologies.
3. Match, a Suspicious verdict cites at least two structural markers. A Not
   Suspicious verdict passes trivially, because it asserts no typology.
4. Conclude, the verdict matches ground truth, and for illicit ground truth the
   typology label matches too.

Only Conclude compares against ground truth. The first three are properties of
the trace, which is why the heuristic can score them at all and why they stay
untouched when the typology extraction changes (see
:mod:`amlc.audit.outcomes`).

Reads
    The archived run directory: one prediction JSON per model for the
    HI-Small coreset under ICL-FS at seed 42, and the coreset labels and
    typologies behind them.

Writes
    ``results/audit/trace_4step_annotations_n1000.csv``, one row per sampled
    (model, case) pair with the four step scores, the outcome class and the
    first failing step.

Outcome vocabulary
------------------
The benign-and-correct class is ``correct_legit``, matching
``amlc.typology.BENIGN_LABEL`` and the released CSVs. The pre-release
sampler spelled it ``correct_benign`` while the recomputation stage spelled it
``correct_legit``, so the class silently vanished from any table that grouped by
name. :func:`normalise_outcome` accepts the old spelling on input.

Sample reproducibility
----------------------
Each (model, outcome) cell is drawn with a fixed ``random_state``, so the 980
stratified picks do not depend on model order. The 20-row top-up that brings the
sample to 1,000 draws from the pooled remainder and does depend on it; the order
is :data:`config.LLM_MODELS`.

Dropped
-------
``VOLUME_ONLY_PATTERNS``, a list of volume-only failure indicators that no
scoring function referenced. The Match step already fails a volume-only
justification by finding fewer than two structural markers.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config
from ..case_ids import is_case_id, position, to_canonical
from ..typology import TYPOLOGY_INT_MAP
from ..triage.doubt_triage import load_coreset_from_archive, resolve_archive

#: Outcome classes, mutually exclusive and exhaustive over (model, case) pairs.
CORRECT_LEGIT = "correct_legit"
OUTCOMES = (CORRECT_LEGIT, "correct_illicit", "over_prediction",
            "under_prediction", "typology_error")

#: The four classes the slice is stratified over. The benign-correct class is
#: left out: it is the most common and the least informative about failure.
SAMPLED_OUTCOMES = ("correct_illicit", "over_prediction",
                    "typology_error", "under_prediction")

#: Pre-release spelling of the benign-correct class, accepted on input.
_LEGACY_OUTCOME = {"correct_benign": CORRECT_LEGIT}


# ── Step 1: Parse ────────────────────────────────────────────────────────
# Structural features a trace has to name. Two of these are enough.
PARSE_KEYWORDS = [
    r"acct[_\-]\w+",                # account identifiers
    r"\bhub\b|\bcentral\b",         # hub identification
    r"\b(out[_\- ]?degree|in[_\- ]?degree|degree)\b",
    r"\b\d+\s+(transactions|edges|accounts|nodes)\b",
    r"\b(volume|total|sum)\b.*\d",
]

# ── Step 2: Recall ───────────────────────────────────────────────────────
# Typology names and the spellings models actually emit. Two distinct names
# count as enumerating candidates.
TYPOLOGY_NAMES = [
    "fan-out", "fan in", "fan-in", "fanout", "fanin",
    "scatter-gather", "scattergather", "gather-scatter", "gatherscatter",
    "cycle", "cyclic",
    "stack", "stacking", "layering",
    "bipartite",
    "random", "structuring", "smurfing",
]

# ── Step 3: Match ────────────────────────────────────────────────────────
# Structural markers, counted once per type.
MARKER_PATTERNS = {
    "temporal": [r"\b(temporal|timestamp|time[ \-]?bucket|monthly|biweekly|weekly|daily|recurring|periodic|interval)\b"],
    "amount":   [r"\b(identical amount|same amount|repeated.*amount|threshold|reporting threshold|just below)\b"],
    "cycle":    [r"\b(cycle|return.*origin|loop[s]?|back[ \-]?and[ \-]?forth)\b"],
    "layering": [r"\blayer(ing)?\b", r"\bmulti[\- ]?(hop|tier|step)\b"],
    "rapid":    [r"\b(rapid|quick|fast|burst)\b.*\b(transfer|movement)\b"],
    "counterparties": [r"\b(multiple|many|several).*counterpart"],
}


def score_parse(trace: str) -> bool:
    """Step 1: at least two structural features named."""
    if not trace or len(trace) < 50:
        return False
    score = 0
    for pat in PARSE_KEYWORDS:
        if re.search(pat, trace, re.IGNORECASE):
            score += 1
    return score >= 2


def score_recall(trace: str) -> bool:
    """Step 2: at least two distinct typology candidates enumerated."""
    if not trace:
        return False
    low = trace.lower()
    found = set()
    for name in TYPOLOGY_NAMES:
        if name in low:
            found.add(name.replace(" ", "-").replace("_", "-"))
    return len(found) >= 2


def score_match(trace: str, illicit: bool) -> bool:
    """Step 3: a Suspicious verdict cites at least two structural markers.

    A Not Suspicious verdict passes: the model asserts no typology, so there is
    no structural claim to verify.
    """
    if not trace:
        return False
    if not illicit:
        return True
    low = trace.lower()
    markers = 0
    for patterns in MARKER_PATTERNS.values():
        for pat in patterns:
            if re.search(pat, low, re.IGNORECASE):
                markers += 1
                break  # count each marker type once
    return markers >= 2


def score_conclude(pred_illicit: bool, pred_typology, gt_label: int,
                   gt_typology: str) -> bool:
    """Step 4: the verdict matches ground truth.

    Benign ground truth passes on a benign verdict. Illicit ground truth needs
    both the verdict and the typology label. This is the uniform
    parser-extraction rule the human-rating pass also applies.
    """
    if gt_label == 0:
        return not bool(pred_illicit)
    if not pred_illicit:
        return False
    if pred_typology is None or (isinstance(pred_typology, float)
                                 and np.isnan(pred_typology)):
        return False
    return str(pred_typology).lower() == str(gt_typology).lower()


def annotate_trace(trace: str, pred: dict, gt_label: int, gt_typology: str) -> dict:
    """Score one trace against all four steps."""
    illicit = bool(pred.get("illicit"))
    return {
        "parse":    score_parse(trace),
        "recall":   score_recall(trace),
        "match":    score_match(trace, illicit),
        "conclude": score_conclude(illicit, pred.get("typology"), gt_label,
                                   gt_typology),
    }


def first_fail(scores) -> str:
    """First step that fails, in pipeline order, or ``all_pass``."""
    for step in config.RUBRIC_STEPS:
        if int(scores[step]) == 0:
            return step
    return "all_pass"


# ── Outcome classes ──────────────────────────────────────────────────────

def classify_outcome(pred_illicit: bool, pred_typology, gt_label: int,
                     gt_typology: str) -> str:
    """Assign one of :data:`OUTCOMES` to a (prediction, ground truth) pair."""
    if gt_label == 0 and not pred_illicit:
        return CORRECT_LEGIT
    if gt_label == 1 and pred_illicit and pred_typology == gt_typology:
        return "correct_illicit"
    if gt_label == 0 and pred_illicit:
        return "over_prediction"
    if gt_label == 1 and not pred_illicit:
        return "under_prediction"
    if gt_label == 1 and pred_illicit and pred_typology != gt_typology:
        return "typology_error"
    return "unclear"


def normalise_outcome(outcome: str) -> str:
    """Map a pre-release outcome spelling onto the canonical one."""
    return _LEGACY_OUTCOME.get(outcome, outcome)


# ── Inputs ───────────────────────────────────────────────────────────────

def load_traces(archive: Path, dataset: str, model: str, prompting: str,
                seed: int) -> Optional[dict]:
    """One model's predictions, keyed by canonical case identifier.

    The archived JSONs carry the pre-release ``v2_`` prefix, because recorded
    output is never rewritten in place; :func:`amlc.case_ids.to_canonical`
    maps it onto the ``amlc_`` identifiers the released tables use.
    """
    import json

    from ..archive import legacy_path

    p = legacy_path(archive, "predictions", dataset, model=model,
                    prompting=prompting, seed=seed)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return {to_canonical(rec["case_id"]): rec
            for rec in data.get("predictions", [])
            if is_case_id(rec.get("case_id", ""))}


def ground_truth(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-position labels and typology names, with "" for untyped edges.

    Benign edges and the 460 illicit HI-Small edges AMLworld leaves untyped both
    carry -1 in the typology array. The rubric treats both as "no typology to
    match", which is what the released annotation CSVs record.
    """
    labels = np.asarray(data["labels"])
    typo = np.asarray(data["typologies"])
    names = np.array(["" if int(t) < 0 else TYPOLOGY_INT_MAP[int(t)] for t in typo],
                     dtype=object)
    return labels, names


# ── Sampling ─────────────────────────────────────────────────────────────

def build_frame(archive: Path, data: dict, dataset: str, prompting: str,
                seed: int) -> pd.DataFrame:
    """Every (model, case) pair with its outcome class and trace length."""
    labels, gt_typologies = ground_truth(data)
    rows = []
    for model in config.LLM_MODELS:
        preds = load_traces(archive, dataset, model, prompting, seed)
        if preds is None:
            continue
        for case_id, rec in preds.items():
            i = position(case_id)
            if i >= len(labels):
                continue
            gt_label = int(labels[i])
            gt_typology = gt_typologies[i]
            rows.append({
                "model": model,
                "case_id": case_id,
                "outcome": classify_outcome(bool(rec.get("illicit")),
                                            rec.get("typology"),
                                            gt_label, gt_typology),
                "pred_illicit": bool(rec.get("illicit")),
                "pred_typology": rec.get("typology"),
                "gt_label": gt_label,
                "gt_typology": gt_typology,
                "trace_len": len(rec.get("reasoning_content", "") or ""),
            })
    return pd.DataFrame(rows)


def stratified_sample(frame: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    """Even draw per (model, outcome) cell, topped up to ``n_target`` rows."""
    per_cell = max(1, n_target // (len(config.LLM_MODELS) * len(SAMPLED_OUTCOMES)))

    picked = []
    for model in config.LLM_MODELS:
        for outcome in SAMPLED_OUTCOMES:
            cell = frame[(frame.model == model) & (frame.outcome == outcome)]
            if len(cell) == 0:
                continue
            picked.append(cell.sample(n=min(len(cell), per_cell), random_state=seed))
    sampled = pd.concat(picked, ignore_index=True)

    if len(sampled) < n_target:
        # Top up from every pair not already taken. The pre-release version
        # prefixed this with a case-id filter OR'd against a model-membership
        # test that is true for every row, so it selected nothing; the pair
        # exclusion below is what did the work.
        taken = sampled.set_index(["model", "case_id"]).index
        remaining = frame[~frame.set_index(["model", "case_id"]).index.isin(taken)]
        extra = remaining.sample(n=min(n_target - len(sampled), len(remaining)),
                                 random_state=seed + 1)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.reset_index(drop=True)


def annotate(archive: Path, sampled: pd.DataFrame, dataset: str, prompting: str,
             seed: int) -> pd.DataFrame:
    """Apply the rubric to every sampled trace."""
    cache: dict[str, dict] = {}
    scores = []
    for row in sampled.itertuples():
        if row.model not in cache:
            cache[row.model] = load_traces(archive, dataset, row.model,
                                           prompting, seed) or {}
        rec = cache[row.model][row.case_id]
        trace = rec.get("reasoning_content", "") or ""
        scores.append(annotate_trace(trace, rec, row.gt_label, row.gt_typology))

    out = pd.concat([sampled, pd.DataFrame(scores)], axis=1)
    for step in config.RUBRIC_STEPS:
        out[step] = out[step].astype(int)
    out["first_fail"] = out.apply(first_fail, axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path, default=None,
                    help="archived experiment tree (default: $AMLC_ARCHIVE)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output CSV (default: <repo>/results/audit/...)")
    ap.add_argument("--dataset", default=config.DATASETS[0], choices=config.DATASETS)
    ap.add_argument("--prompting", default="ICL-FS", choices=config.PROMPTINGS)
    ap.add_argument("--n", type=int, default=config.RUBRIC_N_TRACES)
    ap.add_argument("--seed", type=int, default=config.RUBRIC_SEED)
    ap.add_argument("--no-annotate", action="store_true",
                    help="write the sample without applying the rubric")
    args = ap.parse_args()

    from ..paths import ensure, results

    archive = resolve_archive(args.archive)
    data = load_coreset_from_archive(archive, args.dataset)

    frame = build_frame(archive, data, args.dataset, args.prompting, args.seed)
    print(f"{len(frame)} (model, case) pairs across {frame.model.nunique()} models")
    print(frame.outcome.value_counts().to_string())

    sampled = stratified_sample(frame, args.n, args.seed)
    print(f"\nSampled {len(sampled)} traces")
    print(sampled.groupby(["model", "outcome"]).size().unstack(fill_value=0).to_string())

    out = sampled if args.no_annotate else annotate(
        archive, sampled, args.dataset, args.prompting, args.seed)

    path = Path(args.out) if args.out else (
        ensure(results() / "audit") / f"trace_4step_annotations_n{args.n}.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved {path} ({len(out)} rows)")

    if args.no_annotate:
        return
    print("\nPer-step pass rate")
    for step in config.RUBRIC_STEPS:
        print(f"  {step:10s} {out[step].mean() * 100:5.1f}%")
    print("\nFirst failing step")
    print(out.first_fail.value_counts().to_string())
    print("\nPer-model pass rate")
    per_model = out.groupby("model")[list(config.RUBRIC_STEPS)].mean() * 100
    print(per_model.round(1).to_string())


if __name__ == "__main__":
    main()
