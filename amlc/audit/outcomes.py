"""Re-derive the prediction-dependent columns of the audit tables.

Three columns of the four-step annotation CSVs are functions of the extracted
prediction rather than judgements about the trace: ``pred_typology``, the
``outcome`` class, and ``conclude``. When the typology extraction changed (the
Unicode-aware parser that recovers the non-breaking hyphen GPT-OSS emits inside
compound typology names), those three had to move with it. Parse, Recall and
Match do not: they are properties of the reasoning text and stay exactly as the
annotator scored them.

Conclude is re-derived under the uniform parser-extraction rule, the same rule
the human-rating pack applies: Conclude passes when the extracted prediction
matches ground truth on the binary verdict, and on the typology label where the
ground truth has one. The judge tables name their Conclude columns
``ds_conclude``, ``op_conclude`` or ``gm_conclude``; :func:`conclude_column`
finds the column whatever its prefix.

Applying the uniform rule has a consequence worth stating plainly, because it
changes a published table. Conclude under the uniform rule is a deterministic
function of the prediction and ground truth, identical for every annotator.
Applying it to all four files makes the four Conclude pass rates identical and
the Conclude row of the multi-judge agreement table degenerate: Fleiss' kappa
becomes 1 by construction. The per-judge Conclude spread the appendix reports
(24.0% for DeepSeek, 27.1% for Opus, 20.9% for Gemini against the heuristic's
20.4%) is each judge's own verdict. Run this stage when you want the uniform
rule; keep the judge verdicts when you want to measure whether judges agree.

Idempotent. Running it twice changes nothing the second time.

Reads
    The archived prediction JSONs for the extracted verdict and typology, and
    the annotation CSVs under ``results/audit/``.

Writes
    The same annotation CSVs, in place.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from .. import config
from ..case_ids import is_case_id, to_canonical
from ..archive import legacy_path
from .sample_traces import CORRECT_LEGIT

def annotation_files(n: int = config.RUBRIC_N_TRACES) -> list[str]:
    """Tables this stage maintains, in the order it must visit them.

    The heuristic table comes first: the judge tables carry no ground-truth
    columns and borrow them from it.
    """
    return [f"trace_4step_annotations_n{n}.csv"] + [
        f"trace_4step_{provider}_n{n}.csv"
        for provider in ("deepseek", "gemini", "opus")
    ]


def derive_outcome(pred_illicit: bool, pred_typology: str, gt_label: int,
                   gt_typology: str) -> str:
    """Outcome class for one (prediction, ground truth) pair.

    An illicit edge AMLworld leaves untyped cannot yield ``correct_illicit``:
    with no ground-truth typology there is nothing for the prediction to match,
    so a flagged verdict lands in ``typology_error``.
    """
    if gt_label == 0:
        return CORRECT_LEGIT if not pred_illicit else "over_prediction"
    if not pred_illicit:
        return "under_prediction"
    if gt_typology and (pred_typology or "") == gt_typology:
        return "correct_illicit"
    return "typology_error"


def conclude_column(fieldnames) -> Optional[str]:
    """Name of the Conclude column, whatever judge prefix it carries."""
    for name in fieldnames:
        if name == "conclude" or name.endswith("_conclude"):
            return name
    return None


def is_pass(value) -> bool:
    """Read a step score written as 1/0, True/False or yes/no.

    The heuristic table stores integers and the judge tables store booleans, so
    a string comparison would call every judge row changed on the first run
    even where the verdict is the same.
    """
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def load_predictions(archive: Path, dataset: str = "HI-Small",
                     prompting: str = "ICL-FS",
                     seed: int = config.RUBRIC_SEED) -> dict:
    """``{(model, case_id): (illicit, typology)}`` for the audited cells.

    Keys use the canonical ``amlc_`` identifier; the archived JSONs carry the
    pre-release prefix, which is never rewritten in place.
    """
    out: dict[tuple[str, str], tuple[bool, str]] = {}
    for model in config.LLM_MODELS:
        path = legacy_path(archive, "predictions", dataset, model=model,
                           prompting=prompting, seed=seed)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for pred in data.get("predictions", []):
            cid = pred.get("case_id", "")
            if is_case_id(cid):
                out[(model, to_canonical(cid))] = (pred.get("illicit"),
                                                   pred.get("typology") or "")
    return out


def build_gt_lookup(path: Path) -> dict:
    """``{(model, case_id): {gt_label, gt_typology}}`` from the heuristic table."""
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["case_id"])] = {
                "gt_label": row.get("gt_label", "0"),
                "gt_typology": row.get("gt_typology", "") or "",
            }
    return out


def update_one_csv(path: Path, predictions: dict, gt_lookup: dict) -> dict:
    """Rewrite one annotation table in place. Returns a change report."""
    if not path.exists():
        return {"skipped": True}
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"skipped": True}

    fieldnames = list(rows[0].keys())
    has_pred_typology = "pred_typology" in fieldnames
    conclude = conclude_column(fieldnames)

    report = {"total": len(rows), "typology_changed": 0, "conclude_changed": 0,
              "outcome_changed": 0, "unmatched": 0, "conclude_column": conclude}
    new_outcomes: Counter = Counter()

    for row in rows:
        key = (row.get("model"), row.get("case_id"))
        if key not in predictions:
            report["unmatched"] += 1
            new_outcomes[row.get("outcome", "")] += 1
            continue

        pred_illicit, pred_typology = predictions[key]
        gt_label_raw = row.get("gt_label", "")
        gt_typology = row.get("gt_typology", "") or ""
        if not gt_label_raw:
            # Judge tables hold no ground truth; borrow it from the heuristic.
            gt_row = gt_lookup.get(key)
            if gt_row is None:
                report["unmatched"] += 1
                new_outcomes[row.get("outcome", "")] += 1
                continue
            gt_label = int(gt_row["gt_label"])
            gt_typology = gt_row["gt_typology"]
        else:
            gt_label = int(gt_label_raw)

        if has_pred_typology and row.get("pred_typology", "") != pred_typology:
            report["typology_changed"] += 1
            row["pred_typology"] = pred_typology

        if conclude is not None:
            if gt_label == 0:
                value = 1 if not pred_illicit else 0
            else:
                value = 1 if (pred_illicit and gt_typology
                              and pred_typology == gt_typology) else 0
            if bool(value) != is_pass(row[conclude]):
                report["conclude_changed"] += 1
            row[conclude] = value

        outcome = derive_outcome(pred_illicit, pred_typology, gt_label, gt_typology)
        if row.get("outcome") != outcome:
            report["outcome_changed"] += 1
            row["outcome"] = outcome
        new_outcomes[outcome] += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report["outcomes"] = dict(new_outcomes)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path, default=None,
                    help="archived experiment tree (default: $AMLC_ARCHIVE)")
    ap.add_argument("--audit-dir", type=Path, default=None,
                    help="directory holding the annotation CSVs "
                         "(default: <repo>/results/audit)")
    ap.add_argument("--dataset", default=config.DATASETS[0], choices=config.DATASETS)
    ap.add_argument("--prompting", default="ICL-FS", choices=config.PROMPTINGS)
    ap.add_argument("--seed", type=int, default=config.RUBRIC_SEED)
    args = ap.parse_args()

    from ..paths import archive as archive_path
    from ..paths import ensure, results

    archive = Path(args.archive) if args.archive else archive_path(required=True)
    audit = Path(args.audit_dir) if args.audit_dir else ensure(results() / "audit")

    predictions = load_predictions(archive, args.dataset, args.prompting, args.seed)
    print(f"Loaded {len(predictions)} extracted predictions\n")

    files = annotation_files()
    gt_lookup = build_gt_lookup(audit / files[0])
    for name in files:
        report = update_one_csv(audit / name, predictions, gt_lookup)
        if name == files[0]:
            gt_lookup = build_gt_lookup(audit / name)
        if report.get("skipped"):
            print(f"  [skip] {name}")
            continue
        print(f"  {name}  (conclude column: {report['conclude_column']})")
        print(f"    typology changed: {report['typology_changed']}/{report['total']}")
        print(f"    conclude changed: {report['conclude_changed']}/{report['total']}")
        print(f"    outcome changed:  {report['outcome_changed']}/{report['total']}")
        if report["unmatched"]:
            print(f"    no prediction found: {report['unmatched']}")
        print(f"    outcomes: {report['outcomes']}\n")

    print("Conclude now follows the uniform parser-extraction rule in every "
          "table, so the four judges agree on it by construction. Rerun "
          "analyze_rubric for the summary, and read the module docstring before "
          "quoting the multi-judge Conclude agreement.")


if __name__ == "__main__":
    main()
