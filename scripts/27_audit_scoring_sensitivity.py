#!/usr/bin/env python3
"""Reproduce offline trace scoring sensitivity without API calls or raw traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc import paths
from amlc.analysis.audit_sensitivity import (
    JUDGES,
    STEPS,
    attach_labels,
    prepare_features,
    read_annotations,
    render_readme,
    sha256,
    source_contract,
    summarize,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=paths.results() / "audit")
    parser.add_argument("--inputs", type=Path,
                        default=paths.results() / "analysis/audit_sensitivity/inputs")
    parser.add_argument("--out", type=Path,
                        default=paths.results() / "analysis/audit_sensitivity")
    parser.add_argument("--archive-root", type=Path,
                        help="Optional original outputs directory; rebuild portable features locally")
    args = parser.parse_args(argv)
    filenames = ["trace_4step_annotations_n1000.csv"] + [
        f"trace_4step_{judge}_n1000.csv" for judge in JUDGES]
    hashes = {name: sha256(args.audit_dir / name) for name in filenames}
    annotations = read_annotations(args.audit_dir / filenames[0])
    feature_path = args.inputs / "trace_features.csv"
    provenance_path = args.inputs / "provenance.json"
    if args.archive_root:
        features, provenance = prepare_features(annotations, args.archive_root)
        args.inputs.mkdir(parents=True, exist_ok=True)
        features.to_csv(feature_path, index=False)
        provenance.update({"annotation_sha256": hashes,
                           "features_sha256": sha256(feature_path),
                           "scope": "HI-Small / ICL-FS / seed 42 / frozen 1000-row audit",
                           "feature_module_sha256": sha256(Path(__file__).resolve().parent.parent
                                                             / "amlc/analysis/audit_sensitivity.py")})
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    else:
        provenance = json.loads(provenance_path.read_text())
        if provenance["annotation_sha256"] != hashes:
            raise ValueError("annotation input hashes differ from portable-feature provenance")
        if provenance["features_sha256"] != sha256(feature_path):
            raise ValueError("portable feature checksum mismatch")
        contract = source_contract()
        if any(provenance[key] != contract[key] for key in
               ("definitions_sha256", "original_scorer_sha256")):
            raise ValueError("lexical scoring definitions or original scorer changed")
        features = pd.read_csv(feature_path, keep_default_na=False)
    frame = attach_labels(annotations, features)
    judges = {judge: pd.read_csv(args.audit_dir / f"trace_4step_{judge}_n1000.csv",
                                keep_default_na=False) for judge in JUDGES}
    tables = summarize(frame, judges)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "items.csv", index=False)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    summary = {
        "scope": provenance["scope"],
        "interpretation": "Outcome-stratified descriptive slice; no causal stage attribution, "
                          "population inference, new model calls, or semantic rescoring.",
        "input_sha256": {**hashes, "inputs/trace_features.csv": sha256(feature_path),
                         "inputs/provenance.json": sha256(provenance_path)},
        "script_sha256": sha256(Path(__file__)),
        "module_sha256": sha256(Path(__file__).resolve().parent.parent
                                / "amlc/analysis/audit_sensitivity.py"),
        "n": len(frame), "models": sorted(frame.model.unique()),
        "reference_group_counts": frame.reference_group.value_counts().to_dict(),
        "error_class_counts": frame.error_class.value_counts().to_dict(),
        "integrity": {
            "trace_length_disagreements": int(frame.trace_len.ne(frame.trace_chars).sum()),
            "archive_prediction_disagreements": int(frame.archive_prediction_agrees.eq(0).sum()),
            "archived_conclude_vs_evaluable_rule_disagreements": int(
                (frame.reference_evaluable & frame.conclude.ne(frame.evaluable_correct)).sum()),
            "full_vs_archived_step_disagreements": {
                step: int(frame[step].ne(frame[f"full_{step}"].astype(bool)).sum())
                for step in STEPS},
        },
        "overall": {
            name: json.loads(table[table.model.eq("ALL") & table.group_kind.eq("all")]
                             .to_json(orient="records"))
            for name, table in tables.items()},
        "output_sha256": {name: sha256(args.out / name) for name in
                          ["items.csv"] + [f"{key}.csv" for key in tables]},
    }
    (args.out / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.out / "README.md").write_text(render_readme(frame, tables))
    print(json.dumps({"n": len(frame), "out": str(args.out),
                      "integrity": summary["integrity"]}, indent=2))


if __name__ == "__main__":
    main()
