#!/usr/bin/env python3
"""Validate cached predictions and export portable sampling-analysis inputs.

This stage reads existing files only; it never draws targets or calls a model.
The archive argument is the directory containing ``eval_subsets``,
``test_probs``, and model output directories. The coreset root is either the
released dataset root (containing ``extras``) or the ``extras`` directory.
All four paths are explicit so archive provenance is not tied to a home path.

Each split produces an NPZ with a binary prediction tensor indexed by target,
model/prompting cell, and inference seed, a target-stratum manifest CSV, and
metadata including population sizes, source hashes, and validation results.
The numerical analysis needs only these small derived inputs, not raw text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amlc.archive import legacy_path, prompting_dir
from amlc.case_ids import case_id, is_case_id, position
from amlc.config import CONSTRUCTION_THRESHOLDS, DATASETS, LLM_MODELS, SEEDS
from amlc.coreset.ht_weights import assign_strata
from amlc.triage.doubt_triage import ht_weighted_prf

PROMPT_ORDER = ("ICL-FS", "ICL-ZS")
METRICS = ("subset_p", "subset_r", "subset_f1", "ht_p", "ht_r", "ht_f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path, root: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def score(predictions: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> dict:
    compact = ht_weighted_prf(predictions, labels, np.ones(len(labels)))
    weighted = ht_weighted_prf(predictions, labels, weights)
    return {"n_flagged": int(predictions.sum()),
            **dict(zip(METRICS, (100 * x for x in (*compact, *weighted))))}


def verify_scores(computed: dict, reference: pd.Series, context: str) -> float:
    if computed["n_flagged"] != int(reference["n_flagged"]):
        raise ValueError(f"{context}: n_flagged does not reproduce the reference")
    actual = np.array([computed[key] for key in METRICS])
    expected = np.array([reference[key] for key in METRICS], dtype=float)
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{context}: scores do not reproduce the reference CSV")
    return float(np.max(np.abs(actual - expected)))


def prepare_split(dataset: str, archive: Path, extras: Path,
                  reference: pd.DataFrame, reference_source: dict, out: Path) -> dict:
    released = extras / dataset
    archive_paths = {
        "full_labels": legacy_path(archive, "test_labels", dataset),
        "construction_probs": legacy_path(archive, "construction_probs", dataset),
        "subset_idx": legacy_path(archive, "subset", dataset, draw="ht-coreset"),
        "original_weights": legacy_path(archive, "weights", dataset, draw="ht-coreset"),
    }
    released_paths = {
        name: released / filename for name, filename in {
            "labels": "labels.npy", "weights": "ht_weights.npy",
            "subset_idx": "ht_subset_indices.npy",
            "construction_probs": "construction_probs_coreset.npy",
            "case_index": "case_index.csv",
        }.items()
    }
    full_labels = np.load(archive_paths["full_labels"], allow_pickle=False)
    construction = np.load(archive_paths["construction_probs"], allow_pickle=False)
    subset = np.load(released_paths["subset_idx"], allow_pickle=False)
    labels = np.load(released_paths["labels"], allow_pickle=False)
    weights = np.load(released_paths["weights"], allow_pickle=False)
    n = len(subset)
    if not np.issubdtype(subset.dtype, np.integer):
        raise ValueError(f"{dataset}: subset indices must be integers")
    if len(np.unique(subset)) != n or np.any((subset < 0) | (subset >= len(full_labels))):
        raise ValueError(f"{dataset}: duplicate or out-of-range released target indices")
    archived_subset = np.load(archive_paths["subset_idx"], allow_pickle=False)
    if (subset.dtype != archived_subset.dtype or subset.shape != archived_subset.shape
            or subset.tobytes() != archived_subset.tobytes()):
        raise ValueError(f"{dataset}: released targets differ bit-for-bit from archive")
    if not np.array_equal(labels, full_labels[subset]):
        raise ValueError(f"{dataset}: released labels are not aligned with full labels")
    if not np.array_equal(construction[subset], np.load(
            released_paths["construction_probs"], allow_pickle=False)):
        raise ValueError(f"{dataset}: released construction scores do not match archive")
    if not np.all(np.isin(full_labels, (0, 1))) or construction.shape != full_labels.shape:
        raise ValueError(f"{dataset}: invalid full-label/construction array shapes or labels")
    if not np.all(np.isfinite(construction)):
        raise ValueError(f"{dataset}: construction scores must be finite")

    pools = {"illicit": np.flatnonzero(full_labels == 1), **assign_strata(
        full_labels, construction, CONSTRUCTION_THRESHOLDS[dataset])}
    stratum_ids = np.full(n, "", dtype="U8")
    rebuilt_weights = np.empty(n, dtype=np.float64)
    population_sizes = {name: len(pool) for name, pool in pools.items()}
    sample_sizes = {}
    for name, pool in pools.items():
        selected = np.isin(subset, pool)
        count = int(selected.sum())
        if not count or np.any(stratum_ids[selected] != ""):
            raise ValueError(f"{dataset}/{name}: empty or overlapping selected stratum")
        sample_sizes[name] = count
        stratum_ids[selected] = name
        rebuilt_weights[selected] = len(pool) / count
        if name in ("illicit", "hard_neg") and count != len(pool):
            raise ValueError(f"{dataset}/{name}: expected complete census")
    if np.any(stratum_ids == "") or sum(population_sizes.values()) != len(full_labels):
        raise ValueError(f"{dataset}: strata do not partition the full population")
    if (weights.dtype != rebuilt_weights.dtype or weights.shape != rebuilt_weights.shape
            or weights.tobytes() != rebuilt_weights.tobytes()):
        raise ValueError(f"{dataset}: reconstructed weights differ bit-for-bit from release")

    cases = pd.read_csv(released_paths["case_index"], keep_default_na=False)
    expected_case_ids = np.array([case_id(i) for i in range(n)])
    if (len(cases) != n or not np.array_equal(cases.case_id.values, expected_case_ids)
            or not np.array_equal(cases.subset_index.values, subset)
            or not np.array_equal(cases.label.values, labels)):
        raise ValueError(f"{dataset}: released case index does not match target arrays")
    center_ids = cases.center_edge_id.to_numpy(dtype=str)
    if len(np.unique(center_ids)) != n:
        raise ValueError(f"{dataset}: canonical center-edge IDs are not unique")

    cells = [(model, prompt) for model in LLM_MODELS for prompt in PROMPT_ORDER]
    predictions = np.empty((n, len(cells), len(SEEDS)), dtype=np.uint8)
    sources, validation_rows = [], []
    reference_block = reference[reference.dataset == dataset]
    if reference_block.duplicated(["model", "prompting", "seed"]).any():
        raise ValueError(f"{dataset}: duplicate reference score rows")
    indexed = reference_block.set_index(["model", "prompting", "seed"])
    max_error = 0.0
    for cell, (model, prompt) in enumerate(cells):
        for seed_index, seed in enumerate(SEEDS):
            path = legacy_path(archive, "predictions", dataset,
                               model=model, prompting=prompt, seed=seed)
            raw = path.read_bytes()
            data = json.loads(raw)
            source = {"path": str(path.relative_to(archive)),
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            del raw
            for key, expected in (("dataset", dataset), ("model", model), ("seed", seed)):
                if key in data and data[key] != expected:
                    raise ValueError(f"{path}: unexpected {key} metadata")
            if data.get("method", prompting_dir(prompt)) not in (prompt, prompting_dir(prompt)):
                raise ValueError(f"{path}: unexpected method metadata")
            records = data.get("predictions")
            if not isinstance(records, list):
                raise TypeError(f"{path}: predictions must be a list")
            seen = np.zeros(n, dtype=bool)
            counts = {"missing_illicit_field": 0, "null_illicit": 0,
                      "non_binary_illicit": 0, "missing_center_edge_id": 0,
                      "checked_center_edge_id": 0, "checked_gt_label": 0}
            for record in records:
                identifier = record.get("case_id") if isinstance(record, dict) else None
                if not isinstance(identifier, str) or not is_case_id(identifier):
                    raise ValueError(f"{path}: invalid case identifier")
                i = position(identifier)
                if i >= n or seen[i]:
                    raise ValueError(f"{path}: duplicate or out-of-range target {identifier}")
                seen[i] = True
                if "center_edge_id" in record and record["center_edge_id"] is not None:
                    if record["center_edge_id"] != center_ids[i]:
                        raise ValueError(f"{path}: mismatched center edge for {identifier}")
                    counts["checked_center_edge_id"] += 1
                else:
                    counts["missing_center_edge_id"] += 1
                if "gt_label" in record and record["gt_label"] is not None:
                    if record["gt_label"] != int(labels[i]):
                        raise ValueError(f"{path}: mismatched ground truth for {identifier}")
                    counts["checked_gt_label"] += 1
                value = record.get("illicit", False)
                counts["missing_illicit_field"] += int("illicit" not in record)
                counts["null_illicit"] += int("illicit" in record and value is None)
                counts["non_binary_illicit"] += int(
                    value is not None and not (type(value) in (bool, int) and value in (0, 1)))
                # Preserve precisely the convention that produced the current CSV.
                predictions[i, cell, seed_index] = int(bool(value))
            if not seen.all():
                missing = np.flatnonzero(~seen)
                raise ValueError(f"{path}: {len(missing)} missing target records; first positions "
                                 f"{missing[:10].tolist()}. No absent targets were imputed.")
            computed = score(predictions[:, cell, seed_index], labels, weights)
            error = verify_scores(computed, indexed.loc[(model, prompt, seed)], str(path))
            max_error = max(max_error, error)
            validation_rows.append({"model": model, "prompting": prompt, "seed": int(seed),
                                    "records": len(records), **counts,
                                    "maximum_score_error": error})
            sources.append(source)
        print(f"Validated {dataset}: {model} {prompt}, {len(SEEDS)} seeds", flush=True)
    floor = reference_block[reference_block.model == "predict-all-illicit"]
    if len(floor) != 1:
        raise ValueError(f"{dataset}: expected one predict-all reference row")
    max_error = max(max_error, verify_scores(
        score(np.ones(n, dtype=np.uint8), labels, weights), floor.iloc[0], dataset + " floor"))

    npz_path = out / f"{dataset}.npz"
    np.savez_compressed(npz_path, predictions=predictions, labels=labels,
                        stratum_ids=stratum_ids, subset_idx=subset, weights=weights,
                        models=np.array([cell[0] for cell in cells]),
                        promptings=np.array([cell[1] for cell in cells]),
                        seeds=np.array(SEEDS, dtype=np.int64), case_ids=expected_case_ids,
                        center_edge_ids=center_ids)
    manifest_path = out / f"{dataset}_strata.csv"
    with manifest_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset", "position", "case_id", "center_edge_id", "subset_index",
                         "label", "stratum_id", "population_size", "sample_size", "weight"))
        for i, name in enumerate(stratum_ids):
            writer.writerow((dataset, i, expected_case_ids[i], center_ids[i], int(subset[i]),
                             int(labels[i]), name, population_sizes[name], sample_sizes[name],
                             repr(float(weights[i]))))
    metadata = {
        "schema_version": 1, "dataset": dataset,
        "tensor_axes": ["released_target_position", "model_prompting_cell", "inference_seed"],
        "prediction_shape": list(predictions.shape),
        "prediction_rule": "int(bool(record.get('illicit', False)))",
        "missing_target_rule": "error; no target records imputed",
        "population_sizes": population_sizes, "sample_sizes": sample_sizes,
        "full_population_size": len(full_labels), "n_illicit": int(labels.sum()),
        "construction_threshold": CONSTRUCTION_THRESHOLDS[dataset],
        "sampling_conditioning": (
            "Fixed construction strata, fixed contexts and per-edge prediction vectors; "
            "conditional on realized stratum allocation. LI-Small includes one residual-fill "
            "target in easy_t1; the original archived LI weights are not used for scoring."),
        "archive_sources": {key: source_record(path, archive)
                            for key, path in archive_paths.items()},
        "released_sources": {key: source_record(path, extras)
                             for key, path in released_paths.items()},
        "prediction_sources": sources, "reference_scores": reference_source,
        "preparation_script": {"filename": Path(__file__).name,
                               "sha256": sha256(Path(__file__))},
        "validation": {"targets_match_archive": True, "labels_match_archive": True,
                       "targets_bitwise_equal_archive": True,
                       "construction_scores_match_archive": True,
                       "reconstructed_weights_bitwise_equal": True,
                       "census_strata_complete": True, "all_target_records_present": True,
                       "reference_runs_reproduced": len(validation_rows),
                       "predict_all_reference_reproduced": True,
                       "maximum_absolute_metric_error_percentage_points": max_error,
                       "runs": validation_rows},
        "outputs": {"npz": source_record(npz_path, out),
                    "stratum_manifest": source_record(manifest_path, out)},
    }
    (out / f"{dataset}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--coreset-root", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.resolve()
    extras = args.coreset_root.resolve()
    if (extras / "extras").is_dir():
        extras /= "extras"
    reference = pd.read_csv(args.reference_csv)
    reference_source = {"filename": args.reference_csv.name,
                        "sha256": sha256(args.reference_csv)}
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [prepare_split(dataset, archive, extras, reference,
                               reference_source, args.out) for dataset in DATASETS]
    print(json.dumps({item["dataset"]: {
        "shape": item["prediction_shape"], "population_sizes": item["population_sizes"],
        "maximum_score_error": item["validation"][
            "maximum_absolute_metric_error_percentage_points"]} for item in summaries}, indent=2))


if __name__ == "__main__":
    main()
