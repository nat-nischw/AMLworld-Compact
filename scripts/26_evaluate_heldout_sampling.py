#!/usr/bin/env python3
"""Run the frozen CPU-only cached GFP-tree held-out sampling experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

PACKAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE))

from amlc.analysis.heldout_sampling import (
    METRICS,
    SAMPLERS,
    cached_count_interval,
    constructor_pool,
    draw_counts,
    evaluate_counts,
    make_design,
    metrics_from_counts,
    summarize_draws,
    threshold_predictions,
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _load_family(inputs, dataset, family, seeds):
    probabilities = np.column_stack([
        np.load(inputs / "test_probs" / dataset / family / f"seed_{seed}.npy")
        for seed in seeds])
    thresholds = []
    for seed in seeds:
        meta = json.loads((inputs / "weights" / dataset / family /
                           f"seed_{seed}_meta.json").read_text())
        if meta["dataset"] != dataset or meta["seed"] != seed or meta["method"] != family:
            raise ValueError("saved validation metadata does not match its path")
        thresholds.append(meta["threshold"])
    return probabilities, np.asarray(thresholds)


def _metric_rows(base, stats):
    return [dict(base, metric=metric, **{
        key: float(values[index]) for key, values in stats.items()})
        for index, metric in enumerate(METRICS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=PACKAGE.parent / "amlcompact-dataset" / "ml_baselines")
    parser.add_argument("--out", type=Path,
                        default=PACKAGE / "results" / "analysis" / "heldout_sampling")
    parser.add_argument("--protocol", type=Path,
                        default=PACKAGE / "results" / "analysis" / "heldout_sampling" /
                        "protocol.json")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if (protocol["model_seeds"] != [42, 123, 456, 789, 1011]
            or protocol["samplers"] != list(SAMPLERS)
            or protocol["draws_per_configuration"] != 500):
        raise ValueError("runner requires the frozen five-seed, 500-draw protocol")
    if (out / "protocol.json").resolve() != args.protocol.resolve():
        (out / "protocol.json").write_bytes(protocol_bytes)
    seeds = protocol["model_seeds"]
    input_paths = []
    families = sorted({family for pair in protocol["directions"] for family in pair})
    for dataset in protocol["datasets"]:
        input_paths.append(args.inputs / "test_probs" / dataset / "test_labels.npy")
        for family in families:
            for seed in seeds:
                input_paths.extend([
                    args.inputs / "test_probs" / dataset / family / f"seed_{seed}.npy",
                    args.inputs / "weights" / dataset / family / f"seed_{seed}_meta.json",
                ])
    source_paths = [
        (Path(__file__), "scripts/26_evaluate_heldout_sampling.py", "package"),
        (PACKAGE / "amlc/analysis/heldout_sampling.py", "amlc/analysis/heldout_sampling.py", "package"),
        (PACKAGE / "amlc/analysis/sampling_uncertainty.py",
         "amlc/analysis/sampling_uncertainty.py", "package"),
        (args.protocol, "protocol.json", "output"),
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),  # timezone.utc also supports the local Python 3.10 runner
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "input_path_base": "The --inputs ml_baselines directory",
        "source_path_bases": {"package": "amlcompact package directory", "output": "--out"},
        "inputs": [{"path": path.relative_to(args.inputs).as_posix(),
                    "bytes": path.stat().st_size, "sha256": _sha256(path)}
                   for path in input_paths],
        "sources": [{"path": name, "base": base, "sha256": _sha256(path)}
                    for path, name, base in source_paths],
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
    }
    _write_json(out / "input_manifest.json", manifest)
    started = perf_counter()
    summary_rows, per_seed_rows, truth_rows = [], [], []
    design_rows, diagnostics, mean_draw_rows = [], [], []
    for ds_i, dataset in enumerate(protocol["datasets"]):
        labels = np.load(args.inputs / "test_probs" / dataset / "test_labels.npy")
        if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
            raise ValueError("invalid labels")
        illicit = int(labels.sum())
        benign = len(labels) - illicit
        for direction_i, (constructor, evaluator) in enumerate(protocol["directions"]):
            constructor_probs, constructor_thresholds = _load_family(
                args.inputs, dataset, constructor, seeds)
            scores, tau = constructor_pool(constructor_probs, constructor_thresholds)
            del constructor_probs
            # Freeze every design before accessing this direction's evaluation vectors.
            designs = {}
            for multiplier in protocol["benign_budget_multipliers"]:
                budget = min(multiplier * illicit, benign)
                for sampler in SAMPLERS:
                    designs[multiplier, sampler] = make_design(labels, scores, tau, budget, sampler)
            evaluator_probs, evaluator_thresholds = _load_family(
                args.inputs, dataset, evaluator, seeds)
            predictions = threshold_predictions(evaluator_probs, evaluator_thresholds)
            del evaluator_probs
            if len(predictions) != len(labels):
                raise ValueError("prediction/label alignment requires equal row counts")
            tp = predictions[labels == 1].sum(axis=0)
            full_fp = predictions[labels == 0].sum(axis=0)
            truth = metrics_from_counts(tp, illicit, full_fp)
            base = {"dataset": dataset, "constructor": constructor, "evaluator": evaluator}
            for seed_i, seed in enumerate(seeds):
                truth_rows.append(dict(base, seed=seed, threshold=evaluator_thresholds[seed_i],
                                       population_size=len(labels), illicit_count=illicit,
                                       tp=int(tp[seed_i]), fp=int(full_fp[seed_i]),
                                       **dict(zip(METRICS, truth[seed_i]))))
            truth_rows.append(dict(base, seed="mean", threshold=float(evaluator_thresholds.mean()),
                                   population_size=len(labels), illicit_count=illicit,
                                   tp=float(tp.mean()), fp=float(full_fp.mean()),
                                   **dict(zip(METRICS, truth.mean(axis=0)))))
            hard = (labels == 0) & (scores >= tau / 2)
            outside = (labels == 0) & ~hard
            for seed_i, seed in enumerate(seeds):
                diagnostics.append(dict(base, seed=seed, constructor_tau=tau,
                                        hard_cutoff=tau / 2, hard_population=int(hard.sum()),
                                        full_fp=int(full_fp[seed_i]),
                                        fp_in_hard=int(predictions[hard, seed_i].sum()),
                                        fp_outside_hard=int(predictions[outside, seed_i].sum())))
            for multiplier in protocol["benign_budget_multipliers"]:
                budget = min(multiplier * illicit, benign)
                for sampler_i, sampler in enumerate(SAMPLERS):
                    design = designs[multiplier, sampler]
                    config = dict(base, benign_multiplier=multiplier, sampler=sampler,
                                  benign_budget=budget, total_sample=illicit + budget,
                                  draws=protocol["draws_per_configuration"])
                    components = len(seeds) * sum(
                        s.sample_size < s.population_size for s in design)
                    for stratum in design:
                        design_rows.append(dict(
                            config, stratum=stratum.name,
                            population_size=stratum.population_size,
                            sample_size=stratum.sample_size,
                            inclusion_probability=stratum.inclusion_probability,
                            ht_weight=1 / stratum.inclusion_probability,
                            constructor_tau=tau, hard_cutoff=tau / 2,
                            n_simultaneous_count_intervals=components,
                            component_alpha=.05 / components if components else 0))
                    points, lower, upper = [], [], []
                    fp_points = []
                    for draw in range(protocol["draws_per_configuration"]):
                        rng = np.random.default_rng(np.random.SeedSequence(
                            [20260930, ds_i, direction_i, multiplier, sampler_i, draw]))
                        observed = draw_counts(design, predictions, rng)
                        result = evaluate_counts(design, observed, tp, illicit)
                        points.append(result["point"])
                        lower.append(result["lower"])
                        upper.append(result["upper"])
                        fp_points.append(result["fp"])
                        mean_row = dict(config, draw=draw)
                        for metric_i, metric in enumerate(METRICS):
                            for quantity in ("point", "lower", "upper"):
                                mean_row[f"{metric}_{quantity}"] = float(
                                    result[quantity][:, metric_i].mean())
                        mean_draw_rows.append(mean_row)
                    points, lower, upper = map(np.asarray, (points, lower, upper))
                    stats = summarize_draws(points.mean(axis=1), lower.mean(axis=1),
                                            upper.mean(axis=1), truth.mean(axis=0))
                    summary_rows.extend(_metric_rows(config, stats))
                    per_seed_stats = summarize_draws(points, lower, upper, truth)
                    for seed_i, seed in enumerate(seeds):
                        per_seed_rows.extend(_metric_rows(dict(config, seed=seed), {
                            key: values[seed_i] for key, values in per_seed_stats.items()}))
                    # Raw per-seed estimates/endpoints retain shared draws in a compact array.
                    np.savez_compressed(
                        out / f"draws_{ds_i}_{direction_i}_{multiplier}_{sampler_i}.npz",
                        point=points, lower=lower, upper=upper, truth=truth,
                        fp=np.asarray(fp_points), full_fp=full_fp, seeds=np.asarray(seeds))
                    _write_csv(out / "summary.csv", summary_rows)
                    print(f"{dataset} {constructor}->{evaluator} B={multiplier}x {sampler}: "
                          f"F1 bias={stats['signed_bias'][2]:.6f} "
                          f"MAE={stats['mae'][2]:.6f} RMSE={stats['rmse'][2]:.6f} "
                          f"coverage={stats['coverage95'][2]:.3f} "
                          f"width={stats['mean_width95'][2]:.6f}", flush=True)
            del predictions, scores
    _write_csv(out / "per_seed_summary.csv", per_seed_rows)
    _write_csv(out / "full_population_metrics.csv", truth_rows)
    _write_csv(out / "designs.csv", design_rows)
    _write_csv(out / "transfer_diagnostics.csv", diagnostics)
    _write_csv(out / "seed_mean_draws.csv", mean_draw_rows)
    info = cached_count_interval.cache_info()
    _write_json(out / "run_metadata.json", {
        "finished_utc": datetime.now(timezone.utc).isoformat(),  # timezone.utc also supports the local Python 3.10 runner
        "elapsed_seconds": perf_counter() - started,
        "configurations": len(summary_rows) // len(METRICS),
        "total_draws": len(mean_draw_rows),
        "interval_cache": {"hits": info.hits, "misses": info.misses, "size": info.currsize},
        "protocol_sha256": manifest["protocol_sha256"],
        "scope": protocol["scope"],
    })
    print(f"Completed in {perf_counter() - started:.1f}s; outputs: {out}", flush=True)


if __name__ == "__main__":
    main()
