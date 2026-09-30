"""Drive one evaluation run: models, promptings, datasets, seeds.

Two modes, both of which the paper reports.

``llm``
    One model, served by vLLM or reached over an API, against the HT-Coreset
    under each in-context condition and each seed. The coreset is fixed across
    seeds; what varies is the model's own sampling, which is where the seed
    enters. The requested seed is sent to vLLM on each case request.

``supervised``
    LightGBM+GFP and XGBoost+GFP, trained on the file-order training partition,
    early-stopped on validation and scored individually at validation-selected
    thresholds. Stage 06 combines their saved probabilities and scores the
    reported two-booster ensemble at inherited, full-test-selected thresholds.
    GCPAL is used only in the frozen construction scorer, outside this runner.

Writes, under ``--out`` (default ``paths.results()``)
    ``<model>/<condition>/<dataset>/seed_<n>.json``   per-case predictions, raw
        answers, reasoning traces and token accounting
    ``<model>/<condition>/<dataset>/metrics.json``    aggregated over seeds
    ``test_probs/<dataset>/``                         supervised probabilities,
        typology predictions, test labels and test typologies
    ``summary_all.csv`` and ``.json``                 one row per configuration,
        merged with what is already there so a partial rerun does not erase the
        rest

The LLM half of that tree has the same shape as the archived run directory, and
it is built with :func:`amlc.archive.legacy_path` for that reason:
the HT scorer can read it with ``--runs-dir`` and load labels and weights from
the public coreset. Directory names inside it are therefore the archive's;
everything a user types, and everything a log line says, is the paper's.

The supervised half always scores the full file-order test partition and the LLM half
always scores the coreset, which is what every table reports. The released
checkpoints are published with the dataset; a rerun that wants them can save
from the returned model.

Scoring lives here
------------------

``compute_metrics`` and ``aggregate_metrics_across_seeds`` score
:class:`~amlc.llm.methods.PredictionResult` objects against cases. Both
dataclasses belong to this path, so the functions are here rather than in
:mod:`amlc.baselines.metrics`, which owns the threshold sweep and the
HT-weighted estimator that the supervised stages share.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .. import config, paths
from ..archive import legacy_path, member_dir
from ..typology import TYPOLOGY_CLASSES, TYPOLOGY_TO_IDX
from .clients import MODELS, get_client
from .methods import PredictionResult, get_method
from .prompts import load_icl_examples

#: The two boosted-tree evaluation members. GCPAL belongs to the separate
#: frozen construction scorer and is not trained by this runner.
GCPAL_MEMBER = "GCPAL+GFP"
TRAINABLE_MEMBERS = tuple(m for m in config.ENSEMBLE_MEMBERS if m != GCPAL_MEMBER)

#: Where supervised per-seed predictions land. The archive called this
#: directory ``non-llm``; nothing reads it back, so a fresh run uses the paper's
#: word for the same thing.
SUPERVISED_DIR = "supervised"

#: How often the per-case loop reports progress. A cluster log is read after the
#: fact, so this is a heartbeat rather than a progress bar.
PROGRESS_EVERY = 250


# ─────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """One evaluation row: the ground truth a prediction is scored against."""

    case_id: str
    center_edge_id: str
    label: int
    typology: Optional[str] = None


@dataclass
class RunConfig:
    mode: str                                   # llm | supervised | all
    datasets: Sequence[str] = config.DATASETS
    seeds: Sequence[int] = config.SEEDS
    workers: int = 16
    out: Path = field(default_factory=paths.results)
    # LLM half
    model: Optional[str] = None
    model_id: Optional[str] = None
    vllm_url: Optional[str] = None
    promptings: Sequence[str] = config.PROMPTINGS
    cases_jsonl: Optional[Path] = None          # else the released table
    # Supervised half
    members: Sequence[str] = TRAINABLE_MEMBERS
    data_path: Optional[Path] = None            # AMLworld CSVs
    tuned: bool = True


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{int(s)}s"


# ─────────────────────────────────────────────────────────────────────────
#  Scoring
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MetricsResult:
    """The metric set every stage of the paper reports, per seed.

    The four faithfulness fields and the verifier rate are carried at zero.
    They are not measurements on this path: no in-context condition asks the
    model to cite evidence edges, so ``evidence_edges`` is always empty; the
    pre-release typology verifier is not part of this release; and a case read
    back from the serialised coreset holds no transaction list to check a
    rationale against. They stay in the record because the published result
    files and ``results/summary_all.csv`` carry the columns.
    """

    detection_f1: float
    detection_precision: float
    detection_recall: float
    detection_accuracy: float
    detection_pr_auc: float

    typology_macro_f1: float
    typology_accuracy: float
    verifier_pass_rate: float

    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    unsupported_claim_rate: float

    avg_llm_calls: float
    avg_tokens: float
    avg_latency_ms: float

    typology_f1_per_class: Optional[dict] = None


#: Fields aggregated across seeds, in report order.
AGGREGATED_FIELDS = (
    "detection_f1", "detection_precision", "detection_recall",
    "detection_accuracy", "detection_pr_auc",
    "typology_macro_f1", "typology_accuracy", "verifier_pass_rate",
    "evidence_precision", "evidence_recall", "evidence_f1",
    "unsupported_claim_rate",
    "avg_llm_calls", "avg_tokens", "avg_latency_ms",
)


def compute_metrics(predictions: Sequence[PredictionResult],
                    cases: Sequence[EvalCase]) -> MetricsResult:
    """Score one seed's predictions against the ground truth.

    Detection is scored on every case. Typology is scored only where the case
    is illicit and carries a pattern. A benign verdict or a prediction with no
    typology counts as the sentinel class rather than being skipped, matching
    the postprocessed typology results reported in the paper.
    """
    from sklearn.metrics import (accuracy_score, auc, f1_score,
                                 precision_recall_curve, precision_score,
                                 recall_score)

    if len(predictions) != len(cases):
        raise ValueError(f"{len(predictions)} predictions against "
                         f"{len(cases)} cases")
    by_id = {c.case_id: c for c in cases}

    y_true, y_pred, y_score = [], [], []
    typ_true, typ_pred = [], []
    llm_calls, tokens, latencies = [], [], []

    for pred in predictions:
        case = by_id.get(pred.case_id)
        if case is None:
            continue
        y_true.append(case.label)
        y_pred.append(1 if pred.illicit else 0)
        # The conditions elicit no probability, so the ranking score is the
        # verdict's own coarse confidence, oriented towards illicit.
        y_score.append(pred.confidence if pred.illicit else 1 - pred.confidence)

        if case.label == 1 and case.typology:
            typ_true.append(case.typology)
            typ_pred.append((pred.typology or "none") if pred.illicit else "none")

        llm_calls.append(pred.llm_calls)
        tokens.append(pred.total_tokens)
        latencies.append(pred.latency_ms)

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(recall, precision))
    except ValueError:
        # One class present, or a degenerate score vector.
        pr_auc = 0.0

    macro_f1, typ_accuracy, per_class = _typology_metrics(typ_true, typ_pred)

    return MetricsResult(
        detection_f1=float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        detection_precision=float(precision_score(y_true, y_pred, pos_label=1,
                                                  zero_division=0)),
        detection_recall=float(recall_score(y_true, y_pred, pos_label=1,
                                            zero_division=0)),
        detection_accuracy=float(accuracy_score(y_true, y_pred)),
        detection_pr_auc=pr_auc,
        typology_macro_f1=macro_f1,
        typology_accuracy=typ_accuracy,
        verifier_pass_rate=0.0,
        evidence_precision=0.0,
        evidence_recall=0.0,
        evidence_f1=0.0,
        unsupported_claim_rate=0.0,
        avg_llm_calls=float(np.mean(llm_calls)) if llm_calls else 0.0,
        avg_tokens=float(np.mean(tokens)) if tokens else 0.0,
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        typology_f1_per_class=per_class,
    )


def _typology_metrics(y_true: Sequence[str], y_pred: Sequence[str]
                      ) -> tuple[float, float, dict]:
    """Macro F1, accuracy and per-class F1 over typology names.

    The label space is whatever appears on either side, which includes the
    "none" sentinel for a verdict that named no pattern. "none" is excluded from
    the per-class breakdown but not from the macro average: a model that
    declines to name a pattern is wrong, not absent.
    """
    from sklearn.metrics import accuracy_score, f1_score

    if not y_true:
        return 0.0, 0.0, {}

    labels = sorted(set(y_true) | set(y_pred))
    index = {name: i for i, name in enumerate(labels)}
    true_enc = [index[t] for t in y_true]
    pred_enc = [index[t] for t in y_pred]

    per_class = {}
    for name in labels:
        if name == "none":
            continue
        per_class[name] = float(f1_score([1 if t == name else 0 for t in y_true],
                                         [1 if p == name else 0 for p in y_pred],
                                         zero_division=0))
    return (float(f1_score(true_enc, pred_enc, average="macro", zero_division=0)),
            float(accuracy_score(true_enc, pred_enc)),
            per_class)


def aggregate_metrics_across_seeds(seed_metrics: Sequence[MetricsResult]) -> dict:
    """Mean and standard deviation per field, over seeds."""
    if not seed_metrics:
        return {}
    return {name: (float(np.mean([getattr(m, name) for m in seed_metrics])),
                   float(np.std([getattr(m, name) for m in seed_metrics])))
            for name in AGGREGATED_FIELDS}


# ─────────────────────────────────────────────────────────────────────────
#  Cases
# ─────────────────────────────────────────────────────────────────────────

def load_cases(dataset: str, cases_jsonl: Optional[Path] = None
               ) -> tuple[list[EvalCase], dict[str, str]]:
    """The coreset rows and the prompt text for each.

    By default from the released evaluation table, which carries the typed
    graph the models were actually shown. ``cases_jsonl`` reads instead the file
    :mod:`amlc.serialize.build_coreset_prompts` writes, which is what a
    regeneration produces before anything is published.
    """
    if cases_jsonl:
        rows = [json.loads(line)
                for line in Path(cases_jsonl).open(encoding="utf-8")]
        cases = [EvalCase(r["case_id"], r["center_edge_id"], int(r["label"]),
                          r.get("typology") or None) for r in rows]
        texts = {r["case_id"]: r["serialized_text"] for r in rows}
    else:
        from .. import hub
        df = hub.load_table(dataset)
        cases = [EvalCase(r.case_id, r.center_edge_id, int(r.label),
                          r.typology if isinstance(r.typology, str) else None)
                 for r in df.itertuples()]
        texts = dict(zip(df["case_id"], df["typed_graph_text"]))

    n_illicit = sum(c.label for c in cases)
    print(f"  [{dataset}] {len(cases):,} cases "
          f"({n_illicit:,} illicit, {len(cases) - n_illicit:,} benign)")
    return cases, texts


# ─────────────────────────────────────────────────────────────────────────
#  LLM
# ─────────────────────────────────────────────────────────────────────────

def run_llm(cfg: RunConfig) -> list[dict]:
    """One model over every prompting condition, dataset and seed."""
    print(f"\n  Connecting to {cfg.model} at "
          f"{cfg.vllm_url or 'the endpoint in the environment'}")
    client = get_client(cfg.model, base_url=cfg.vllm_url, model_id=cfg.model_id)
    probe = client.call("Say 'OK'.")
    print(f"  connected ({probe.content[:20].strip()!r})")

    rows = []
    for dataset in cfg.datasets:
        cases, texts = load_cases(dataset, cfg.cases_jsonl)

        icl_examples = None
        if any(p != "ICL-ZS" for p in cfg.promptings):
            icl_examples = load_icl_examples(dataset)
            print(f"  [{dataset}] demonstration pool: "
                  f"{icl_examples['n_suspicious']} suspicious + "
                  f"{icl_examples['n_non_suspicious']} non-suspicious, "
                  f"~{icl_examples['total_tokens_approx']:,} tokens per prompt")

        for prompting in cfg.promptings:
            seed_metrics, seed_tokens = [], []

            for seed in cfg.seeds:
                seeded_client = get_client(
                    cfg.model, base_url=cfg.vllm_url, model_id=cfg.model_id, seed=seed)
                method = get_method(
                    prompting, seeded_client, texts, icl_examples=icl_examples)
                t0 = time.time()
                predictions, n_errors = predict_all(method, cases, cfg.workers)
                elapsed = time.time() - t0

                metrics = compute_metrics(predictions, cases)
                seed_metrics.append(metrics)
                _, token_stats = save_seed_predictions(
                    cfg, cfg.model, prompting, dataset, seed, predictions, metrics)
                seed_tokens.append(token_stats)

                print(f"    {prompting} {dataset} seed={seed}: "
                      f"F1={metrics.detection_f1 * 100:.1f}% "
                      f"Typ-F1={metrics.typology_macro_f1 * 100:.1f}% "
                      f"valid={token_stats['valid_ratio']:.1f}% "
                      f"{fmt_time(elapsed)}"
                      + (f" {n_errors} errors" if n_errors else ""))

            tokens = {
                "avg_input_tokens": round(_mean(seed_tokens, "avg_input_tokens"), 2),
                "avg_output_tokens": round(_mean(seed_tokens, "avg_output_tokens"), 2),
                "avg_valid_ratio": round(_mean(seed_tokens, "valid_ratio"), 2),
                "total_valid": sum(t["n_valid"] for t in seed_tokens),
                "total_failed": sum(t["n_failed"] for t in seed_tokens),
            }
            _, agg = save_method_metrics(cfg, cfg.model, prompting, dataset,
                                         seed_metrics, token_stats=tokens)
            rows.append({
                "method": prompting,
                "model": cfg.model,
                "dataset": dataset,
                "n_seeds": len(cfg.seeds),
                **_percent_columns(agg),
                "avg_input_tokens": tokens["avg_input_tokens"],
                "avg_output_tokens": tokens["avg_output_tokens"],
                "valid_ratio": tokens["avg_valid_ratio"],
            })

            d, t = agg["detection_f1"], agg["typology_macro_f1"]
            print(f"  {prompting:8s} | {dataset:9s} | "
                  f"Det-F1={d[0] * 100:.1f}+-{d[1] * 100:.1f}% | "
                  f"Typ-F1={t[0] * 100:.1f}+-{t[1] * 100:.1f}% | "
                  f"valid={tokens['avg_valid_ratio']:.1f}%")
    return rows


def predict_all(method, cases: Sequence[EvalCase], workers: int
                ) -> tuple[list[PredictionResult], int]:
    """One prompt per case, in parallel, returned in case order.

    A case whose call raises still gets a record. Dropping it would quietly
    shorten the evaluation set; a benign verdict carrying the error in its
    rationale is scoreable, is counted as invalid by the token accounting, and
    is visible in the saved file.
    """
    def predict_one(case: EvalCase):
        try:
            return method.predict(case), None
        except Exception as e:  # broad on purpose: recorded below, not swallowed
            return PredictionResult(
                case_id=case.case_id,
                method=method.name,
                illicit=False,
                typology=None,
                evidence_edges=[],
                confidence=0.5,
                rationale=f"Error: {e}",
                center_edge_id=case.center_edge_id,
            ), str(e)

    predictions, n_errors, total_tokens = [], 0, 0
    n_workers = max(1, min(workers, len(cases)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(predict_one, c) for c in cases]
        for i, future in enumerate(as_completed(futures), start=1):
            pred, err = future.result()
            predictions.append(pred)
            total_tokens += pred.total_tokens
            if err:
                n_errors += 1
            if i % PROGRESS_EVERY == 0:
                print(f"      {i:,}/{len(cases):,} cases, {total_tokens:,} tokens, "
                      f"{n_errors} errors", flush=True)

    order = {c.case_id: i for i, c in enumerate(cases)}
    predictions.sort(key=lambda p: order.get(p.case_id, 0))
    return predictions, n_errors


# ─────────────────────────────────────────────────────────────────────────
#  Supervised
# ─────────────────────────────────────────────────────────────────────────

def load_tuned(dataset: str, member: str, typology: bool = False) -> dict:
    """One member's Optuna result, from the shipped ``data/tuned_params``."""
    path = paths.tuned_params(dataset, member, typology=typology)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def run_supervised(cfg: RunConfig) -> list[dict]:
    """Train and score the boosted-tree ensemble members."""
    # Imported here, not at module scope: scoring an LLM must not require
    # SnapML, LightGBM or XGBoost.
    from ..baselines.metrics import find_optimal_threshold
    from ..baselines.ml.gbt import get_supervised_baseline
    from ..data.gfp import load_amlworld_for_snapml

    rows = []
    for dataset in cfg.datasets:
        print(f"\n  [{dataset}] extracting Graph-Feature-Preprocessor features")
        t0 = time.time()
        data = load_amlworld_for_snapml(
            variant=dataset, data_path=cfg.data_path,
            num_threads=max(cfg.workers, 4))
        print(f"  features ready in {fmt_time(time.time() - t0)}")

        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        edge_typologies = data["edge_typologies"]
        t2_idx = data["t2_idx"]

        # Written once per dataset rather than once per member: the ensemble,
        # the coreset construction and the Doubt Triage stage all read them.
        labels_path = legacy_path(cfg.out, "test_labels", dataset)
        paths.ensure(labels_path.parent)
        np.save(labels_path, y_test)

        test_typologies = [edge_typologies[t2_idx + i] for i in range(len(y_test))]
        test_typ_ids = np.full(len(y_test), -1, dtype=int)
        for i, typ in enumerate(test_typologies):
            if y_test[i] == 1 and typ in TYPOLOGY_TO_IDX:
                test_typ_ids[i] = TYPOLOGY_TO_IDX[typ]
        np.save(legacy_path(cfg.out, "test_typologies", dataset), test_typ_ids)
        print(f"  wrote test labels and {int((test_typ_ids >= 0).sum()):,} "
              "typed test typologies")

        # The typology head is fitted on train and validation together: typed
        # illicit edges are scarce and it is not what early stopping watches.
        y_train_val = np.concatenate([y_train, y_val])
        combined_typ = np.full(len(y_train_val), -1, dtype=int)
        for i in range(len(y_train_val)):
            typ = edge_typologies[i]
            if y_train_val[i] == 1 and typ in TYPOLOGY_TO_IDX:
                combined_typ[i] = TYPOLOGY_TO_IDX[typ]

        # LightGBM and XGBoost release the GIL inside their C++ training loops,
        # so seeds run as threads over one shared copy of the feature matrices.
        total_threads = max(cfg.workers, os.cpu_count() or cfg.workers)
        parallel_seeds = min(len(cfg.seeds), max(1, total_threads // 6))
        threads_per_seed = max(4, total_threads // max(parallel_seeds, 1))
        print(f"  {parallel_seeds} seeds in parallel, {threads_per_seed} threads "
              f"each, {total_threads} CPUs")

        for member in cfg.members:
            tuned = load_tuned(dataset, member) if cfg.tuned else {}
            params = dict(tuned.get("best_params", {}))
            typ_params = (load_tuned(dataset, member,
                                     typology=True).get("best_params")
                          if cfg.tuned else None)

            # Optuna searched n_estimators with early stopping, so the round
            # count that actually trained is the one to reuse, not the ceiling
            # the search was allowed.
            n_rounds = tuned.get("actual_n_rounds")
            if n_rounds is not None and "n_estimators" in params:
                print(f"  [{member}] n_estimators {params['n_estimators']} -> "
                      f"{n_rounds} (early stopping during tuning)")
                params["n_estimators"] = n_rounds

            def train_seed(seed: int, member=member, params=params,
                           typ_params=typ_params) -> dict:
                t_seed = time.time()
                model = get_supervised_baseline(member, **params)
                model.fit_from_features(
                    X_train, y_train, X_val, y_val,
                    typology_labels=combined_typ,
                    typology_params=typ_params,
                    num_threads=threads_per_seed,
                    seed=seed,
                )
                train_time = time.time() - t_seed

                # The operating point is chosen on validation and applied to
                # test. Choosing it on test is the diagnostic below, not this.
                val_probs = model.predict_proba_from_features(X_val)
                threshold, val_f1 = find_optimal_threshold(y_val, val_probs)

                test_probs = model.predict_proba_from_features(X_test)
                probs_path = legacy_path(cfg.out, "member_probs", dataset,
                                         member=member_dir(member), seed=seed)
                paths.ensure(probs_path.parent)
                np.save(probs_path, test_probs)

                test_typ = model.predict_typology_from_features(X_test)
                if test_typ is not None:
                    np.save(legacy_path(cfg.out, "member_typology", dataset,
                                        member=member_dir(member), seed=seed),
                            test_typ)

                case_ids = [f"e_{t2_idx + j}" for j in range(len(y_test))]
                predictions = supervised_predictions(
                    member, case_ids, test_probs, test_typ, threshold)
                cases = [EvalCase(cid, cid, int(y_test[i]), test_typologies[i])
                         for i, cid in enumerate(case_ids)]
                metrics = compute_metrics(predictions, cases)

                # Diagnostic only, and it is threshold selection on the test
                # split: it separates a ranking failure from a thresholding one.
                # It does not set this runner's per-member predictions.
                # Stage 06 separately applies the reported ensemble thresholds.
                oracle_threshold, oracle_f1 = find_optimal_threshold(y_test,
                                                                     test_probs)
                print(f"    {member} {dataset} seed={seed}: "
                      f"threshold={threshold:.3f} "
                      f"F1={metrics.detection_f1 * 100:.1f}% "
                      f"(val F1={val_f1 * 100:.1f}%, oracle {oracle_threshold:.3f} "
                      f"/ {oracle_f1 * 100:.1f}%) {fmt_time(train_time)}")
                return {"seed": seed, "metrics": metrics,
                        "predictions": predictions, "threshold": threshold,
                        "val_f1": val_f1, "train_time": train_time}

            if parallel_seeds > 1:
                with ThreadPoolExecutor(max_workers=parallel_seeds) as pool:
                    futures = [pool.submit(train_seed, s) for s in cfg.seeds]
                    results = [f.result() for f in as_completed(futures)]
            else:
                results = [train_seed(s) for s in cfg.seeds]
            results.sort(key=lambda r: list(cfg.seeds).index(r["seed"]))

            for r in results:
                save_seed_predictions(cfg, member, member, dataset, r["seed"],
                                      r["predictions"], r["metrics"])
            _, agg = save_method_metrics(cfg, member, member, dataset,
                                         [r["metrics"] for r in results])
            rows.append({
                "method": member,
                "model": member,
                "dataset": dataset,
                "n_seeds": len(results),
                **_percent_columns(agg),
            })
            d = agg["detection_f1"]
            mean_train = sum(r["train_time"] for r in results) / len(results)
            print(f"  {member:14s} | {dataset:9s} | "
                  f"F1={d[0] * 100:.1f}+-{d[1] * 100:.1f}% | "
                  f"{mean_train:.1f}s/seed")
    return rows


def supervised_predictions(member: str, case_ids: Sequence[str],
                           probs: np.ndarray, typology_ids: Optional[np.ndarray],
                           threshold: float) -> list[PredictionResult]:
    """Wrap a member's probability vector as predictions, for scoring.

    The typology head is consulted only where the detector fires, which is the
    two-stage design: a typology on an edge the model calls benign is not a
    prediction anyone reads. The pre-release wrapper lived on the baseline class
    and called the head one row at a time; the head's own bulk prediction is the
    same argmax over a million rows fewer calls.
    """
    predictions = []
    for i, (cid, prob) in enumerate(zip(case_ids, probs)):
        illicit = bool(prob >= threshold)
        typology = None
        if illicit and typology_ids is not None:
            typology = TYPOLOGY_CLASSES[int(typology_ids[i])]
        predictions.append(PredictionResult(
            case_id=cid,
            method=member,
            illicit=illicit,
            typology=typology,
            evidence_edges=[],
            confidence=float(prob) if illicit else float(1 - prob),
            rationale=f"Predicted by {member} (p={prob:.3f}, typology={typology})",
            llm_calls=0,
            total_tokens=0,
            latency_ms=0,
        ))
    return predictions


# ─────────────────────────────────────────────────────────────────────────
#  Saving
# ─────────────────────────────────────────────────────────────────────────

def _mean(dicts: Sequence[dict], key: str) -> float:
    return sum(d[key] for d in dicts) / len(dicts) if dicts else 0.0


def _percent_columns(agg: dict) -> dict:
    """The rate-like aggregates, formatted as mean+-std in percent."""
    keys = ("f1", "precision", "recall", "accuracy", "rate")
    return {k: f"{v[0] * 100:.1f}+-{v[1] * 100:.1f}"
            for k, v in agg.items() if any(x in k for x in keys)}


def prediction_dir(cfg: RunConfig, model: str, method: str, dataset: str) -> Path:
    """Where one configuration's per-seed files go.

    An LLM run lands where the archived tree put it, through ``legacy_path``, so
    a fresh run can be read by the downstream stages unchanged. A supervised
    member has no prompting condition and lands under its own name.
    """
    if method in config.PROMPTINGS:
        return legacy_path(cfg.out, "predictions", dataset, model=model,
                           prompting=method, seed=0).parent
    return cfg.out / SUPERVISED_DIR / method / dataset


def save_seed_predictions(cfg: RunConfig, model: str, method: str, dataset: str,
                          seed: int, predictions: Sequence[PredictionResult],
                          metrics: Optional[MetricsResult] = None
                          ) -> tuple[Path, dict]:
    """Write one seed's predictions, with the token accounting beside them."""
    directory = paths.ensure(prediction_dir(cfg, model, method, dataset))

    n = len(predictions)
    # A prediction that spent no tokens never reached the model, so the token
    # count is also the denominator behind the valid_ratio column.
    n_valid = sum(1 for p in predictions if p.total_tokens > 0)
    total_input = sum(p.input_tokens for p in predictions)
    total_output = sum(p.output_tokens for p in predictions)
    total = sum(p.total_tokens for p in predictions)
    token_stats = {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total,
        "avg_input_tokens": round(total_input / n, 2) if n else 0,
        "avg_output_tokens": round(total_output / n, 2) if n else 0,
        "avg_total_tokens": round(total / n, 2) if n else 0,
        "n_valid": n_valid,
        "n_failed": n - n_valid,
        "valid_ratio": round(n_valid / n * 100, 2) if n else 0,
    }

    payload = {
        "method": method,
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "n_predictions": n,
        "token_stats": token_stats,
        "metrics": asdict(metrics) if metrics is not None else None,
        "predictions": [asdict(p) for p in predictions],
    }
    path = directory / f"seed_{seed}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path, token_stats


def save_method_metrics(cfg: RunConfig, model: str, method: str, dataset: str,
                        seed_metrics: Sequence[MetricsResult],
                        token_stats: Optional[dict] = None) -> tuple[Path, dict]:
    """Aggregate over seeds and write ``metrics.json`` beside the predictions."""
    agg = aggregate_metrics_across_seeds(list(seed_metrics))
    directory = paths.ensure(prediction_dir(cfg, model, method, dataset))

    payload = {
        "method": method,
        "model": model,
        "dataset": dataset,
        "n_seeds": len(seed_metrics),
        "aggregated": {k: {"mean": round(v[0], 4), "std": round(v[1], 4)}
                       for k, v in agg.items()},
    }
    if token_stats:
        payload["token_stats"] = token_stats

    path = directory / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path, agg


def save_summary(cfg: RunConfig, rows: Sequence[dict]) -> Path:
    """Merge this run's rows into the summary table.

    A (model, method, dataset) configuration this run produced replaces its old
    row; every other row is left alone.
    """
    import pandas as pd

    paths.ensure(cfg.out)
    csv_path = cfg.out / "summary_all.csv"
    json_path = cfg.out / "summary_all.json"

    df_new = pd.DataFrame(list(rows))
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        replaced = set(zip(df_new["model"], df_new["method"], df_new["dataset"]))
        keep = df_old[~df_old.apply(
            lambda r: (r["model"], r["method"], r["dataset"]) in replaced, axis=1)]
        df = pd.concat([keep, df_new], ignore_index=True)
    else:
        df = df_new
    df = df.sort_values(["model", "method", "dataset"]).reset_index(drop=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n{df.to_string(index=False)}\n\n  wrote {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Run one AMLworld-Compact evaluation.",
        epilog="examples:\n"
               "  python -m amlc.llm.runner --mode llm "
               "--model GPT-OSS-120B --vllm-url http://node:18809/v1\n"
               "  python -m amlc.llm.runner --mode supervised "
               "--data-path /path/to/amlworld",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["llm", "supervised", "all"])
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS),
                    choices=config.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None,
                    help="output tree; defaults to results/")
    ap.add_argument("--model", default=None, choices=list(MODELS),
                    help="the model to evaluate, for --mode llm")
    ap.add_argument("--model-id", default=None,
                    help="override --served-model-name on the vLLM server")
    ap.add_argument("--vllm-url", default=None,
                    help="vLLM endpoint; defaults to $VLLM_BASE_URL")
    ap.add_argument("--promptings", nargs="+", default=list(config.PROMPTINGS),
                    choices=config.PROMPTINGS)
    ap.add_argument("--cases-jsonl", type=Path, default=None,
                    help="prompts from a local cases_typed_graph.jsonl instead "
                         "of the released evaluation table")
    ap.add_argument("--members", nargs="+", default=list(TRAINABLE_MEMBERS),
                    choices=TRAINABLE_MEMBERS,
                    help="supervised members to train")
    ap.add_argument("--data-path", type=Path, default=None,
                    help="AMLworld CSVs, for --mode supervised")
    ap.add_argument("--no-tuned-params", action="store_true",
                    help="ignore data/tuned_params and use library defaults")
    args = ap.parse_args(argv)

    if args.mode in ("llm", "all") and not args.model:
        ap.error("--mode llm needs --model")

    cfg = RunConfig(
        mode=args.mode,
        datasets=args.datasets,
        seeds=args.seeds,
        workers=args.workers,
        out=args.out or paths.results(),
        model=args.model,
        model_id=args.model_id,
        vllm_url=args.vllm_url,
        promptings=args.promptings,
        cases_jsonl=args.cases_jsonl,
        members=args.members,
        data_path=args.data_path,
        tuned=not args.no_tuned_params,
    )

    print("AMLworld-Compact evaluation")
    print(f"  mode      {cfg.mode}")
    print(f"  datasets  {list(cfg.datasets)}")
    print(f"  seeds     {list(cfg.seeds)}")
    print(f"  out       {cfg.out}")
    if cfg.mode in ("llm", "all"):
        print(f"  model     {cfg.model}")
        print(f"  prompting {list(cfg.promptings)}")
        print(f"  cases     {cfg.cases_jsonl or 'released evaluation table'}")
    if cfg.mode in ("supervised", "all"):
        print(f"  members   {list(cfg.members)}")
        print(f"  tuned     {'data/tuned_params' if cfg.tuned else 'library defaults'}")

    t0 = time.time()
    rows = []
    if cfg.mode in ("supervised", "all"):
        rows.extend(run_supervised(cfg))
    if cfg.mode in ("llm", "all"):
        rows.extend(run_llm(cfg))
    if rows:
        save_summary(cfg, rows)
    print(f"\n  {len(rows)} configurations in {fmt_time(time.time() - t0)}")


if __name__ == "__main__":
    sys.exit(main())
