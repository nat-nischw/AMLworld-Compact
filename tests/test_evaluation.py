"""Offline checks for fresh prediction scoring and the LLM runner interface."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from amlc import hub
from amlc.analysis import llm_ht
from amlc.archive import legacy_path
from amlc.llm import clients, runner
from amlc.llm.methods import PredictionResult
from amlc.triage.doubt_triage import load_llm_predictions


def test_typology_scoring_requires_an_illicit_verdict():
    """A named pattern accompanying a benign verdict is a missed detection.

    The published postprocessing maps it to ``none``; that sentinel is also
    included in the macro label union, rather than dropping the missed case.
    """
    cases = [runner.EvalCase("amlc_00000", "e_0", 1, "fan-out"),
             runner.EvalCase("amlc_00001", "e_1", 1, "cycle")]
    predictions = [
        PredictionResult(case_id=case.case_id, method="ICL-FS", illicit=illicit,
                         typology=case.typology, evidence_edges=[],
                         confidence=0.8, rationale="")
        for case, illicit in zip(cases, [True, False])
    ]
    result = runner.compute_metrics(predictions, cases)
    assert result.detection_f1 == pytest.approx(2 / 3)
    assert result.typology_macro_f1 == pytest.approx(1 / 3)
    assert result.typology_accuracy == pytest.approx(1 / 2)
    assert result.typology_f1_per_class == {"cycle": 0.0, "fan-out": 1.0}


@pytest.fixture
def coreset(monkeypatch):
    data = {"n": 3, "labels": np.array([1, 0, 0]),
            "weights": np.array([1., 3., 6.])}

    def load(dataset, with_ensemble_probs=True):
        assert dataset == "HI-Small"
        assert with_ensemble_probs is False
        return data

    monkeypatch.setattr(hub, "load_coreset", load)
    monkeypatch.delenv("AMLC_ARCHIVE", raising=False)
    return data


def write_predictions(root, records=None, **metadata):
    path = legacy_path(root, "predictions", "HI-Small", model="GPT-OSS-20B",
                       prompting="ICL-ZS", seed=42)
    path.parent.mkdir(parents=True, exist_ok=True)
    if records is None:
        # Deliberately shuffled: JSON order must not decide alignment.
        records = [{"case_id": "amlc_00002", "illicit": False},
                   {"case_id": "amlc_00000", "illicit": True},
                   {"case_id": "amlc_00001", "illicit": True}]
    path.write_text(json.dumps({"predictions": records, **metadata}))
    return path


def test_fresh_cli_scores_complete_cohort_without_archive(tmp_path, coreset):
    write_predictions(tmp_path)
    out = tmp_path / "evaluation"
    df, summary = llm_ht.main([
        "--runs-dir", str(tmp_path), "--models", "GPT-OSS-20B",
        "--promptings", "ICL-ZS", "--seeds", "42", "--dataset", "HI-Small",
        "--out", str(out)])
    row = df[df.model == "GPT-OSS-20B"].iloc[0]
    assert row.n_flagged == 2
    assert row.subset_p == pytest.approx(50)
    assert row.subset_r == pytest.approx(100)
    assert row.subset_f1 == pytest.approx(100 * 2 / 3)
    assert row.ht_p == pytest.approx(25)
    assert row.ht_r == pytest.approx(100)
    assert row.ht_f1 == pytest.approx(40)
    assert summary.iloc[0].n_cells == 1
    assert summary.iloc[0].floor_ht_f1 == pytest.approx(100 * 2 / 11)
    assert len(pd.read_csv(out / "llm_ht_weighted.csv")) == 2
    assert len(pd.read_csv(out / "llm_ht_weighted_summary.csv")) == 1


@pytest.mark.parametrize("records,match", [
    ([{"case_id": "amlc_00000", "illicit": True}], "expected 3 unique cases"),
    ([{"case_id": "amlc_00000", "illicit": True},
      {"case_id": "v2_00000", "illicit": False}], "duplicate or out-of-range"),
    ([{"case_id": "amlc_00003", "illicit": True}], "duplicate or out-of-range"),
    ([{"case_id": "amlc_00000", "illicit": "false"}], "boolean or integer 0/1"),
    ([{"case_id": "amlc_00000", "illicit": 2}], "boolean or integer 0/1"),
    ([{"case_id": "transaction_1", "illicit": True}], "invalid case_id"),
])
def test_fresh_scoring_rejects_unscoreable_records(tmp_path, coreset, records, match):
    write_predictions(tmp_path, records)
    with pytest.raises(ValueError, match=match):
        llm_ht.score_dataset("HI-Small", runs_dir=tmp_path,
                             models=["GPT-OSS-20B"], promptings=["ICL-ZS"], seeds=[42])


@pytest.mark.parametrize("metadata,match", [
    ({"dataset": "LI-Small"}, "dataset='LI-Small'"),
    ({"method": "ICL-FS"}, "method='ICL-FS'"),
])
def test_fresh_scoring_rejects_mismatched_metadata(tmp_path, coreset, metadata, match):
    write_predictions(tmp_path, **metadata)
    with pytest.raises(ValueError, match=match):
        llm_ht.score_dataset("HI-Small", runs_dir=tmp_path,
                             models=["GPT-OSS-20B"], promptings=["ICL-ZS"], seeds=[42])


def test_fresh_scoring_requires_every_requested_seed(tmp_path, coreset):
    write_predictions(tmp_path)
    with pytest.raises(FileNotFoundError, match="seed_123.json"):
        llm_ht.score_dataset("HI-Small", runs_dir=tmp_path,
                             models=["GPT-OSS-20B"], promptings=["ICL-ZS"], seeds=[42, 123])


def test_legacy_archive_loader_keeps_missing_file_behavior(tmp_path, coreset, monkeypatch):
    archive = tmp_path / "outputs"
    monkeypatch.setenv("AMLC_ARCHIVE", str(archive))
    monkeypatch.setattr(llm_ht, "load_coreset_from_archive",
                        lambda archive, dataset, verify: coreset)
    rows = llm_ht.score_dataset("HI-Small", models=["GPT-OSS-20B"],
                                promptings=["ICL-ZS"], seeds=[42])
    assert [r["model"] for r in rows] == ["predict-all-illicit"]
    write_predictions(archive, [{"case_id": "v2_00000", "illicit": True}])
    preds, _, _ = load_llm_predictions(archive, "HI-Small", "GPT-OSS-20B",
                                       "ICL-ZS", 42, 3)
    np.testing.assert_array_equal(preds, [1, 0, 0])


@pytest.fixture
def fake_openai(monkeypatch):
    calls = []
    content = "- Conclusion: Suspicious\n- Observed Pattern: cycle"
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="mock-server-model",
        model_dump=lambda: {"choices": [{"message": {"content": content}}]},
    )

    def create(**kwargs):
        calls.append(kwargs)
        return response

    completions = SimpleNamespace(create=create)
    fake = SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        BadRequestError=type("BadRequestError", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "openai", fake)
    return calls, response, completions, fake


def test_runner_forwards_each_seed_and_writes_scoreable_output(
        tmp_path, monkeypatch, coreset, fake_openai):
    calls, _, _, _ = fake_openai
    cases = [runner.EvalCase(f"amlc_{i:05d}", str(i), int(y),
                             "cycle" if y else None)
             for i, y in enumerate(coreset["labels"])]
    monkeypatch.setattr(runner, "load_cases", lambda dataset, path:
                        (cases, {c.case_id: "mock typed graph" for c in cases}))
    cfg = runner.RunConfig(mode="llm", model="GPT-OSS-20B", model_id="served-name",
                           datasets=["HI-Small"], promptings=["ICL-ZS"],
                           seeds=[42, 123], workers=1, out=tmp_path)
    rows = runner.run_llm(cfg)
    assert "seed" not in calls[0]  # connectivity probe
    assert [c["seed"] for c in calls[1:]] == [42] * 3 + [123] * 3
    assert all(c["model"] == "served-name" for c in calls)
    assert rows[0]["n_seeds"] == 2
    scored = llm_ht.score_dataset("HI-Small", runs_dir=tmp_path,
                                  models=["GPT-OSS-20B"], promptings=["ICL-ZS"],
                                  seeds=[42, 123])
    assert [r["seed"] for r in scored] == [-1, 42, 123]
    assert all(r["n_flagged"] == 3 for r in scored)


def test_seed_and_vendor_sampling_survive_context_retry(fake_openai):
    calls, response, completions, fake = fake_openai

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise fake.BadRequestError("max_tokens must be at least 1, got -20")
        return response

    completions.create = create
    client = clients.get_client("Qwen3.5-35B-A3B", model_id="served-name", seed=456)
    client.call("mock prompt")
    assert len(calls) == 2
    for call in calls:
        assert call["seed"] == 456
        assert call["temperature"] == 0.6
        assert call["top_p"] == 0.95
        assert call["extra_body"] == {
            "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}
    assert calls[1]["max_tokens"] == clients.RETRY_MAX_TOKENS
    assert clients.MODELS["Qwen3.5-35B-A3B"].seed is None


def test_summary_preserves_other_models_and_replaces_only_rerun(tmp_path):
    cfg = runner.RunConfig(mode="llm", out=tmp_path)
    base = {"method": "ICL-ZS", "dataset": "HI-Small"}
    runner.save_summary(cfg, [{**base, "model": "model-A", "n_seeds": 1}])
    runner.save_summary(cfg, [{**base, "model": "model-B", "n_seeds": 2}])
    runner.save_summary(cfg, [{**base, "model": "model-A", "n_seeds": 5}])
    saved = pd.read_csv(tmp_path / "summary_all.csv").set_index("model")
    assert saved.n_seeds.to_dict() == {"model-A": 5, "model-B": 2}
