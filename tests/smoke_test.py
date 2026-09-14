#!/usr/bin/env python3
"""What a fresh clone must be able to do, with no archive and no AMLworld.

Run with `pytest tests/` or directly. Everything here works from the published
dataset plus what is committed in this repository. Anything needing the
original run directory is out of scope by design: those stages are for
regenerating results, and a clone should not silently appear to run them.

Set AMLC_CORESET_DIR to a local build to run offline; otherwise the
coreset is downloaded once and cached.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from amlc import config, paths
from amlc.case_ids import case_id, position, retag
from amlc.llm.prompts import PROMPTINGS, load_icl_examples, render
from amlc.typology import (
    TYPOLOGY_CLASSES, verify_against_strings,
)

OFFLINE = not (os.environ.get("AMLC_CORESET_DIR")
               or os.environ.get("AMLC_ALLOW_DOWNLOAD"))
needs_coreset = pytest.mark.skipif(
    OFFLINE, reason="set AMLC_CORESET_DIR, or AMLC_ALLOW_DOWNLOAD to fetch")


# ── the package itself ────────────────────────────────────────────────────

def test_every_module_imports_without_an_archive():
    import importlib
    import pkgutil
    import amlc
    os.environ.pop("AMLC_ARCHIVE", None)
    broken = []
    for m in pkgutil.walk_packages(amlc.__path__, "amlc."):
        try:
            importlib.import_module(m.name)
        except Exception as e:                      # pragma: no cover
            broken.append(f"{m.name}: {type(e).__name__}: {e}")
    assert not broken, "\n".join(broken)


def test_every_script_imports(capsys):
    """Each file under scripts/ must at least import against this package.

    The check above walks the package only, which is how
    ``scripts/build_hf_dataset.py`` came to ship importing ``DATASETS`` from
    ``amlc.archive``, where it has never been defined: ``make dataset`` died on
    the import line and nothing in the suite ran the file. Executing the module
    body catches that, and a typo in a constant, and a stage wired to a name the
    package does not export.

    A training script imports lightgbm or torch at module level, and those live
    in the ``[ml]`` and ``[gnn]`` extras rather than ``[dev]``, so on the install
    this suite runs under they are legitimately absent. A missing third-party
    module is therefore skipped, and the skipped files are printed so the
    coverage this test does not give is visible rather than implied. A missing
    *amlc* module is not skipped: that is the typo this test exists for.
    ``SystemExit`` is allowed because a script may exit on a missing argument.
    """
    import importlib.util

    broken, skipped = [], []
    for path in sorted((REPO / "scripts").rglob("*.py")):
        # A vLLM server plugin, imported by vLLM itself and not by us.
        if "host_vllm" in path.parts:
            continue
        rel = path.relative_to(REPO)
        spec = importlib.util.spec_from_file_location(
            "amlc_script_" + path.stem.replace(".", "_"), path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass
        except ModuleNotFoundError as exc:
            root = (exc.name or "").split(".")[0]
            if root and root != "amlc":
                skipped.append(f"{rel} (needs {root})")
            else:
                broken.append(f"{rel}: ModuleNotFoundError: {exc}")
        except Exception as exc:
            broken.append(f"{rel}: {type(exc).__name__}: {exc}")

    if skipped:
        with capsys.disabled():
            print("\n    not imported, optional dependency absent: "
                  + ", ".join(skipped))
    assert not broken, "\n".join(broken)


def test_archive_access_fails_loudly_rather_than_guessing():
    os.environ.pop("AMLC_ARCHIVE", None)
    with pytest.raises(FileNotFoundError):
        paths.archive()


# ── identifiers ───────────────────────────────────────────────────────────

def test_case_identifiers_round_trip():
    assert case_id(417) == "amlc_00417"
    assert position("amlc_00417") == 417 == position("v2_00417")


def test_retag_touches_only_whole_identifier_tokens():
    text = "case v2_01210 near acct_v2_9999 and v2_1 and e_v2_00001x"
    out, n = retag(text)
    assert n == 1
    assert "amlc_01210" in out
    for untouched in ("acct_v2_9999", "v2_1", "e_v2_00001x"):
        assert untouched in out


# ── prompts ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dataset", config.DATASETS)
@pytest.mark.parametrize("prompting", PROMPTINGS)
def test_every_prompt_template_renders(dataset, prompting):
    ex = load_icl_examples(dataset)
    out = render(prompting, "=== graph ===", ex)
    assert out.strip()
    assert "{{" not in out and "{%" not in out, "an unrendered Jinja tag survived"


def test_shipped_hyperparameters_cover_the_reported_baselines():
    for dataset in config.DATASETS:
        for method in ("LightGBM+GFP", "XGBoost+GFP"):
            for typ in (False, True):
                assert paths.tuned_params(dataset, method, typ).exists()


# ── the released coreset ──────────────────────────────────────────────────

@needs_coreset
@pytest.mark.parametrize("dataset", config.DATASETS)
def test_ht_weights_sum_to_the_full_split(dataset):
    from amlc.hub import load_coreset
    d = load_coreset(dataset)
    assert d["n"] == config.N_CORESET[dataset]
    assert np.isclose(d["weights"].sum(), config.N_TEST_FULL[dataset], rtol=0, atol=1e-6)


@needs_coreset
@pytest.mark.parametrize("dataset", config.DATASETS)
def test_typology_encoding_matches_the_ground_truth_strings(dataset):
    import pandas as pd
    from amlc.hub import coreset_dir, load_coreset
    d = load_coreset(dataset)
    index = pd.read_csv(coreset_dir(dataset) / "case_index.csv")
    report = verify_against_strings(d["typologies"], index["typology"].to_numpy())
    assert report["checked"] and report["agreement"] == 1.0
    assert set(index["typology"].dropna()) <= set(TYPOLOGY_CLASSES)


@needs_coreset
@pytest.mark.parametrize("dataset,expected_f1", [("HI-Small", 68.071), ("LI-Small", 28.629)])
def test_shipped_ensemble_reproduces_the_published_number(dataset, expected_f1):
    from amlc.hub import load_coreset
    from amlc.triage.doubt_triage import ht_weighted_prf
    d = load_coreset(dataset)
    if "ml_probs" not in d:
        pytest.skip("ensemble probabilities are not in this coreset copy")
    preds = (d["ml_probs"] >= d["ml_threshold"]).astype(int)
    _, _, f1 = ht_weighted_prf(preds, d["labels"], d["weights"])
    assert f1 * 100 == pytest.approx(expected_f1, abs=0.01)


@needs_coreset
@pytest.mark.parametrize("dataset", config.DATASETS)
def test_case_index_is_in_identifier_order(dataset):
    import pandas as pd
    from amlc.hub import coreset_dir
    index = pd.read_csv(coreset_dir(dataset) / "case_index.csv")
    assert list(index["case_id"]) == [case_id(i) for i in range(len(index))]
    assert "legacy_case_id" not in index.columns


@needs_coreset
@pytest.mark.parametrize("dataset", config.DATASETS)
def test_case_index_columns_agree_with_the_arrays_beside_them(dataset):
    """The case index duplicates three shipped arrays; all three must match.

    The first release shipped an LI-Small ``case_index.csv`` whose ``weight``
    column summed to 2,767,353 against a population of 1,384,810, while the
    ``ht_weights.npy`` beside it was correct: the builder copied the column
    from the archive, where it predated the spare-fill fix, and only checked
    the repaired vector. Duplicated data needs the duplicate checked.
    """
    import pandas as pd
    from amlc.hub import coreset_dir
    d = coreset_dir(dataset)
    index = pd.read_csv(d / "case_index.csv")
    for column, filename in [("subset_index", "ht_subset_indices.npy"),
                             ("label", "labels.npy"),
                             ("weight", "ht_weights.npy")]:
        if column not in index.columns:
            continue
        array = np.load(d / filename)
        assert len(index) == len(array), f"{column}: {len(index)} rows vs {len(array)}"
        assert np.array_equal(index[column].to_numpy(), array), \
            f"case_index.{column} disagrees with {filename}"
    assert np.isclose(index["weight"].sum(), config.N_TEST_FULL[dataset],
                      rtol=0, atol=1e-6)


# ── the frontier-API probe ────────────────────────────────────────────────

def test_frontier_probe_reproduces_every_published_number():
    from amlc.audit import frontier_probe
    if not frontier_probe.predictions_path().exists():
        pytest.skip("predictions.csv is not in this checkout")
    frontier_probe.verify()


def test_frontier_probe_predictions_carry_no_generated_text():
    """The probe responses hold proprietary chain-of-thought; only the parsed
    fields may ship. Guard the shape rather than trusting the extractor."""
    import csv
    from amlc.audit import frontier_probe
    path = frontier_probe.predictions_path()
    if not path.exists():
        pytest.skip("predictions.csv is not in this checkout")
    with path.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == frontier_probe.FIELDS
        longest = 0
        for row in reader:
            assert not row["case_id"].startswith("v2_")
            longest = max(longest, max(len(v) for v in row.values()))
    assert longest < 40, f"a field is {longest} characters, too long to be a parsed value"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
