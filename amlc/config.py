"""Experiment constants, in the paper's vocabulary.

The values are not here. They are in ``config.yaml`` beside this file, which is
the single place to look or to edit; this module loads it, applies the
environment overrides and re-exports every entry under the name the rest of the
package already imports.

Splitting the data out of the code buys three things. A reader checking the
paper against the artefact reads one annotated file instead of tracing
constants through seven modules. Values that used to be scattered as literals,
the per-model sampling parameters in ``llm/clients.py`` and the judge model ids
in ``audit/judges.py`` among them, now sit next to the split sizes they belong
with. And the addresses, the dataset repo and the vLLM endpoint, can be pointed
somewhere else without editing a file that ships inside the package.

Construction provenance and the primary evaluation ensemble have separate
constants so changing an evaluator cannot alter the released sampling design.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(
    os.environ.get("AMLC_CONFIG", Path(__file__).with_name("config.yaml"))
)


def load(path: Path | str | None = None) -> dict[str, Any]:
    """Read the YAML. Separate from module import so a test can load a fixture."""
    p = Path(path) if path is not None else CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing. It ships inside the package; if you are running "
            "from a source tree, check that config.yaml was not lost, and if "
            "from an install, that package data was included."
        )
    with open(p) as f:
        return yaml.safe_load(f)


_C = load()


def _env(name: str, default):
    """Environment override, documented next to the key in config.yaml."""
    v = os.environ.get(name)
    return v if v else default


# ── sources ──────────────────────────────────────────────────────────────
_S = _C["sources"]

#: The published evaluation set on the Hugging Face Hub.
DATASET_REPO: str = _env("AMLC_DATASET", _S["dataset_repo"])

#: A local coreset build to read instead of downloading, or None.
CORESET_DIR: str | None = _env("AMLC_CORESET_DIR", _S["coreset_dir"])

#: AMLworld on Kaggle. Not redistributed; fetched under your own account.
KAGGLE_DATASET: str = _env("AMLC_KAGGLE_DATASET", _S["kaggle_dataset"])

CODE_REPO: str = _S["code_repo"]
PAPER_URL: str = _S["paper"]

#: Per-split array filenames the coreset loader needs. The dataset's layout.
CORESET_ARRAYS: tuple[str, ...] = tuple(_S["coreset_arrays"])

# ── splits ───────────────────────────────────────────────────────────────
DATASETS: tuple[str, ...] = tuple(_C["datasets"])

#: Full file-order test partition sizes. The HT weights sum to these by construction.
N_TEST_FULL: dict[str, int] = dict(_C["n_test_full"])

#: Released coreset sizes.
N_CORESET: dict[str, int] = dict(_C["n_coreset"])

SEEDS: tuple[int, ...] = tuple(_C["seeds"])

#: Fixed evaluation thresholds, inherited without retuning from construction.
ML_THRESHOLDS: dict[str, float] = dict(_C["ml_thresholds"])

#: Primary evaluation ensemble; both members use file-order training partitions.
ENSEMBLE_MEMBERS: tuple[str, ...] = tuple(_C["ensemble_members"])

#: Frozen historical scorer that selected the released targets and strata.
CONSTRUCTION_MEMBERS: tuple[str, ...] = tuple(_C["construction_members"])
CONSTRUCTION_THRESHOLDS: dict[str, float] = dict(_C["construction_thresholds"])

PROMPTINGS: tuple[str, ...] = tuple(_C["promptings"])

#: The released coreset and a separate same-size ablation draw. Use ht-coreset
#: when joining the released LLM predictions; the two draws have different rows.
CORESET_DRAWS: tuple[str, ...] = tuple(_C["coreset_draws"])

# ── construction ─────────────────────────────────────────────────────────
K_HOP: int = _C["serialisation"]["k_hop"]
MAX_NEIGHBOURS_PER_HOP: int = _C["serialisation"]["max_neighbours_per_hop"]

BENIGN_MULTIPLIER: float = _C["coreset"]["benign_multiplier"]
HARD_NEG_RATIO: float = _C["coreset"]["hard_neg_ratio"]
HARD_NEG_THRESHOLD_FRAC: float = _C["coreset"]["hard_neg_threshold_frac"]

# ── audit rubric ─────────────────────────────────────────────────────────
RUBRIC_STEPS: tuple[str, ...] = tuple(_C["rubric"]["steps"])
RUBRIC_N_TRACES: int = _C["rubric"]["n_traces"]
RUBRIC_SEED: int = _C["rubric"]["seed"]

# ── the evaluated models ─────────────────────────────────────────────────
#: The seven open-weight thinking models in the main tables, in the paper's
#: order. Derived from the sampling table so the two cannot drift apart.
LLM_MODELS: tuple[str, ...] = tuple(_C["llm"]["models"])

#: Per-model sampling, as each vendor recommends for thinking mode. Table 7.
LLM_SAMPLING: dict[str, dict[str, float]] = {
    name: dict(params) for name, params in _C["llm"]["models"].items()
}

LLM_MAX_TOKENS: int = _C["llm"]["max_tokens"]
VLLM_BASE_URL: str = _env("VLLM_BASE_URL", _C["llm"]["vllm_base_url"])
VLLM_API_KEY: str = _env("VLLM_API_KEY", _C["llm"]["vllm_api_key"])

# ── the rubric annotators ────────────────────────────────────────────────
_J = _C["judges"]

JUDGE_PROVIDERS: tuple[str, ...] = tuple(_J["providers"])

#: The model that produced the released annotations for each provider. This is
#: provenance, not a tuning knob; override per provider with
#: ``AMLC_JUDGE_MODEL_DEEPSEEK`` and so on, or with ``--model``.
RELEASED_JUDGE_MODEL: dict[str, str] = dict(_J["models"])

#: Column prefix each judge's scores are written under. The released CSVs and
#: amlc.audit.iaa join on these.
JUDGE_COLUMN_PREFIX: dict[str, str] = dict(_J["column_prefix"])

JUDGE_API_KEY_ENV: dict[str, tuple[str, ...]] = {
    k: tuple(v) for k, v in _J["api_key_env"].items()
}
DEEPSEEK_BASE_URL: str = _env("DEEPSEEK_BASE_URL", _J["deepseek_base_url"])

#: Characters of reasoning trace shown to a judge. Longer traces are cut.
JUDGE_MAX_TRACE_CHARS: int = _J["max_trace_chars"]

#: Attempts per trace, with exponential backoff between them.
JUDGE_MAX_ATTEMPTS: int = _J["max_attempts"]
