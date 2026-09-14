"""Build paths into the archived experiment run directory.

The 18 GB run directory this project produced uses its own filenames and
directory names, fixed before the paper's vocabulary was settled. Nothing in
the package hardcodes them: every path into that tree is built here, so the
rest of the code speaks only the names the paper and the dataset use.

Only regeneration touches the archive. A clone that consumes the released
artefacts never imports this module.
"""

from __future__ import annotations

from pathlib import Path

from .config import CORESET_DRAWS, PROMPTINGS

#: paper draw name -> filename stem in the archive
_DRAW_STEM = {"ht-coreset": "v2_best", "ablation-redraw": "V2_full"}

#: paper prompting name -> directory name in the archive
_PROMPTING_DIR = {
    "ICL-FS": "LLM+ICL-AML",
    "ICL-ZS": "LLM+ICL-AML-NoFS",
    "ICL-V": "LLM+ICL-AML-V",
}

#: paper ensemble-member name -> directory name in the archive.
#: GCPAL is stored under the name of the inference pass that produced its
#: probabilities, not the checkpoint that produced them; see
#: baselines/ml/gcpal_infer.py.
_MEMBER_DIR = {
    "LightGBM+GFP": "LightGBM+GFP",
    "XGBoost+GFP": "XGBoost+GFP",
    "GCPAL+GFP": "GCPAL_knn_temporal",
}

#: directory holding the serialised prompts in the archive
#: Paper ensemble-member name -> (parent directory under outputs/, directory
#: holding the checkpoints). GCPAL is the odd one: its weights are filed under
#: the training run's name and its probabilities under the inference pass's.
_MEMBER_WEIGHTS = {
    "LightGBM+GFP": ("saved_models", "LightGBM+GFP"),
    "XGBoost+GFP": ("saved_models", "XGBoost+GFP"),
    "GCPAL+GFP": ("models", "GCPAL_knn"),
}

SERIALISED_DIR = "v2_subset"

#: Deferral-strategy names. The archive spells the paper's Doubt Triage rule
#: differently; accepted on input so an archived config still runs.
_STRATEGY_ALIASES = {"wasd": "dt"}


def draw_stem(draw: str) -> str:
    """Filename stem in the archived tree for a paper draw name."""
    if draw not in _DRAW_STEM:
        raise ValueError(f"draw must be one of {CORESET_DRAWS}, got {draw!r}")
    return _DRAW_STEM[draw]


def prompting_dir(prompting: str) -> str:
    """Directory name in the archived tree for a paper prompting name."""
    if prompting not in _PROMPTING_DIR:
        raise ValueError(f"prompting must be one of {PROMPTINGS}, got {prompting!r}")
    return _PROMPTING_DIR[prompting]


def member_dir(member: str) -> str:
    """Directory name in the archived tree for a paper ensemble-member name."""
    if member not in _MEMBER_DIR:
        raise ValueError(f"unknown ensemble member {member!r}")
    return _MEMBER_DIR[member]


def member_weights_dir(member: str) -> tuple[str, str]:
    """Parent and checkpoint directory in the archived tree for a member."""
    try:
        return _MEMBER_WEIGHTS[member]
    except KeyError:
        raise KeyError(f"unknown ensemble member {member!r}; "
                       f"expected one of {sorted(_MEMBER_WEIGHTS)}") from None


def canonical_strategy(strategy: str) -> str:
    """Map a deprecated strategy alias onto its paper name."""
    return _STRATEGY_ALIASES.get(strategy, strategy)


def legacy_path(archive: Path, kind: str, dataset: str, **kw) -> Path:
    """Build a path into the archived experiment tree.

    ``kind`` is one of ``subset``, ``weights``, ``construction_probs``,
    ``ensemble_probs`` (legacy alias for construction scores),
    ``test_labels``, ``test_typologies``, ``member_probs``, ``member_typology``,
    ``serialised``, ``predictions``.
    """
    archive = Path(archive)
    es = archive / "eval_subsets" / dataset
    tp = archive / "test_probs" / dataset

    if kind == "subset":
        return es / f"subset_{dataset}_{draw_stem(kw['draw'])}.npy"
    if kind == "weights":
        return es / f"weights_{dataset}_{draw_stem(kw['draw'])}.npy"
    if kind in ("construction_probs", "ensemble_probs"):
        # Frozen three-scorer construction array. Never replace with evaluation
        # scores: the released subset and its weights depend on these values.
        return es / f"ens_probs_{dataset}.npy"
    if kind == "test_labels":
        return tp / "test_labels.npy"
    if kind == "test_typologies":
        return tp / "test_typologies.npy"
    if kind == "member_probs":
        return tp / kw["member"] / f"seed_{kw['seed']}.npy"
    if kind == "member_typology":
        return tp / kw["member"] / f"seed_{kw['seed']}_typ.npy"
    if kind == "serialised":
        return archive / "llm_datasets" / dataset / SERIALISED_DIR
    if kind == "predictions":
        return (archive / kw["model"] / prompting_dir(kw["prompting"]) / dataset
                / f"seed_{kw['seed']}.json")
    raise ValueError(f"unknown kind {kind!r}")
