"""Fetch the released evaluation set from Hugging Face.

The coreset is not vendored into this repository. It is published once, as a
dataset, and pulled on demand into the local Hugging Face cache. That keeps a
single copy authoritative: a vendored copy and a published copy drift, and the
one that drifts is always the one nobody rebuilt.

    from amlc import hub
    d = hub.load_coreset("HI-Small")     # downloads on first use, cached after

Set ``AMLC_DATASET`` to point at a fork, or ``AMLC_CORESET_DIR`` at
a local directory laid out like the ``extras/`` tree of the dataset repo, which
is what the build scripts produce before the dataset is published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .typology import TYPOLOGY_INT_MAP

DATASET_REPO = config.DATASET_REPO

#: Files that make up one split's arrays inside the dataset repo's extras tree.
_ARRAYS = config.CORESET_ARRAYS


def _local_dir() -> Optional[Path]:
    v = config.CORESET_DIR
    return Path(v) if v else None


def coreset_dir(dataset: str) -> Path:
    """Directory holding one split's arrays, downloading it if needed."""
    local = _local_dir()
    if local is not None:
        d = local / dataset
        if not d.is_dir():
            raise FileNotFoundError(
                f"AMLC_CORESET_DIR is set but {d} does not exist")
        return d

    from huggingface_hub import snapshot_download  # optional dependency

    root = snapshot_download(
        DATASET_REPO, repo_type="dataset",
        allow_patterns=[f"extras/{dataset}/*"],
    )
    d = Path(root) / "extras" / dataset

    # snapshot_download can hand back an incomplete snapshot: an interrupted
    # fetch, a cache written while the dataset was still private, or a cache
    # directory that is no longer writable all leave a directory that exists
    # with files missing from it. Reading straight through gives the caller a
    # FileNotFoundError naming a path inside the hub cache, which tells them
    # nothing about what to do. Check here instead.
    missing = [n for n in _ARRAYS if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"{DATASET_REPO} resolved to {d}, but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not there. The snapshot is "
            "incomplete rather than absent, which usually means a cached copy "
            "from an interrupted or unauthorised download. Clear it and retry:\n"
            f"  huggingface-cli delete-cache   # or remove {Path(root).parent.parent}\n"
            "If the dataset is still private you also need an account with access "
            "and either HF_TOKEN set or `huggingface-cli login`."
        )
    return d


def load_coreset(dataset: str = "HI-Small",
                 with_ensemble_probs: bool = True) -> dict:
    """Load the released HT-Coreset for one split.

    Returns indices into the full temporal test split, HT weights, labels,
    typologies and, unless turned off, the supervised ensemble probability per
    row. The weights sum to the full split size; that is asserted here because
    a coreset whose weights do not is not usable for weighted metrics.
    """
    d = coreset_dir(dataset)
    arrays = {name.removesuffix(".npy"): np.load(d / name) for name in _ARRAYS}

    total = arrays["ht_weights"].sum()
    if not np.isclose(total, config.N_TEST_FULL[dataset], rtol=0, atol=1e-6):
        raise ValueError(
            f"{dataset}: HT weights sum to {total:,.1f} but the full test split "
            f"holds {config.N_TEST_FULL[dataset]:,} edges. Weighted metrics "
            "computed against this vector would be wrong."
        )

    out = {
        "dataset": dataset,
        "subset_idx": arrays["ht_subset_indices"],
        "weights": arrays["ht_weights"],
        "labels": arrays["labels"],
        "typologies": arrays["typologies"],
        "gt_typo_str": np.array(
            [TYPOLOGY_INT_MAP.get(int(t), "legit") for t in arrays["typologies"]],
            dtype=object),
        "ml_threshold": config.ML_THRESHOLDS[dataset],
        "n": len(arrays["ht_subset_indices"]),
        "source": str(d),
    }
    if with_ensemble_probs:
        p = d / "ensemble_probs_coreset.npy"
        if p.exists():
            out["ensemble_probs"] = out["ml_probs"] = np.load(p)
    return out


def load_table(dataset: str = "HI-Small"):
    """Load the full evaluation table, prompts included, as a DataFrame."""
    from datasets import load_dataset  # optional dependency
    return load_dataset(DATASET_REPO, dataset, split="test").to_pandas()


def load_ml_weights(dataset: str, member: str) -> Path:
    """Download one supervised member's checkpoints and return the directory."""
    from huggingface_hub import snapshot_download

    if member not in config.ENSEMBLE_MEMBERS:
        raise ValueError(f"member must be one of {config.ENSEMBLE_MEMBERS}")
    root = snapshot_download(
        DATASET_REPO, repo_type="dataset",
        allow_patterns=[f"ml_baselines/weights/{dataset}/{member}/*"],
    )
    return Path(root) / "ml_baselines" / "weights" / dataset / member


def load_test_probs(dataset: str, member: str, seed: int,
                    typology: bool = False) -> np.ndarray:
    """Full-test-split probabilities for one member and seed."""
    from huggingface_hub import hf_hub_download

    name = f"seed_{seed}{'_typ' if typology else ''}.npy"
    p = hf_hub_download(
        DATASET_REPO, repo_type="dataset",
        filename=f"ml_baselines/test_probs/{dataset}/{member}/{name}",
    )
    return np.load(p)
