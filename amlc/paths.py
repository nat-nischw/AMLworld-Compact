"""Every filesystem location the package uses, resolved in one place.

Two roots, and the distinction matters:

``repo``
    This checkout. Holds the small inputs a clone can run from: the released
    coreset under ``data/``, the demonstration pool, the tuned hyperparameters,
    the prompt templates, and every result table under ``results/``.

``archive``
    The original run directory, with the 18 GB of intermediate outputs. Only
    regeneration needs it, and it is never assumed to exist. Set
    ``AMLC_ARCHIVE`` to enable those paths; anything that needs it and
    cannot find it raises rather than silently writing somewhere unexpected.

Nothing else in the package builds a path from a string literal. Overrides go
through the environment so a clone is not tied to one machine:

===========================  ===============================================
``AMLC_REPO``          repo root, default: this file's grandparent
``AMLC_DATA``          default ``<repo>/data``
``AMLC_RESULTS``       default ``<repo>/results``
``AMLC_PROMPTS``       default ``<repo>/prompts``
``AMLC_FIGURES``       default ``<repo>/results/figures``
``AMLC_ARCHIVE``       the run directory's ``outputs/``; no default
===========================  ===============================================
"""

from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: Path | None = None) -> Path | None:
    v = os.environ.get(name)
    return Path(v) if v else default


def repo() -> Path:
    """Root of this checkout."""
    return _env("AMLC_REPO", Path(__file__).resolve().parents[1])


def data() -> Path:
    return _env("AMLC_DATA", repo() / "data")


def results() -> Path:
    return _env("AMLC_RESULTS", repo() / "results")


def prompts() -> Path:
    return _env("AMLC_PROMPTS", repo() / "prompts")


def figures() -> Path:
    return _env("AMLC_FIGURES", results() / "figures")


def coreset(dataset: str) -> Path:
    """The released coreset for one dataset, shipped in the clone."""
    return data() / "coreset" / dataset


def icl_examples(dataset: str) -> Path:
    return data() / "icl_examples" / dataset / "icl_examples.json"


def tuned_params(dataset: str, method: str, typology: bool = False) -> Path:
    """Optuna result for one supervised baseline."""
    suffix = "_typology" if typology else ""
    return data() / "tuned_params" / dataset / f"{method}{suffix}.json"


def archive(required: bool = True) -> Path | None:
    """The original run directory's ``outputs/``.

    Raises when ``required`` and ``AMLC_ARCHIVE`` is unset, because the
    alternative is a stage silently reading or writing the wrong tree.
    """
    p = _env("AMLC_ARCHIVE")
    if p is None:
        if not required:
            return None
        raise FileNotFoundError(
            "This stage regenerates from the original run directory. Set "
            "AMLC_ARCHIVE to its outputs/ directory, or use the released "
            "artefacts under data/ and results/ instead."
        )
    return p if p.name == "outputs" else p / "outputs"


def ensure(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
