"""The AMLworld typology label encoding, in one place.

The integer arrays on disk (``test_typologies.npy``, the ML typology-head
predictions ``seed_*_typ.npy``, and ``data/coreset/*/typologies.npy``) are all
encoded with the ordering below. It is set by ``TYPOLOGY_TO_IDX`` in the
pre-release ``non_llm_baselines.py`` and frozen at the moment
``run_all_baselines.py`` first wrote ``test_typologies.npy``, so it is not a
choice this module gets to make.

    0 fan-out   1 fan-in   2 cycle    3 scatter-gather
    4 gather-scatter   5 stack   6 bipartite   7 random

with -1 for benign edges.

Why this module exists
----------------------
Two consumers hardcoded a *different* permutation:

    0 cycle   1 fan-out   2 stack   3 fan-in
    4 bipartite   5 scatter-gather   6 random   7 gather-scatter

Decoding the released integer arrays with it agrees with the ground-truth
string column on **0.00%** of the 791 typed HI-Small rows and 174 typed
LI-Small rows, against 100.00% for the encoding above. The failure was quiet
because the permutation was applied to both sides of the ML-vs-ground-truth
comparison, which cancels; it only surfaces where a correctly-named string
meets a wrongly-decoded one, which is exactly the LLM half of the Doubt Triage
typology metric and the whole of the per-class error analysis.

Anything that turns a typology integer into a name imports from here.
"""

from __future__ import annotations

import numpy as np

#: Canonical class order. Index == the integer stored on disk.
TYPOLOGY_CLASSES: list[str] = [
    "fan-out", "fan-in", "cycle", "scatter-gather",
    "gather-scatter", "stack", "bipartite", "random",
]

#: Integer -> name, including the benign sentinel.
TYPOLOGY_INT_MAP: dict[int, str] = {-1: "legit"} | {
    i: name for i, name in enumerate(TYPOLOGY_CLASSES)
}

#: Name -> integer.
TYPOLOGY_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(TYPOLOGY_CLASSES)}

BENIGN_LABEL = "legit"
UNKNOWN_LABEL = "unknown"


def decode(values, unknown: str = BENIGN_LABEL) -> np.ndarray:
    """Decode an integer typology array into names."""
    return np.array([TYPOLOGY_INT_MAP.get(int(v), unknown) for v in values],
                    dtype=object)


def verify_against_strings(int_values, string_values) -> dict:
    """Check the encoding against a ground-truth string column.

    Use on any artefact that carries both, for example
    ``data/coreset/<dataset>/typologies.npy`` beside the ``typology`` column of
    ``case_index.csv``. Returns a report; raises when they disagree.
    """
    ints = np.asarray(int_values)
    strings = np.asarray(string_values, dtype=object)
    typed = (ints >= 0) & np.array([isinstance(s, str) for s in strings])
    if not typed.any():
        return {"n_typed": 0, "checked": False}

    decoded = decode(ints[typed])
    agree = decoded == strings[typed]
    report = {"n_typed": int(typed.sum()), "checked": True,
              "agreement": float(agree.mean())}
    if not agree.all():
        wrong = int((~agree).sum())
        example = (decoded[~agree][0], strings[typed][~agree][0])
        raise ValueError(
            f"typology encoding disagrees with the ground-truth strings on "
            f"{wrong}/{typed.sum()} rows (decoded {example[0]!r} where the "
            f"string column says {example[1]!r}). The integer arrays were not "
            "written with the encoding in amlc.typology."
        )
    return report
