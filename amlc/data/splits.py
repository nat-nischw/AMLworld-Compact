"""The temporal 60/20/20 cut of AMLworld, computed in one place.

AMLworld ships each variant's transactions already ordered by timestamp, so
the split of Altman et al. (2023) is a pair of row indices rather than a date
comparison:

    train   rows [0, t1)      the first 60 per cent
    val     rows [t1, t2)     the next 20 per cent
    test    rows [t2, n)      the last 20 per cent

Two stages needed those indices before the release and each computed them
itself: the case builder in ``data_loader.py`` and the Snap ML feature matrix
in ``feature_extractor.py``. The arithmetic agreed, but nothing held it
together, and the boundaries decide which edges the supervised baselines ever
see. They now come from :func:`temporal_boundaries` in both places.

The ratios are floats and the truncation is deliberate. ``int(n * 0.6)`` and
``int(n * (0.6 + 0.2))`` are the exact expressions the published run used, and
they reproduce the test-split sizes in :data:`amlc.config.N_TEST_FULL`
(1,015,669 on HI-Small and 1,384,810 on LI-Small). Rewriting them as ``n * 4 //
5`` or ``int(n * 0.8)`` is not obviously equivalent for every n, so they stay
as they were.

Nothing here reads or writes a file.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

#: Split proportions, in the order the rows appear.
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

#: Split names, in temporal order. An edge belongs to exactly one.
SPLITS = ("train", "val", "test")

T = TypeVar("T", bound=Sequence)


def temporal_boundaries(
    n_total: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[int, int]:
    """Row indices ``(t1, t2)`` of the two cuts, for ``n_total`` transactions.

    ``t1`` is the first validation row and ``t2`` the first test row, so the
    three splits are ``[0:t1]``, ``[t1:t2]`` and ``[t2:]``.
    """
    t1_idx = int(n_total * train_ratio)
    t2_idx = int(n_total * (train_ratio + val_ratio))
    return t1_idx, t2_idx


def split_of(row: int, t1_idx: int, t2_idx: int) -> str:
    """The split a transaction row belongs to."""
    if row < t1_idx:
        return "train"
    if row < t2_idx:
        return "val"
    return "test"


def split_three(values: T, t1_idx: int, t2_idx: int) -> tuple[T, T, T]:
    """Cut a row-aligned sequence or array into ``(train, val, test)``."""
    return values[:t1_idx], values[t1_idx:t2_idx], values[t2_idx:]


def describe(n_total: int, t1_idx: int, t2_idx: int) -> str:
    """One-line log message naming the three row ranges."""
    return (f"train[0:{t1_idx:,}] val[{t1_idx:,}:{t2_idx:,}] "
            f"test[{t2_idx:,}:{n_total:,}]")
