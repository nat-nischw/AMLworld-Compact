"""The three-rater human validation of the four-step rubric.

Nothing on disk computed these tables: the appendix numbers were read off the
rater spreadsheets by hand. This module derives them, so the human-rating
appendix has a script behind it like every other table.

Three annotators, one NLP researcher and two AML practitioners (junior and
senior), independently scored the same 28-trace slice: 7 models by 4 outcome
classes, one trace per cell, all HI-Small at seed 42. Conclude is binarised
under the uniform parser-extraction rule described in
``results/human_rating/README.md``.

It reproduces the appendix exactly. Per-rater Conclude is 21.4% for all three
(6 of 28), three-way Fleiss' kappa on Conclude is 1.00, Match is 0.10, the
per-rater pass rates are 100.0 / 100.0 / 53.6 for the NLP rater, 100.0 / 100.0 /
96.4 for the junior and 96.4 / 100.0 / 67.9 for the senior on Parse / Recall /
Match, and the outcome breakdown gives Conclude 85.7% on correct-illicit and
0.0% on all three failure classes.

Parse and Recall are the two cells the appendix marks as a ceiling effect, and
the flag this module prints says so rather than reporting a number that reads
like agreement. Recall is unanimous on all 28 items, so the expected agreement
is exactly 1 and kappa is undefined; the shared implementation returns 1.0 in
that degenerate case. Parse is unanimous on 27 of 28 and computes to -0.01,
which is a ceiling artefact and not disagreement.

Reads
    ``results/human_rating/rater_{nlp,aml_practitioner_1,aml_practitioner_2}.csv``

Writes
    ``results/human_rating/human_rating_summary.csv`` (per-rater pass rates and
    the per-step agreement) and ``human_rating_by_outcome.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from .iaa import fleiss_kappa

#: Sheet stem to the label the appendix uses for that rater.
RATERS = {"nlp": "NLP", "aml_practitioner_1": "Jr.", "aml_practitioner_2": "Sr."}

#: Expected agreement above this counts as a ceiling: kappa is uninformative
#: because the raters had almost no room to disagree by chance. The appendix
#: marks Parse and Recall this way at 27/28 and 28/28 unanimity, which put
#: expected agreement at 0.98 and 1.00; Match and Conclude sit near 0.6.
CEILING_PE = 0.95


def load_raters(directory: Path) -> dict[str, pd.DataFrame]:
    """One frame per rater, indexed by case identifier and aligned across all."""
    frames = {name: pd.read_csv(directory / f"rater_{name}.csv").set_index("case_id")
              for name in RATERS}
    index = next(iter(frames.values())).index
    return {name: frame.loc[index] for name, frame in frames.items()}


def ratings(frames: dict[str, pd.DataFrame], step: str) -> np.ndarray:
    """Item-by-rater 0/1 matrix for one step."""
    return np.column_stack([f[f"rater_{step}"].astype(float).astype(int).values
                            for f in frames.values()])


def agreement(votes: np.ndarray) -> dict:
    """Fleiss' kappa for one step, with the ceiling diagnostic beside it."""
    n_raters = votes.shape[1]
    counts = np.column_stack([n_raters - votes.sum(axis=1), votes.sum(axis=1)])
    marginals = counts.sum(axis=0) / counts.sum()
    p_e = float((marginals ** 2).sum())
    return {
        "fleiss_kappa": float(fleiss_kappa(counts)),
        "expected_agreement": p_e,
        "unanimous_items": int((counts.max(axis=1) == n_raters).sum()),
        "n_items": int(len(votes)),
        "ceiling": bool(p_e >= CEILING_PE),
    }


def summarise(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-step pass rates and agreement, plus the outcome breakdown."""
    per_step, per_outcome = [], []
    outcomes = list(dict.fromkeys(next(iter(frames.values()))["outcome"]))

    for step in config.RUBRIC_STEPS:
        row = {"step": step}
        row.update({label: round(frames[name][f"rater_{step}"].astype(float).mean() * 100, 1)
                    for name, label in RATERS.items()})
        row["mean"] = round(float(np.mean([row[label] for label in RATERS.values()])), 1)
        row.update(agreement(ratings(frames, step)))
        per_step.append(row)

        for outcome in outcomes:
            rates = [f.loc[f["outcome"] == outcome, f"rater_{step}"].astype(float).mean()
                     for f in frames.values()]
            per_outcome.append({
                "outcome": outcome, "step": step,
                "n": int((next(iter(frames.values()))["outcome"] == outcome).sum()),
                "pass_pct": round(float(np.mean(rates)) * 100, 1),
            })

    by_outcome = (pd.DataFrame(per_outcome)
                  .pivot(index="outcome", columns="step", values="pass_pct")
                  .reindex(columns=list(config.RUBRIC_STEPS)).reset_index())
    return pd.DataFrame(per_step), by_outcome


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=None,
                    help="rater sheets (default: <repo>/results/human_rating)")
    args = ap.parse_args()

    from ..paths import ensure, results

    directory = Path(args.dir) if args.dir else ensure(results() / "human_rating")
    frames = load_raters(directory)
    per_step, by_outcome = summarise(frames)

    print(f"Three raters, {per_step['n_items'].iloc[0]} traces\n")
    print(per_step[["step", *RATERS.values(), "mean", "fleiss_kappa",
                    "unanimous_items", "ceiling"]].to_string(index=False))
    print("\nPass rate by outcome class, mean across raters\n")
    print(by_outcome.to_string(index=False))

    per_step.to_csv(directory / "human_rating_summary.csv", index=False)
    by_outcome.to_csv(directory / "human_rating_by_outcome.csv", index=False)
    print(f"\nSaved {directory / 'human_rating_summary.csv'} and "
          f"{directory / 'human_rating_by_outcome.csv'}")


if __name__ == "__main__":
    main()
