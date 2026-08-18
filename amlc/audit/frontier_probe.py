#!/usr/bin/env python3
"""Frontier-API context-engineering probe: extraction and scoring.

Three proprietary reasoning APIs (DeepSeek-V4-Pro, Gemini 3.1 Pro, Claude
Sonnet 4.6) were run over six prompt designs on a 198-case pool, giving the
eighteen cells of the frontier-probe table in the appendix. This module holds
the two halves of that table's provenance: :func:`extract`, which turns the
archived API responses into a redistributable predictions table, and
:func:`score`, which turns that table back into the published numbers.

What is redistributable, and what is not
----------------------------------------

The archived responses carry a ``thinking`` field, the chain of thought returned
by a proprietary API, averaging about seven thousand characters per case. That
is the most terms-sensitive category in this release and it is not
redistributed. Neither is ``response_text``.

What ships is ``results/frontier_probe/predictions.csv``: for each of the 4,356
model-variant-case rows, the gold class, the parsed typology, the parsed binary
verdict, the token counts and whether the call errored. Every number in the
published table is recomputable from those columns alone, which is the point.
:func:`extract` is included so the step is auditable by anyone holding the raw
responses; it is not runnable from the release alone.

The probe pool
--------------

The 198 cases are a stratified subsample of the released HI-Small coreset: 22
cases for each of the nine classes, so 176 illicit against 22 benign, an 8:1
balance chosen to differ from the 1:2 of the main evaluation. All 198
identifiers resolve inside ``data/coreset/HI-Small`` and their labels agree with
the released coreset on all 198, so the probe is checkable against the published
dataset rather than against a private pool. Identifiers were retagged from
``v2_`` to ``amlc_`` by :func:`amlc.case_ids.retag`.

The scoring convention, recovered rather than documented
---------------------------------------------------------

No generator for this table survives in any repository, so the convention was
recovered by fitting candidate rules to the ninety published numbers. The
predicted class is ``parsed.observed_pattern``, with ``legitimate`` meaning the
negative class. What the rest of the recovery turned up is worth stating
carefully, because it is a camera-ready item rather than implementation trivia.

**The model's own binary answer is not used.** ``parsed.conclusion`` plays no
part in any published column. Scoring detection from it instead reproduces 31 of
the 54 detection numbers and none of DeepSeek's six cells. So the probe's
detection metric measures whether the model named a laundering typology, not
whether it said the word suspicious. For *FS-TypFirst* that is the stated
design, since its prompt derives the verdict from the typology. For the other
five variants it is not what the prompt asked for, and the appendix does not say
so.

**The table's columns do not agree on what an unparsed response means.** 411 of
the 3,564 scored responses parsed no typology at all. Fitting each column
separately gives two different answers, and neither fits both:

===========================  ==========  =========  ========
convention for unparsed      Det (54)    Typ-F1     Typ-Acc
===========================  ==========  =========  ========
counts as a benign answer    **54/54**   **18/18**  11/18
counts as no answer at all   15/54       5/18       **18/18**
===========================  ==========  =========  ========

So Det-F1/P/R and Typ-F1 were computed treating a parse failure as a confident
benign prediction, while Typ-Acc in the same row was computed treating it as a
non-answer that matches nothing. Each convention is defensible alone. Using both
in one table is not, because a reader compares Typ-F1 against Typ-Acc across the
row. The gap is widest for *FS-CoT-Elim* on Claude Sonnet, where 95 of 198
responses failed to parse: the table prints Typ-Acc 15.2, and the convention its
own Typ-F1 column uses would print 21.2.

:func:`score` therefore takes a ``convention`` argument. ``"published"``
reproduces the table exactly, mixed rules and all, and :func:`verify` asserts
that all ninety numbers come back. ``"benign"`` and ``"unanswered"`` apply one
rule to every column, which is what a corrected table should do; running this
module as a script prints both and the deltas between them.

``results/frontier_probe/metrics_archived/`` holds the twenty-two
``metrics.json`` aggregates written at run time by the original analysis script,
``analyze_vllm_pilot.py``. Its ``detection_accuracy`` uses ``parsed.conclusion``
and therefore disagrees with the published table. They ship as a record of what
was computed at the time, not as a source for the table.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

from amlc import case_ids, paths

#: Directory name per model, and the paper's name for it.
MODELS = {
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}

#: Archive variant directory, and the paper's name for it. The archive numbering
#: is not contiguous: ``V3`` was designed and never run, and the abstain variant
#: that ran is the softened ``V4p1`` rather than the ``V4`` the appendix boxes.
#: ``prompts/frontier_probe/README.md`` records the four places the appendix and
#: the executed runs disagree.
# These V-numbers are prompt variants of the probe. They are unrelated to the
# pre-release name of the HT-Coreset, which also appears as "V2" in archived
# filenames such as v2_best.npy; see amlc.archive.
VARIANTS = {
    "V0": "ICL-ZS",
    "V1": "ICL-FS",
    "V2": "FS-Base",
    "V4p1": "ZS-Abstain",
    "V7": "FS-CoT-Elim",
    "V8": "FS-TypFirst",
}

#: The nine probe classes. ``benign`` is the negative class; the other eight are
#: the AMLworld typologies in the order of :mod:`amlc.typology`.
CLASSES = [
    "benign", "fan-out", "fan-in", "cycle", "scatter-gather",
    "gather-scatter", "stack", "bipartite", "random",
]

BENIGN = "benign"

#: The probe prompts' word for the negative class.
LEGITIMATE = "legitimate"

#: Stand-in class for a response that parsed no typology. Never equal to a gold
#: label, so under the ``"unanswered"`` convention it can only ever be a miss.
UNANSWERED = "<unanswered>"

#: A fourth model, Claude Opus 4.7, was attempted on four variants and returned
#: an error on every one of its 198 cases in all four, so it carries no result.
#: It stays in the predictions table and is excluded from the scored table, so
#: the failure is visible rather than absent.
FAILED_MODELS = {"claude-opus-4-7"}

#: The published table, for :func:`verify`. Rows are (Det-F1, Det-P, Det-R,
#: Typ-F1, Typ-Acc) in percent, as printed in ``tables/probe_frontier_apis.tex``.
PUBLISHED = {
    ("deepseek-v4-pro", "V0"):   (89.3, 88.8, 89.8, 20.9, 25.8),
    ("deepseek-v4-pro", "V1"):   (89.0, 88.7, 89.2, 17.3, 20.7),
    ("deepseek-v4-pro", "V2"):   (86.4, 90.1, 83.0, 26.5, 25.8),
    ("deepseek-v4-pro", "V4p1"): (51.2, 90.0, 35.8, 16.1, 21.7),
    ("deepseek-v4-pro", "V7"):   (82.3, 88.8, 76.7, 21.3, 23.7),
    ("deepseek-v4-pro", "V8"):   (90.2, 89.0, 91.5, 30.3, 32.3),
    ("gemini-3.1-pro-preview", "V0"):   (81.5, 87.6, 76.1, 24.5, 26.8),
    ("gemini-3.1-pro-preview", "V1"):   (80.7, 87.4, 75.0, 23.5, 26.3),
    ("gemini-3.1-pro-preview", "V2"):   (82.4, 90.5, 75.6, 25.5, 27.3),
    ("gemini-3.1-pro-preview", "V4p1"): (37.6, 97.6, 23.3, 11.9, 20.7),
    ("gemini-3.1-pro-preview", "V7"):   (70.5, 88.8, 58.5, 19.5, 23.7),
    ("gemini-3.1-pro-preview", "V8"):   (85.5, 87.5, 83.5, 27.3, 27.8),
    ("claude-sonnet-4-6", "V0"):   (91.8, 88.8, 94.9, 12.4, 18.7),
    ("claude-sonnet-4-6", "V1"):   (91.5, 88.8, 94.3, 12.5, 18.2),
    ("claude-sonnet-4-6", "V2"):   (90.8, 89.5, 92.0, 21.0, 25.3),
    ("claude-sonnet-4-6", "V4p1"): (80.4, 92.6, 71.0, 18.0, 21.7),
    ("claude-sonnet-4-6", "V7"):   (66.9, 91.2, 52.8, 17.9, 15.2),
    ("claude-sonnet-4-6", "V8"):   (93.5, 89.6, 97.7, 21.1, 25.8),
}

#: ``pred_pattern`` is the raw parsed ``observed_pattern``, empty when the
#: response parsed no typology. It is deliberately *not* folded onto the label
#: space on the way out, because the published columns disagree about what an
#: empty value means and a pre-folded table could not reproduce both.
FIELDS = [
    "model", "variant", "case_id", "gold_class", "gold_label",
    "pred_pattern", "pred_conclusion", "errored",
    "input_tokens", "output_tokens", "latency_ms",
]

CONVENTIONS = ("published", "benign", "unanswered")


def predictions_path() -> Path:
    """Where the redistributable predictions table lives."""
    return paths.results() / "frontier_probe" / "predictions.csv"


def canonical_class(pred_pattern: str, unparsed: str = BENIGN) -> str:
    """Fold a raw ``observed_pattern`` onto the nine-class label space.

    ``legitimate`` always becomes ``benign``. An empty value, meaning the
    response parsed no typology, becomes whatever ``unparsed`` says, and that
    choice is the whole of the disagreement described in the module docstring.
    """
    if not pred_pattern:
        return unparsed
    return BENIGN if pred_pattern == LEGITIMATE else pred_pattern


# --------------------------------------------------------------------------- #
# Extraction. Needs the archived API responses, which are not redistributable.
# --------------------------------------------------------------------------- #

def responses_root() -> Path:
    """Directory holding ``<model>/<variant>/responses.jsonl``."""
    p = os.environ.get("AMLC_PROBE_RESPONSES")
    if p:
        return Path(p)
    raise FileNotFoundError(
        "The raw frontier-API responses are not part of the release: they "
        "contain chain-of-thought returned by proprietary APIs. Set "
        "AMLC_PROBE_RESPONSES to a directory of <model>/<variant>/responses.jsonl "
        "if you hold them. To reproduce the published table you do not need "
        f"them; score {predictions_path()} instead."
    )


def extract(root: Path | None = None, out: Path | None = None) -> Path:
    """Write the redistributable predictions table from archived responses.

    Drops ``thinking`` and ``response_text``, keeps the parsed fields and the
    call metadata, and retags identifiers onto the ``amlc_`` vocabulary.
    """
    root = root or responses_root()
    out = out or predictions_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, seen = [], Counter()
    for f in sorted(root.glob("*/*/responses.jsonl")):
        for line in f.open():
            if not line.strip():
                continue
            r = json.loads(line)
            parsed = r.get("parsed") or {}
            case_id, n = case_ids.retag(r["case_id"])
            seen["retagged"] += n
            for withheld in ("thinking", "response_text"):
                if r.get(withheld):
                    seen[withheld] += 1
            usage = r.get("usage") or {}
            rows.append({
                "model": r["model"],
                "variant": r["variant"],
                "case_id": case_id,
                "gold_class": r.get("class", ""),
                "gold_label": int(r.get("label", 0)),
                "pred_pattern": parsed.get("observed_pattern") or "",
                "pred_conclusion": parsed.get("conclusion") or "",
                "errored": int(bool(r.get("error"))),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "latency_ms": round(float(r.get("latency_ms") or 0.0), 1),
            })

    if any(case_ids.is_case_id(r["case_id"]) is False for r in rows):
        raise AssertionError("extraction produced an identifier outside the amlc_ vocabulary")

    rows.sort(key=lambda d: (d["model"], d["variant"], d["case_id"]))
    # lineterminator is pinned so re-running extraction reproduces the committed
    # file byte for byte; csv defaults to CRLF, which git normalises on the way
    # in and would leave the working tree dirty after every run.
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return out


# --------------------------------------------------------------------------- #
# Scoring. Runs from the shipped CSV alone.
# --------------------------------------------------------------------------- #

def _binary_prf(gold_illicit, pred_illicit):
    tp = sum(1 for g, p in zip(gold_illicit, pred_illicit) if g and p)
    fp = sum(1 for g, p in zip(gold_illicit, pred_illicit) if not g and p)
    fn = sum(1 for g, p in zip(gold_illicit, pred_illicit) if g and not p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1 * 100, prec * 100, rec * 100


def _macro_f1(gold, pred, labels=CLASSES):
    per = []
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(per) / len(per) * 100


def load(path: Path | None = None) -> dict[tuple[str, str], list[dict]]:
    """Group the predictions table into cells."""
    path = Path(path or predictions_path())
    cells: dict[tuple[str, str], list[dict]] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            cells.setdefault((r["model"], r["variant"]), []).append(r)
    return cells


def score(path: Path | None = None, convention: str = "published") -> dict[tuple[str, str], dict]:
    """Score every cell from the predictions table.

    ``convention`` selects how a response that parsed no typology is treated.
    ``"published"`` reproduces the table exactly, which means treating it as a
    benign answer for Det and Typ-F1 and as a non-answer for Typ-Acc.
    ``"benign"`` and ``"unanswered"`` apply one rule to every column.

    Returns ``{(model, variant): {det_f1, det_p, det_r, typ_f1, typ_acc, n,
    n_errors, n_unparsed}}``, with the wholly failed model excluded.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")

    out = {}
    for key, rows in sorted(load(path).items()):
        if key[0] in FAILED_MODELS:
            continue
        gold = [r["gold_class"] for r in rows]
        as_benign = [canonical_class(r["pred_pattern"], BENIGN) for r in rows]
        as_unanswered = [canonical_class(r["pred_pattern"], UNANSWERED) for r in rows]

        if convention == "published":
            det_pred, f1_pred, acc_pred = as_benign, as_benign, as_unanswered
        elif convention == "benign":
            det_pred = f1_pred = acc_pred = as_benign
        else:
            det_pred = f1_pred = acc_pred = as_unanswered

        f1, prec, rec = _binary_prf(
            [int(r["gold_label"]) == 1 for r in rows],
            [c not in (BENIGN, UNANSWERED) for c in det_pred],
        )
        out[key] = {
            "det_f1": f1, "det_p": prec, "det_r": rec,
            "typ_f1": _macro_f1(gold, f1_pred),
            "typ_acc": sum(1 for g, p in zip(gold, acc_pred) if g == p) / len(rows) * 100,
            "n": len(rows),
            "n_errors": sum(int(r["errored"]) for r in rows),
            "n_unparsed": sum(1 for r in rows if not r["pred_pattern"]),
        }
    return out


def verify(path: Path | None = None, tol: float = 0.06) -> None:
    """Raise unless every published cell is reproduced from the shipped table.

    ``tol`` is half a printed unit: the table carries one decimal, so anything
    within 0.06 rounds to the same string.
    """
    got = score(path, convention="published")
    missing = set(PUBLISHED) - set(got)
    if missing:
        raise AssertionError(f"no scored cell for {sorted(missing)}")
    bad = []
    for key, want in PUBLISHED.items():
        g = got[key]
        for name, a, b in zip(
            ("Det-F1", "Det-P", "Det-R", "Typ-F1", "Typ-Acc"),
            (g["det_f1"], g["det_p"], g["det_r"], g["typ_f1"], g["typ_acc"]),
            want,
        ):
            if abs(a - b) > tol:
                bad.append(f"{key[0]}/{key[1]} {name}: got {a:.1f}, table says {b:.1f}")
    if bad:
        raise AssertionError(
            f"{len(bad)} of {len(PUBLISHED) * 5} published numbers not reproduced:\n  "
            + "\n  ".join(bad)
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extract", action="store_true",
                    help="rebuild predictions.csv from AMLC_PROBE_RESPONSES")
    ap.add_argument("--convention", choices=CONVENTIONS, default="published")
    ap.add_argument("--predictions", type=Path, default=None)
    args = ap.parse_args()

    if args.extract:
        print(f"wrote {extract(out=args.predictions)}")

    got = score(args.predictions, convention=args.convention)
    hdr = (f"{'model':<18}{'variant':<13}{'Det-F1':>8}{'Det-P':>8}{'Det-R':>8}"
           f"{'Typ-F1':>8}{'Typ-Acc':>9}{'unparsed':>10}")
    print(f"\nconvention: {args.convention}")
    print(hdr)
    print("-" * len(hdr))
    for (m, v), g in got.items():
        print(f"{MODELS.get(m, m):<18}{VARIANTS.get(v, v):<13}"
              f"{g['det_f1']:8.1f}{g['det_p']:8.1f}{g['det_r']:8.1f}"
              f"{g['typ_f1']:8.1f}{g['typ_acc']:9.1f}{g['n_unparsed']:10d}")

    verify(args.predictions)
    print(f"\nall {len(PUBLISHED) * 5} published numbers reproduced from "
          f"{(args.predictions or predictions_path()).name}")

    if args.convention == "published":
        pub = got
        for alt in ("benign", "unanswered"):
            other = score(args.predictions, convention=alt)
            moved = [(k, pub[k]["typ_acc"], other[k]["typ_acc"])
                     for k in pub if abs(pub[k]["typ_acc"] - other[k]["typ_acc"]) > 0.06]
            worst = max((abs(a - b) for _, a, b in moved), default=0.0)
            print(f"under one consistent rule ({alt}): Typ-Acc moves in "
                  f"{len(moved)} of {len(pub)} cells, by up to {worst:.1f} points")


if __name__ == "__main__":
    main()
