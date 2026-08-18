"""Score-only deferral: the grid-searched baseline of Appendix G.3.

Doubt Triage decides what to send to the LLM by asking which Horvitz-Thompson
stratum an edge belongs to. That is a property of how the evaluation set was
drawn, not a property of the edge, so a deployed system cannot compute it. Doubt
Triage is therefore an upper bound on what selective deferral can buy, and this
module is the comparison that is actually implementable: gate on the supervised
ensemble score alone, and grid search the gate.

The appendix reports the peak at 66.0% weighted F1 on HI-Small. The archived
per-cell grids bracket that figure: the ten HI-Small cells peak between 65.9%
and 67.2%, against 65.9% for the ensemble alone at the same gate. The gain is
that small because any band wide enough to admit sampled-stratum edges pays
that stratum's weight, about 470 on HI-Small, for every benign edge the LLM
flags.

Two gates, searched independently, as in the archived script:

``recall_gate``
    The ensemble's own illicit predictions stand. Below the decision threshold
    and above ``tau``, consult the LLM and flip to illicit when it says so. The
    LLM can only add detections, so recall rises and precision can only fall
    through the flips.
``precision_gate``
    The ensemble's legit predictions stand. Above the decision threshold and
    below ``tau``, consult the LLM and flip to legit when it disagrees. The LLM
    can only remove detections.

The ensemble decision threshold used by both gates is 0.5, which is what the
archived grid ran. The paper's operating points are :data:`config.ML_THRESHOLDS`
(0.80 and 0.48), so the ensemble-only reference inside this grid is 65.9% on
HI-Small rather than Table 4's 70.9%. Pass ``ml_threshold`` to move it.

Reads
    The archived run directory, because the LLM predictions only exist there:
    the coreset draw and its HT weights, the ensemble probabilities, and one
    prediction JSON per (model, prompting, seed). Everything is reached through
    :mod:`amlc.archive`.

Writes
    One ``score_deferral_grid_<dataset>_<model>_<prompting>.csv`` per cell, plus
    ``score_deferral_summary.csv`` holding the best gate of each kind. Default
    output directory is ``results/triage``.

Fixed here
----------
The routing array was built as ``np.where(ml_preds == 1, "ml_illicit",
"ml_legit")``. Numpy infers the dtype from those two literals as ``<U10``, and the two
labels assigned afterwards are longer: ``llm_consulted`` is 13 characters and
``llm_flipped`` is 11, so both are truncated on assignment and neither
``routing == "llm_consulted"`` nor ``routing == "llm_flipped"`` ever matches.
Every ``n_llm_consulted``, ``n_llm_flipped`` and ``llm_pct`` value in the
archived grid CSVs is 0 as a result, including at the reported optimum. The
routing array is built with dtype ``object`` here so the labels survive. No F1
column moves: the predictions themselves are assigned through boolean masks and
never through the routing labels.

What a rerun reproduces
-----------------------
Against ``--draw ablation-redraw --archived-weights``, which is the
configuration the archived grids were written under, the ensemble half comes
back to the digit: the ungated row at tau = 0.5 gives TP 744, FP 263 and F1
0.659 on HI-Small with GPT-OSS-120B, matching
``results/triage/grid_HI-Small_GPT-OSS-120B_LLM+ICL-AML.csv`` exactly. Every
gated row moves, by up to 1.1 pp F1, and that cell's peak goes from 66.1% to
66.9%. The cause is on the LLM side: the prediction JSONs were repatched after
the grid was written, so a rerun scores verdicts the archived grid never saw.
Nothing in this module explains the difference, and the ungated row is the
control that shows it.

Dropped
-------
The pre-release ``hybrid_predict`` (superseded by the two gates and marked
"kept for backward compat"), ``eval_at_thresholds`` (its only caller was
``hybrid_predict``), and the three plotting functions. ``plot_heatmap`` pivoted
on ``tau_low`` and ``tau_high`` columns that this grid has never produced, and
none of the three figures appears in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .. import config
from ..config import CORESET_DRAWS
from .doubt_triage import (
    ht_weighted_prf,
    load_coreset_from_archive,
    load_llm_predictions,
    resolve_archive,
)

#: Ensemble decision threshold the archived grid gated on. Deliberately not
#: config.ML_THRESHOLDS: the grid searched around 0.5 on both datasets, and
#: moving it would change every number in Appendix G.3.
GRID_ML_THRESHOLD = 0.5

#: Gate sweeps, unchanged from the archived script: 100 points below the
#: decision threshold for the recall gate, 98 above it for the precision gate.
RECALL_TAUS = np.arange(0.001, 0.50, 0.005)
PRECISION_TAUS = np.arange(0.50, 0.99, 0.005)

#: Gate names as they appear in the ``strategy`` column. The archived CSVs
#: label these ``A_recall_boost`` and ``B_precision_boost``.
GATES = ("recall_gate", "precision_gate")


def _routing(ml_preds: np.ndarray) -> np.ndarray:
    """Per-edge routing labels, wide enough to hold every label.

    dtype is object rather than a fixed-width unicode type because the labels
    assigned later are longer than the two this array is initialised from.
    """
    return np.where(ml_preds == 1, "ml_illicit", "ml_legit").astype(object)


def recall_gate(
    ml_probs: np.ndarray,
    llm_preds: np.ndarray,
    tau: float,
    ml_threshold: float = GRID_ML_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Consult the LLM on ensemble-legit edges scoring at least ``tau``.

    The LLM can only add detections: an edge the ensemble already calls illicit
    is never revisited, so precision falls only through the flips this gate
    makes and recall is at least the ensemble's own.
    """
    ml_preds = (ml_probs >= ml_threshold).astype(int)
    preds = ml_preds.copy()
    routing = _routing(ml_preds)

    consulted = (ml_probs >= tau) & (ml_probs < ml_threshold)
    flipped = consulted & (llm_preds == 1)
    preds[flipped] = 1
    routing[consulted] = "llm_consulted"
    routing[flipped] = "llm_flipped"
    return preds, routing


def precision_gate(
    ml_probs: np.ndarray,
    llm_preds: np.ndarray,
    tau: float,
    ml_threshold: float = GRID_ML_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Consult the LLM on ensemble-illicit edges scoring below ``tau``.

    The mirror of :func:`recall_gate`: the LLM can only remove detections, so
    precision is at least the ensemble's own and recall can only fall.
    """
    ml_preds = (ml_probs >= ml_threshold).astype(int)
    preds = ml_preds.copy()
    routing = _routing(ml_preds)

    consulted = (ml_probs >= ml_threshold) & (ml_probs < tau)
    flipped = consulted & (llm_preds == 0)
    preds[flipped] = 0
    routing[consulted] = "llm_consulted"
    routing[flipped] = "llm_flipped"
    return preds, routing


_GATE_FN = {"recall_gate": recall_gate, "precision_gate": precision_gate}
_GATE_TAUS = {"recall_gate": RECALL_TAUS, "precision_gate": PRECISION_TAUS}


def confusion(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Raw confusion counts, reported alongside the weighted metrics."""
    return {
        "TP": int(((preds == 1) & (labels == 1)).sum()),
        "FP": int(((preds == 1) & (labels == 0)).sum()),
        "FN": int(((preds == 0) & (labels == 1)).sum()),
        "TN": int(((preds == 0) & (labels == 0)).sum()),
    }


def unweighted_prf(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Unweighted precision, recall, F1 over the coreset as drawn."""
    c = confusion(preds, labels)
    p = c["TP"] / (c["TP"] + c["FP"]) if (c["TP"] + c["FP"]) > 0 else 0.0
    r = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def grid_search(
    data: dict,
    llm_preds: np.ndarray,
    dataset: str,
    model: str,
    prompting: str,
    seed: int,
    ml_threshold: float = GRID_ML_THRESHOLD,
) -> pd.DataFrame:
    """Sweep both gates for one (dataset, model, prompting, seed) cell."""
    labels, weights = data["labels"], data["weights"]
    rows = []
    for gate in GATES:
        for tau in _GATE_TAUS[gate]:
            preds, routing = _GATE_FN[gate](data["ml_probs"], llm_preds, tau,
                                            ml_threshold)
            wp, wr, wf1 = ht_weighted_prf(preds, labels, weights)
            up, ur, uf1 = unweighted_prf(preds, labels)
            n_consulted = int(((routing == "llm_consulted")
                               | (routing == "llm_flipped")).sum())
            n_flipped = int((routing == "llm_flipped").sum())
            rows.append({
                "strategy": gate,
                "dataset": dataset,
                "llm_model": model,
                "prompting": prompting,
                "seed": seed,
                "tau": round(float(tau), 3),
                "n_llm_consulted": n_consulted,
                "n_llm_flipped": n_flipped,
                "llm_pct": round(n_consulted / len(preds) * 100, 1),
                "w_precision": round(wp, 4),
                "w_recall": round(wr, 4),
                "w_f1": round(wf1, 4),
                **confusion(preds, labels),
                "uw_precision": round(up, 4),
                "uw_recall": round(ur, 4),
                "uw_f1": round(uf1, 4),
            })
    return pd.DataFrame(rows)


def run_cell(
    archive: Path,
    data: dict,
    dataset: str,
    model: str,
    prompting: str,
    seed: int,
    ml_threshold: float = GRID_ML_THRESHOLD,
) -> Optional[dict]:
    """Grid search one cell and summarise the best gate of each kind.

    Returns ``None`` when the cell has no prediction file, and otherwise a dict
    holding the full grid plus the ensemble-only and LLM-only references.
    """
    loaded = load_llm_predictions(archive, dataset, model, prompting, seed, data["n"])
    if loaded is None:
        return None
    llm_preds, _, _ = loaded

    grid = grid_search(data, llm_preds, dataset, model, prompting, seed, ml_threshold)

    ml_only = (data["ml_probs"] >= ml_threshold).astype(int)
    ml_f1 = ht_weighted_prf(ml_only, data["labels"], data["weights"])[2]
    llm_f1 = ht_weighted_prf(llm_preds, data["labels"], data["weights"])[2]

    best = {}
    for gate in GATES:
        sub = grid[grid["strategy"] == gate]
        if not sub.empty:
            best[gate] = sub.loc[sub["w_f1"].idxmax()]

    return {"grid": grid, "best": best, "ml_f1": ml_f1, "llm_f1": llm_f1}


def summarise(cell: dict, dataset: str, model: str, prompting: str, seed: int) -> dict:
    """One summary row per cell: the best tau and F1 of each gate."""
    row = {
        "dataset": dataset, "llm_model": model, "prompting": prompting, "seed": seed,
        "ml_f1": round(cell["ml_f1"], 4), "llm_f1": round(cell["llm_f1"], 4),
    }
    for gate in GATES:
        b = cell["best"].get(gate)
        row[f"best_{gate}_f1"] = float(b["w_f1"]) if b is not None else None
        row[f"best_{gate}_tau"] = float(b["tau"]) if b is not None else None
        row[f"best_{gate}_delta"] = (round(float(b["w_f1"]) - cell["ml_f1"], 4)
                                     if b is not None else None)
        row[f"best_{gate}_consult_pct"] = float(b["llm_pct"]) if b is not None else None
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", type=Path, default=None,
                    help="archived experiment tree (default: $AMLC_ARCHIVE)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <repo>/results/triage)")
    ap.add_argument("--draw", choices=CORESET_DRAWS, default="ht-coreset")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the draw alignment check (needed for the re-draw)")
    ap.add_argument("--archived-weights", action="store_true",
                    help="use the archived HT weights rather than the repaired "
                         "ones; needed to reproduce the archived grid CSVs")
    ap.add_argument("--dataset", choices=config.DATASETS)
    ap.add_argument("--model", choices=config.LLM_MODELS)
    ap.add_argument("--prompting", choices=config.PROMPTINGS)
    ap.add_argument("--seed", type=int, default=config.SEEDS[0])
    ap.add_argument("--ml-threshold", type=float, default=GRID_ML_THRESHOLD,
                    help="ensemble decision threshold the gates sit around")
    args = ap.parse_args()

    from ..paths import ensure, results  # local: keeps the module import light

    archive = resolve_archive(args.archive)
    out_dir = ensure(Path(args.out) if args.out else results() / "triage")

    datasets = [args.dataset] if args.dataset else list(config.DATASETS)
    models = [args.model] if args.model else list(config.LLM_MODELS)
    # ICL-V was run on two models at seed 42 only, so it is opt-in.
    promptings = [args.prompting] if args.prompting else ["ICL-FS", "ICL-ZS"]

    summary = []
    for dataset in datasets:
        data = load_coreset_from_archive(archive, dataset, draw=args.draw,
                                         verify=not args.no_verify,
                                         repair_weights=not args.archived_weights)
        for model in models:
            for prompting in promptings:
                cell = run_cell(archive, data, dataset, model, prompting,
                                args.seed, args.ml_threshold)
                if cell is None:
                    print(f"  {dataset:9s} | {model:24s} | {prompting:7s} | no predictions")
                    continue
                name = f"score_deferral_grid_{dataset}_{model}_{prompting}.csv"
                cell["grid"].to_csv(out_dir / name, index=False)
                row = summarise(cell, dataset, model, prompting, args.seed)
                summary.append(row)
                print(f"  {dataset:9s} | {model:24s} | {prompting:7s} | "
                      f"ensemble={row['ml_f1']*100:5.1f} | "
                      f"recall gate={row['best_recall_gate_f1']*100:5.1f} "
                      f"@tau={row['best_recall_gate_tau']:.3f} "
                      f"({row['best_recall_gate_consult_pct']:.1f}% consulted) | "
                      f"precision gate={row['best_precision_gate_f1']*100:5.1f} "
                      f"@tau={row['best_precision_gate_tau']:.3f} "
                      f"({row['best_precision_gate_consult_pct']:.1f}% consulted)")

    if not summary:
        print("No results. Check that the LLM prediction files exist.")
        return
    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "score_deferral_summary.csv", index=False)
    print(f"\nSaved {out_dir / 'score_deferral_summary.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
