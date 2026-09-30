#!/usr/bin/env python3
"""Export manuscript tables from the cached sampling and audit analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLERS = {"uniform_ht": "Uniform + HT", "difficulty_ht": "Terciles + HT",
            "hard_terciles_ht": "Hard + terciles + HT"}
FAMILIES = {"LightGBM+GFP": "LGB", "XGBoost+GFP": "XGB"}
ROW = r" \\" + "\n"


def sampling_tables(summary):
    f1 = summary[summary.metric == "f1"]
    main = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lrrr}", r"\toprule",
            r"\textbf{Sampler} & \textbf{RMSE} & \textbf{Width} & \textbf{Cov.}" + ROW.rstrip(),
            r"\midrule"]
    for (dataset, constructor, evaluator), block in f1[f1.benign_multiplier == 2].groupby(
            ["dataset", "constructor", "evaluator"], sort=False):
        label = f"{dataset}: {FAMILIES[constructor]} $\\to$ {FAMILIES[evaluator]}"
        main.append(r"\multicolumn{4}{l}{\textit{" + label + "}}" + ROW.rstrip())
        for sampler, name in SAMPLERS.items():
            row = block[block.sampler == sampler].iloc[0]
            main.append(f"{name} & {100*row.rmse:.2f} & {100*row.mean_width95:.1f} & "
                        f"{100*row.coverage95:.1f}" + ROW.rstrip())
        main.append(r"\addlinespace[2pt]")
    main.extend([r"\bottomrule\end{tabular}", r"\caption{Sampling with a held-out predictor family at the released target counts.",
                 r"LGB=LightGBM; XGB=XGBoost. The arrow runs from the stratum constructor to the evaluated family.",
                 r"All samplers retain every illicit edge and use HT weights. RMSE and mean $95\%$ interval",
                 r"width concern F1 and are in percentage points; coverage (Cov.) is a percentage over $500$ draws.",
                 r"Each draw estimates the mean of five fixed, validation-thresholded seed metrics.",
                 r"Intervals are conservative and can remain broad despite small point errors.}",
                 r"\label{tab:heldout_sampling}", r"\end{table}"])
    extra = [r"\begin{table*}[!ht]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"\textbf{Sampler} & \textbf{P RMSE} & \textbf{F1 bias} & \textbf{F1 RMSE} & \textbf{F1 width} & \textbf{Cov.}" + ROW.rstrip(),
             r"\midrule"]
    keys = ["dataset", "constructor", "evaluator", "benign_multiplier"]
    for key, block in f1[f1.benign_multiplier > 2].groupby(keys, sort=False):
        dataset, constructor, evaluator, multiplier = key
        label = (f"{dataset}: {FAMILIES[constructor]} $\\to$ {FAMILIES[evaluator]}, "
                 f"benign budget ${multiplier}\\times$ illicit count")
        extra.append(r"\multicolumn{6}{l}{\textit{" + label + "}}" + ROW.rstrip())
        for sampler, name in SAMPLERS.items():
            row = block[block.sampler == sampler].iloc[0]
            mask = (summary.metric == "precision") & (summary.sampler == sampler)
            for col, value in zip(keys, key):
                mask &= summary[col] == value
            precision = summary[mask].iloc[0]
            extra.append(f"{name} & {100*precision.rmse:.2f} & {100*row.signed_bias:+.2f} & "
                         f"{100*row.rmse:.2f} & {100*row.mean_width95:.1f} & "
                         f"{100*row.coverage95:.1f}" + ROW.rstrip())
        extra.append(r"\addlinespace[2pt]")
    extra.extend([r"\bottomrule\end{tabular}",
                  r"\caption{Larger budgets in the held-out-family study ($500$ draws per configuration).",
                  r"Bias is mean estimated F1 minus exact full-test F1. RMSE, bias, and mean $95\%$ interval",
                  r"width are in percentage points; coverage is a percentage. Every quantity concerns the",
                  r"mean of the five fixed seed metrics. The primary $1{:}2$ budgets appear in",
                  r"Table~\ref{tab:heldout_sampling}; all budgets and per-seed results are released as CSVs.}",
                  r"\label{tab:heldout_sampling_budgets}", r"\end{table*}"])
    return {"heldout_sampling_main.tex": "\n".join(main) + "\n",
            "heldout_sampling_budgets.tex": "\n".join(extra) + "\n"}


def audit_tables(items, conjunctions):
    main = [r"\begin{table}[!ht]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lrrr}", r"\toprule",
            r"\textbf{Binary verdict} & $n$ & \textbf{Full trace} & \textbf{First 8K}" + ROW.rstrip(),
            r"\midrule"]
    for correct, name in [(True, "Correct"), (False, "Incorrect")]:
        cohort = items[items.binary_correct == correct]
        values = []
        for prefix in ["full", "prefix8000"]:
            count = int(cohort[[f"{prefix}_parse", f"{prefix}_recall",
                                f"{prefix}_marker_ge2"]].astype(bool).all(axis=1).sum())
            values.append(f"{count} ({100*count/len(cohort):.1f}\\%)")
        main.append(f"{name} & {len(cohort)} & {values[0]} & {values[1]}" + ROW.rstrip())
    main.extend([r"\bottomrule\end{tabular}",
                 r"\caption{Textual checks by binary correctness in the $1{,}000$-trace audit.",
                 r"Counts require Parse, Recall, and at least two structural-marker categories for every",
                 r"verdict, removing the original benign-verdict Match bypass. Percentages use the row",
                 r"denominator. The 8K-character window matches the API judges' text limit. These are",
                 r"outcome-stratified descriptive rates, not population estimates.}",
                 r"\label{tab:audit_sensitivity}", r"\end{table}"])
    appendix = [r"\begin{table}[!ht]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
                r"\begin{tabular}{llrr}", r"\toprule",
                r"\textbf{Checks} & \textbf{References} & \textbf{Full} & \textbf{First 8K}" + ROW.rstrip(),
                r"\midrule"]
    for rule, name in [("original_PRM", "Original PRM"),
                       ("PR_and_two_marker_categories", "No bypass")]:
        for correctness, reference in [("strict_archived", "All"),
                                        ("exclude_illicit_reference_absent", "Available")]:
            cells = []
            for window in ["full", "prefix8000"]:
                row = conjunctions[(conjunctions.model == "ALL") &
                                   (conjunctions.group_kind == "all") &
                                   (conjunctions.rule == rule) &
                                   (conjunctions.correctness == correctness) &
                                   (conjunctions.window == window)].iloc[0]
                cells.append(f"{int(row['count'])}/{int(row.denominator)}")
            appendix.append(f"{name} & {reference} & {cells[0]} & {cells[1]}" + ROW.rstrip())
    appendix.extend([r"\bottomrule\end{tabular}",
                     r"\caption{Traces passing the specified textual checks but failing the stated answer",
                     r"checks. All retains archived Conclude failures, including missing-reference cases;",
                     r"Available-reference rows exclude all $240$ illicit cases without a",
                     r"reference typology; benign cases remain evaluable. No-bypass rows require two",
                     r"structural-marker categories even for benign verdicts. Original labels and scores",
                     r"remain unchanged; each row applies a stated sensitivity rule to the same archived outputs.}",
                     r"\label{tab:audit_sensitivity_rules}", r"\end{table}"])
    return {"audit_sensitivity_main.tex": "\n".join(main) + "\n",
            "audit_sensitivity_rules.tex": "\n".join(appendix) + "\n"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "results/analysis")
    parser.add_argument("--out", type=Path, default=ROOT / "results/tables")
    args = parser.parse_args()
    tables = sampling_tables(pd.read_csv(args.analysis / "heldout_sampling/summary.csv"))
    tables.update(audit_tables(pd.read_csv(args.analysis / "audit_sensitivity/items.csv"),
                              pd.read_csv(args.analysis / "audit_sensitivity/conjunctions.csv")))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, text in tables.items():
        (args.out / name).write_text("% Generated by scripts/29_export_cached_evidence_tables.py\n" + text)
        print(args.out / name)


if __name__ == "__main__":
    main()
