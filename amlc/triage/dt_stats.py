"""Paired Doubt-Triage-versus-ML statistics: the pooled test and the per-cell table.

This is the generator that did not exist. Section 5 quotes a pooled bootstrap mean,
a 95 percent interval, a Wilcoxon p and a Cliff's delta for the DT lift, and the
appendix carries a per-cell table and a Friedman test behind them. No script in any
repository produced those numbers, so when the coreset draw was repaired there was
no way to recompute them and they were the last figures in the paper still resting
on the old draw.

Everything here is computed from the per-seed table that
:mod:`amlc.triage.doubt_triage` writes, so the draw is whatever that run used and is
recorded in its ``draw`` column rather than assumed here.

The unit of pairing
-------------------
One pair is one (dataset, model, prompting, seed) cell: the DT weighted F1 against
the ML ensemble weighted F1 that the same cell was scored against. With 7 models,
2 datasets, 2 prompting variants and 5 seeds that is 140 pairs, which is the
population the pooled statistics describe and the one Section 5 names.

Per-cell rows aggregate the 5 seeds of a single (dataset, model, prompting). Their
Wilcoxon p bottoms out at 0.03125, which is 2^-5 and the smallest one-sided value
five paired observations can produce; it means every seed moved the same way and
nothing more. The pooled test is the one with power.

Cliff's delta is computed between the DT and ML samples rather than on the paired
differences, so a value of 1 says every DT observation exceeds every ML observation.

Reads
    the per-seed DT results CSV (``--results``).
Writes
    ``dt_vs_ml_stats.csv``   one row per (dataset, model, prompting)
    ``dt_friedman.csv``      one row per (dataset, prompting, exclude-outlier)
    ``dt_pooled_stats.json`` the pooled figures Section 5 quotes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .. import paths

#: The strategy column value that is Doubt Triage. The other three are the
#: naive hybrids of the same run and are not what Section 5 reports.
DT_STRATEGY = "dt"

#: The model the paper treats as an engineering outlier. Friedman is reported
#: with and without it because its spread dominates the statistic otherwise.
OUTLIER_MODEL = "Qwen3.5-27B"

BOOT_SEED = 0
BOOT_N = 10_000


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta of ``a`` over ``b``: 1 when every a exceeds every b."""
    gt = int((a[:, None] > b[None, :]).sum())
    lt = int((a[:, None] < b[None, :]).sum())
    return (gt - lt) / (a.size * b.size)


def boot_ci(diffs: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean difference."""
    if np.ptp(diffs) == 0:
        return float(diffs[0]), float(diffs[0])
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, diffs.size, size=(BOOT_N, diffs.size))
    means = diffs[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def wilcoxon_greater(dt: np.ndarray, ml: np.ndarray) -> float:
    """One-sided signed-rank p for DT exceeding ML, NaN when every pair ties."""
    if np.all(dt == ml):
        return float("nan")
    return float(stats.wilcoxon(dt, ml, alternative="greater",
                                zero_method="zsplit").pvalue)


def wilcoxon_two_sided(dt: np.ndarray, ml: np.ndarray) -> float:
    """Two-sided signed-rank p, which is the convention Section 5 prints.

    Kept alongside the one-sided value because the published sentence quotes
    $1.0 \\times 10^{-24}$, which is twice the one-sided figure. Reporting both
    makes the convention visible instead of leaving a reader to infer it from a
    factor of two.
    """
    if np.all(dt == ml):
        return float("nan")
    return float(stats.wilcoxon(dt, ml, alternative="two-sided",
                                zero_method="zsplit").pvalue)


def per_cell(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, prompting), aggregating over seeds."""
    rows = []
    keys = ["dataset", "llm_model", "prompting"]
    for (dataset, model, prompting), grp in frame.groupby(keys, sort=True):
        grp = grp.sort_values("seed")
        dt, ml = grp.dt_f1.to_numpy(), grp.ml_f1.to_numpy()
        diffs = dt - ml
        lo, hi = boot_ci(diffs)
        rows.append({
            "dataset": dataset, "model": model, "method": prompting,
            "n_seeds": len(grp),
            "ml_f1_mean": ml.mean(), "dt_f1_mean": dt.mean(),
            "diff_mean": diffs.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
            "wilcoxon_p": wilcoxon_greater(dt, ml),
            "cliffs_delta": cliffs_delta(dt, ml),
        })
    return pd.DataFrame(rows)


def pooled(frame: pd.DataFrame) -> dict:
    """The pooled figures Section 5 quotes, over every paired cell at once."""
    dt, ml = frame.dt_f1.to_numpy(), frame.ml_f1.to_numpy()
    diffs = dt - ml
    lo, hi = boot_ci(diffs)
    return {
        "n_pairs": int(diffs.size),
        "n_dt_exceeds_ml": int((diffs > 0).sum()),
        "mean_pp": float(diffs.mean() * 100),
        "ci_lo_pp": lo * 100, "ci_hi_pp": hi * 100,
        "wilcoxon_p_two_sided": wilcoxon_two_sided(dt, ml),
        "wilcoxon_p_one_sided": wilcoxon_greater(dt, ml),
        "cliffs_delta": cliffs_delta(dt, ml),
    }


def pooled_by_dataset(frame: pd.DataFrame) -> dict:
    """The same pooled figures split by dataset, which Appendix G.2 reports."""
    return {ds: pooled(grp) for ds, grp in frame.groupby("dataset", sort=True)}


def friedman(frame: pd.DataFrame) -> pd.DataFrame:
    """Across-model agreement of the DT score, blocked on seed."""
    rows = []
    for (dataset, prompting), grp in frame.groupby(["dataset", "prompting"],
                                                   sort=True):
        for drop in (False, True):
            sub = grp[grp.llm_model != OUTLIER_MODEL] if drop else grp
            wide = sub.pivot_table(index="seed", columns="llm_model",
                                   values="dt_f1")
            if wide.shape[1] < 3:
                continue
            chi2, p = stats.friedmanchisquare(*[wide[c].to_numpy()
                                                for c in wide.columns])
            spread = wide.max(axis=1) - wide.min(axis=1)
            rows.append({
                "dataset": dataset, "method": prompting,
                "exclude_outlier": drop,
                "n_models": wide.shape[1], "n_seeds": wide.shape[0],
                "friedman_chi2": chi2, "friedman_p": p,
                "spread_max": float(spread.max()),
                "spread_mean": float(spread.mean()),
            })
    return pd.DataFrame(rows)


def run(results: Path, out: Path) -> dict:
    frame = pd.read_csv(results)
    frame = frame[frame.strategy == DT_STRATEGY].copy()
    if frame.empty:
        raise SystemExit(f"{results}: no rows with strategy={DT_STRATEGY!r}")

    out.mkdir(parents=True, exist_ok=True)
    cells = per_cell(frame)
    cells.to_csv(out / "dt_vs_ml_stats.csv", index=False)
    fried = friedman(frame)
    fried.to_csv(out / "dt_friedman.csv", index=False)
    summary = pooled(frame)
    summary["by_dataset"] = pooled_by_dataset(frame)
    draws = sorted(frame.draw.unique())
    summary["draw"] = draws[0] if len(draws) == 1 else draws
    (out / "dt_pooled_stats.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  draw: {summary['draw']}")
    print(f"  pooled over {summary['n_pairs']} paired cells "
          f"({summary['n_dt_exceeds_ml']} with DT above ML)")
    print(f"    mean lift   {summary['mean_pp']:+.2f} pp"
          f"  95% CI [{summary['ci_lo_pp']:+.2f}, {summary['ci_hi_pp']:+.2f}]")
    print(f"    Wilcoxon    p = {summary['wilcoxon_p_two_sided']:.3g}"
          f" two-sided, {summary['wilcoxon_p_one_sided']:.3g} one-sided")
    print(f"    Cliff's     delta = {summary['cliffs_delta']:.4f}")
    for ds, s in summary["by_dataset"].items():
        print(f"  {ds:10s} {s['mean_pp']:+.2f} pp"
              f"  95% CI [{s['ci_lo_pp']:+.2f}, {s['ci_hi_pp']:+.2f}]"
              f"  p = {s['wilcoxon_p_two_sided']:.3g}")
    print(f"\n  wrote {out}/dt_vs_ml_stats.csv ({len(cells)} rows), "
          f"dt_friedman.csv ({len(fried)} rows), dt_pooled_stats.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=None,
                    help="per-seed DT results CSV from stage 17 "
                         "(default: <results>/triage/dt_results.csv)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <results>/triage)")
    args = ap.parse_args()
    default_dir = paths.results() / "triage"
    run(args.results or default_dir / "dt_results.csv", args.out or default_dir)


if __name__ == "__main__":
    main()
