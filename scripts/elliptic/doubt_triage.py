#!/usr/bin/env python3
"""Doubt Triage evaluation on Elliptic.

Replicates the AMLworld §6 hybrid (ML alone vs LLM alone vs Doubt Triage) on the
Elliptic test split. For each case in the HT-Coreset we already have:

  - p_ml = mean ensemble probability across 5 ML seeds
    (loaded from outputs/elliptic/ml/ensemble_seed*.npz)
  - y_llm ∈ {0, 1} = LLM verdict
    (one prediction per case_id, taken from `--llm-preds` JSON whose schema
     matches outputs/{Model}/{Method}/HI-Small/seed_*.json — i.e. each entry
     has `case_id` and `pred_illicit`)

Doubt Triage rule: predict illicit if
    p_ml ≥ τ_high                                 (ML census positive)
 OR (τ_low ≤ p_ml < τ_high  AND  y_llm = 1)       (LLM consults sampled stratum)

τ_low and τ_high are calibrated on the train split so that
  (i) the LLM is consulted on the stratum where it is most useful and
  (ii) the LLM's vote can only flip a borderline ML decision.

This script reports P/R/F1 for ML-alone, LLM-alone, OR/AND fusion baselines,
threshold-gated (LLM-only on cases between τ_low and τ_high), and Doubt Triage,
under HT importance weights `w_i` from `v2.npz`.

Outputs: outputs/elliptic/doubt_triage/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR  = REPO_ROOT / "data" / "elliptic" / "proc"
ML_DIR    = REPO_ROOT / "outputs" / "elliptic" / "ml"
CORE_DIR  = REPO_ROOT / "outputs" / "elliptic" / "coreset"
OUT_DIR   = REPO_ROOT / "outputs" / "elliptic" / "doubt_triage"


def weighted_metrics(y_true, y_pred, w) -> dict:
    tp = float(w[(y_true == 1) & (y_pred == 1)].sum())
    fp = float(w[(y_true == 0) & (y_pred == 1)].sum())
    fn = float(w[(y_true == 1) & (y_pred == 0)].sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn)


def load_llm_predictions(paths: list[Path]) -> dict[str, int]:
    """Map case_id → 1/0 from the union of supplied prediction JSONs."""
    out: dict[str, int] = {}
    for p in paths:
        d = json.loads(p.read_text())
        for rec in d.get("predictions", []):
            cid = rec.get("case_id"); pred = rec.get("pred_illicit")
            if cid is None or pred is None: continue
            out[str(cid)] = int(bool(pred))
    return out


def calibrate_strata(p_ml: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pick τ_low, τ_high to bracket the borderline stratum.

    Heuristic (matches the Doubt Triage calibration used in §6.2):
      τ_low  = max F1 threshold − 0.10
      τ_high = max F1 threshold + 0.10
    capped to [0.05, 0.95].
    """
    best_f, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y, (p_ml >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return float(np.clip(best_t - 0.10, 0.05, 0.95)), \
           float(np.clip(best_t + 0.10, 0.05, 0.95))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--ml-dir",   type=Path, default=ML_DIR)
    ap.add_argument("--core-dir", type=Path, default=CORE_DIR)
    ap.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--llm-preds", type=Path, nargs="+", required=True,
                    help="one or more LLM prediction JSONs (case_id + pred_illicit)")
    ap.add_argument("--llm-name",  type=str, default="LLM")
    ap.add_argument("--tau-low",   type=float, default=None)
    ap.add_argument("--tau-high",  type=float, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    test = np.load(args.proc_dir / "test.npz")
    v2   = np.load(args.core_dir / "v2.npz")
    y    = test["y"].astype(np.int32)[v2["idx"]]
    p_ml = v2["ensemble_probs"][v2["idx"]]
    w    = v2["weights"]
    thr  = float(v2["threshold"])
    case_ids = [f"elliptic_test_{int(i)}" for i in v2["idx"]]
    print(f"[doubt-triage] HT-Coreset n={len(y)}  illicit={int((y==1).sum())}  "
          f"ML thr={thr:.3f}")

    llm_map = load_llm_predictions(args.llm_preds)
    have_llm = np.array([cid in llm_map for cid in case_ids])
    if have_llm.sum() == 0:
        raise SystemExit("[doubt-triage] no LLM predictions matched any case_id; "
                         "check that --llm-preds points at JSONs produced "
                         "from the HT-Coreset.")
    y_llm = np.array([llm_map.get(cid, 0) for cid in case_ids], dtype=int)
    print(f"[doubt-triage] LLM coverage: {have_llm.sum()}/{len(have_llm)} cases")

    # Stratum band, when not given on the command line: bracket the ML decision
    # threshold by +/-0.10, clamped into [0.05, 0.95].
    #
    # This used to read as a calibration step. It loaded train.npz, globbed the
    # per-seed ensemble files and opened an empty `train_probs` list, then
    # ignored all three and bracketed `thr` anyway, because the saved npz holds
    # only test probabilities. The load and the glob are gone; the arithmetic is
    # what always ran. The clamp now applies unconditionally, where before the
    # no-seeds branch skipped it and could hand back a negative tau_low.
    if args.tau_low is None or args.tau_high is None:
        args.tau_low  = max(0.05, thr - 0.10)
        args.tau_high = min(0.95, thr + 0.10)
    print(f"[doubt-triage] τ_low={args.tau_low:.3f}  τ_high={args.tau_high:.3f}")

    # ── Predictions for each rule ────────────────────────────────────────
    yhat_ml  = (p_ml >= thr).astype(int)
    yhat_llm = y_llm
    yhat_or  = ((yhat_ml | yhat_llm)).astype(int)
    yhat_and = ((yhat_ml & yhat_llm)).astype(int)
    in_band  = (p_ml >= args.tau_low) & (p_ml < args.tau_high)
    yhat_thresh = np.where(in_band, yhat_llm, yhat_ml)
    yhat_dt   = ((p_ml >= args.tau_high) | (in_band & (yhat_llm == 1))).astype(int)

    rows = []
    for name, pred in [
        ("ML_alone",  yhat_ml),
        (f"{args.llm_name}_alone", yhat_llm),
        ("OR",        yhat_or),
        ("AND",       yhat_and),
        ("ThreshGated", yhat_thresh),
        ("Doubt Triage",      yhat_dt),
    ]:
        m = weighted_metrics(y, pred, w)
        consult = float((in_band).sum() / len(in_band)) if name == "Doubt Triage" else 0.0
        rows.append(dict(
            method=name,
            precision=m["precision"], recall=m["recall"], f1=m["f1"],
            tp=m["tp"], fp=m["fp"], fn=m["fn"],
            consult_pct=consult,
        ))
        print(f"  {name:<14s}  P={m['precision']*100:5.2f}  "
              f"R={m['recall']*100:5.2f}  F1={m['f1']*100:5.2f}"
              + (f"  consult={consult*100:.1f}%" if name == "Doubt Triage" else ""))

    out_csv = args.out_dir / f"results_{args.llm_name}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            for k, v in r.items():
                if isinstance(v, float):
                    r[k] = f"{v:.6f}"
            writer.writerow(r)
    print(f"\n[doubt-triage] wrote {out_csv}")


if __name__ == "__main__":
    main()
