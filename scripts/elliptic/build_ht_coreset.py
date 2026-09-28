#!/usr/bin/env python3
"""HT-Coreset + 7 ablation baselines on the Elliptic test split.

Mirrors the AMLworld HT-Coreset construction (deterministic minority retention
+ hard-negative oversampling + difficulty-stratified random sample), reweighted
by Horvitz-Thompson inverse inclusion probability.

Selection budget:
  - All illicit nodes are kept (recall is exact).
  - The benign budget is `--benign-multiplier × n_illicit` (default 2:1, the
    same ratio as the AMLworld HT-Coreset release: 2502:1251 on HI-Small).

Outputs (all under `outputs/elliptic/coreset/`):
  v2.npz        idx, weights, draw=0 (deterministic)
  B{1..7}.npz   one .npz per baseline, k draws each (default k=50)
  meta.json     full-set metrics, subset sizes, hyperparameters

The 7 ablation baselines use the following Elliptic rules. B3 differs from
the AMLworld B3 sampler in the paper's main comparison:

  B1  Random Uniform                   benign+illicit drawn iid, no IW
  B2  Stratified Proportional          stratified by time_step, no IW
  B3  HT selection with unit weights   retain the HT-selected nodes, no IW
  B4  HT-Coreset w/o Hard-Neg          exclude hard nodes, weight the rest
  B5  Fogliato (Neyman allocation)     proportional minority sampling + IW
  B6  Leskovec random walk             walk-based traversal from illicit nodes
  B7  Gao stratified graph sampling    stratified by node degree

Eval is performed in `eval_ht_coreset.py` (separate module).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR  = REPO_ROOT / "data" / "elliptic" / "proc"
ML_DIR    = REPO_ROOT / "outputs" / "elliptic" / "ml"
OUT_DIR   = REPO_ROOT / "outputs" / "elliptic" / "coreset"


# ─────────────────────────────────────────────────────────────────────────────
#  Difficulty score = ensemble probability averaged across seeds
# ─────────────────────────────────────────────────────────────────────────────

def load_ensemble_probs(ml_dir: Path, seeds: list[int]) -> tuple[np.ndarray, float]:
    """Average ensemble probabilities across seeds; return mean threshold too."""
    probs, thrs = [], []
    for s in seeds:
        path = ml_dir / f"ensemble_seed{s}.npz"
        if not path.exists():
            print(f"  [coreset] WARN: missing {path}; skipping seed {s}")
            continue
        d = np.load(path)
        probs.append(d["ensemble"])
        thrs.append(float(d["threshold"]))
    if not probs:
        raise FileNotFoundError(f"no ML predictions in {ml_dir}; "
                                "run train_ml_baselines.py first")
    mean_probs = np.mean(np.stack(probs, axis=0), axis=0)
    mean_thr   = float(np.mean(thrs))
    return mean_probs, mean_thr


# ─────────────────────────────────────────────────────────────────────────────
#  Selection strategies
# ─────────────────────────────────────────────────────────────────────────────

def ht_select(rng: np.random.Generator, ben_pool: np.ndarray,
              probs: np.ndarray, threshold: float, time_step: np.ndarray,
              n_select: int, hard_neg_ratio: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Stratified hard-negative + difficulty-tertile sampling with HT weights.

    Weights are assigned once, at the end, as ``N_s / n_s`` over the realised
    sample, so ``sum(w)`` equals the benign population exactly.

    Fixed 2026-08-10. The previous version computed each stratum's weight at
    the moment that stratum was drawn and then appended a spare-fill block
    weighted ``len(spare) / len(extra)``. By then every benign edge was already
    represented by the four stratum blocks, so the fill added a second copy of
    almost the whole population: the Elliptic coreset summed to 30,092 against
    a population of 16,670 (1.805x), with a single edge carrying 13,422 where
    the next largest benign weight is three figures. The identical defect sits
    in the AMLworld generator; see amlc/coreset/ht_weights.py, which
    repairs already-drawn coresets by the same recompute-from-membership rule.
    """
    p_ben = probs[ben_pool]
    hard_mask = p_ben >= (threshold * 0.5)
    hard_pool, easy_pool = ben_pool[hard_mask], ben_pool[~hard_mask]

    n_hard_t, n_easy_t = len(hard_pool), len(easy_pool)
    n_hard_b = min(int(n_select * hard_neg_ratio), n_hard_t)
    n_easy_b = min(n_select - n_hard_b, n_easy_t)
    if n_easy_b < n_select - n_hard_b and n_hard_t > n_hard_b:
        n_hard_b = min(n_select - n_easy_b, n_hard_t)

    # (population pool, chosen members) per stratum. Weights come later.
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    if n_hard_b > 0:
        blocks.append((hard_pool, rng.choice(hard_pool, n_hard_b, replace=False)))

    if n_easy_b > 0 and n_easy_t > 0:
        diff_easy = probs[easy_pool]
        q33, q67 = np.percentile(diff_easy, [33, 67])
        tertiles = [diff_easy < q33,
                    (diff_easy >= q33) & (diff_easy < q67),
                    diff_easy >= q67]
        n_per = max(1, n_easy_b // 3)
        for mask in tertiles:
            pool = easy_pool[mask]
            if len(pool) == 0:
                continue
            n_take = min(n_per, len(pool))
            blocks.append((pool, rng.choice(pool, n_take, replace=False)))

    if not blocks:
        return np.array([], dtype=int), np.array([], dtype=float)

    n_drawn = sum(len(c) for _, c in blocks)
    if n_drawn > n_select:
        # Trim from the back, keeping the block structure consistent.
        excess = n_drawn - n_select
        trimmed = []
        for pool, chosen in reversed(blocks):
            if excess and len(chosen) > 1:
                cut = min(excess, len(chosen) - 1)
                chosen = chosen[:len(chosen) - cut]
                excess -= cut
            trimmed.append((pool, chosen))
        blocks = list(reversed(trimmed))
    elif n_drawn < n_select:
        # Top up from whatever is left, then let the weighting below count each
        # extra edge inside the stratum it actually belongs to.
        used = np.zeros(len(probs), dtype=bool)
        for _, chosen in blocks:
            used[chosen] = True
        spare = ben_pool[~used[ben_pool]]
        if len(spare):
            extra = rng.choice(spare, min(n_select - n_drawn, len(spare)),
                               replace=False)
            merged = []
            for pool, chosen in blocks:
                add = extra[np.isin(extra, pool)]
                merged.append((pool, np.concatenate([chosen, add]) if len(add) else chosen))
            blocks = merged

    sel = np.concatenate([c for _, c in blocks])
    w = np.concatenate([np.full(len(c), len(pool) / len(c)) for pool, c in blocks])
    return sel, w


def b1_random(rng, pool, probs, threshold, time_step, n_select, **_):
    """Uniform draw over the WHOLE test set, no importance weighting.

    Mirrors ``select_B1_random_uniform`` in the AMLworld ablation, which is
    what the shared table label "B1 Random" means: no stratification, no
    minority retention, no weighting. The point of the baseline is that
    uniform sampling captures the minority class only in proportion to its
    prevalence.

    Fixed 2026-08-10. The pre-release version drew from the benign pool only,
    so every subset contained zero illicit nodes, precision and recall
    collapsed to 0, and the reported bias was the full-set metric itself
    (|dP| 89.99, |dR| 73.04, |dF1| 80.63 with sigma exactly 0 across 50 draws,
    which is the tell: 50 random draws cannot have zero variance unless the
    metric is degenerate).
    """
    if n_select >= len(pool):
        return pool, np.ones(len(pool))
    sel = rng.choice(pool, n_select, replace=False)
    return sel, np.ones(n_select, dtype=float)   # no IW, as in AMLworld


def b2_stratified(rng, pool, probs, threshold, time_step, n_select, y=None, **_):
    """Stratified by CLASS, proportional allocation, no importance weighting.

    Mirrors ``select_B2_stratified_proportional`` in the AMLworld ablation so
    the shared table label means the same thing on both datasets: proportional
    allocation preserves the class ratio, so the subset holds the minority
    class only at its population rate and never retains it deterministically.

    Fixed 2026-08-10 together with B1. The pre-release version stratified by
    time step over the benign pool only, which is both a different stratifier
    and the zero-illicit bug.
    """
    if y is None:
        raise ValueError("b2_stratified needs the label vector")
    ill_idx = pool[y[pool] == 1]
    ben_idx = pool[y[pool] == 0]
    n_tot = len(pool)
    n_ill = min(max(1, round(n_select * len(ill_idx) / n_tot)), len(ill_idx))
    n_ben = min(n_select - n_ill, len(ben_idx))
    sel = np.sort(np.concatenate([
        rng.choice(ill_idx, n_ill, replace=False),
        rng.choice(ben_idx, n_ben, replace=False),
    ]))
    return sel, np.ones(len(sel), dtype=float)   # no IW, as in AMLworld


def b2t_stratified_time(rng, ben_pool, probs, threshold, time_step, n_select, **_):
    """Time-step stratified, proportional, HT-weighted, benign pool only.

    The pre-release ``B2_stratified`` with its zero-illicit bug left in place
    is not a meaningful baseline, but time-step stratification itself is worth
    reporting on a temporal graph. This keeps that variant under an honest
    name, drawing from the benign pool by design and stating so.
    """
    ts = time_step[ben_pool]
    parts, ws = [], []
    uniq, counts = np.unique(ts, return_counts=True)
    total = counts.sum()
    for u, c in zip(uniq, counts):
        n_take = max(1, int(round(n_select * c / total)))
        pool = ben_pool[ts == u]
        n_take = min(n_take, len(pool))
        if n_take == 0: continue
        chosen = rng.choice(pool, n_take, replace=False)
        parts.append(chosen); ws.append(np.full(n_take, c / n_take))
    if not parts:
        return ben_pool[:0], np.array([], dtype=float)
    sel = np.concatenate(parts); w = np.concatenate(ws)
    return sel[:n_select], w[:n_select]


def b3_no_iw(rng, ben_pool, probs, threshold, time_step, n_select, **_):
    """Same as the HT-Coreset but with weights forced to 1 (illustrates the IW correction)."""
    sel, _ = ht_select(rng, ben_pool, probs, threshold, time_step, n_select)
    return sel, np.ones(len(sel))


def b4_no_hardneg(rng, ben_pool, probs, threshold, time_step, n_select, **_):
    """The HT-Coreset minus the hard-negative oversampling stage (hard_neg_ratio = 0)."""
    return ht_select(rng, ben_pool, probs, threshold, time_step, n_select,
                     hard_neg_ratio=0.0)


def precompute_fogliato_strata(probs, y, n_strata: int = 10):
    """K-means strata over the ensemble score, with Neyman allocation inputs.

    Computed once rather than per draw. Mirrors the AMLworld implementation so
    the shared "B5 Fogliato" label means the same method on both datasets.
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_strata, random_state=42, n_init=10)
    strata = km.fit_predict(probs.reshape(-1, 1))
    correct = ((probs >= 0.5).astype(float) == y).astype(float)

    ids = np.unique(strata)
    N_h, S_h, idx_h = {}, {}, {}
    for h in ids:
        mask = strata == h
        idx_h[h] = np.where(mask)[0]
        N_h[h] = int(mask.sum())
        S_h[h] = max(float(correct[mask].std()), 1e-8)
    return dict(stratum_ids=ids, N_h=N_h, S_h=S_h, idx_h=idx_h)


def b5_fogliato(rng, ben_pool, probs, threshold, time_step, n_select,
                ill_pool=None, strata=None, **_):
    """Fogliato et al. (ECCV 2024): k-means strata + Neyman optimal allocation.

    n_h proportional to N_h * S_h, where S_h is the within-stratum standard
    deviation of prediction correctness, with Horvitz-Thompson weights
    N_h / n_h. Crucially it does NOT deterministically retain the minority
    class, which is the property the ablation is testing.

    Fixed 2026-08-10. The pre-release version allocated
    ``n_ill = round(n_subset * N_ill / N_total)``, i.e. plain class-proportional
    allocation over two strata. That is not Neyman allocation, it is precisely
    what B2 is defined to be, and under the shared per-draw random stream the
    two baselines selected byte-identical samples in all 50 draws, so the table
    carried the same estimator twice under two names, one of them citing a
    method it did not implement.
    """
    if strata is None:
        raise ValueError("b5_fogliato needs precomputed strata")
    ids, N_h, S_h, idx_h = (strata["stratum_ids"], strata["N_h"],
                            strata["S_h"], strata["idx_h"])
    n_subset = n_select + len(ill_pool)

    total = sum(N_h[h] * S_h[h] for h in ids)
    alloc = {h: max(1, int(round(n_subset * N_h[h] * S_h[h] / total))) for h in ids}
    diff = n_subset - sum(alloc.values())
    if diff:
        largest = max(ids, key=lambda h: N_h[h])
        alloc[largest] = max(1, alloc[largest] + diff)

    sel, w = [], []
    for h in ids:
        pool_h = idx_h[h]
        n_h = min(alloc[h], len(pool_h))
        if n_h <= 0:
            continue
        chosen = rng.choice(pool_h, n_h, replace=False)
        sel.append(chosen)
        w.append(np.full(n_h, len(pool_h) / n_h))
    sel = np.concatenate(sel)
    w = np.concatenate(w)
    # Split into the (illicit, benign) shape the caller expects.
    is_ill = np.isin(sel, ill_pool)
    return sel[is_ill], sel[~is_ill], w[is_ill], w[~is_ill]


def b6_leskovec(rng, ben_pool, probs, threshold, time_step, n_select,
                ill_pool=None, src=None, dst=None, **_):
    """Random-walk sampling seeded from illicit nodes (Leskovec 2006)."""
    if src is None or dst is None:
        return b1_random(rng, ben_pool, probs, threshold, time_step, n_select)
    n = len(probs)
    adj: list[list[int]] = [[] for _ in range(n)]
    for s, d in zip(src, dst):
        adj[s].append(d); adj[d].append(s)
    seen, sel = set(), []
    seeds = rng.choice(ill_pool, min(64, len(ill_pool)), replace=False)
    while len(sel) < n_select and len(seeds) > 0:
        v = int(rng.choice(seeds))
        for _ in range(50):
            if not adj[v]: break
            v = int(rng.choice(adj[v]))
            if v in seen: continue
            if v in ben_pool: sel.append(v); seen.add(v)
            if len(sel) >= n_select: break
    sel_arr = np.array(sel[:n_select], dtype=int)
    if len(sel_arr) < n_select:
        used = np.zeros(n, dtype=bool); used[sel_arr] = True
        spare = ben_pool[~used[ben_pool]]
        extra = rng.choice(spare, n_select - len(sel_arr), replace=False)
        sel_arr = np.concatenate([sel_arr, extra])
    return sel_arr, np.full(len(sel_arr), len(ben_pool) / len(sel_arr))


def b7_gao(rng, ben_pool, probs, threshold, time_step, n_select,
           src=None, dst=None, **_):
    """Stratified by node degree (Gao et al. 2018) — graph-aware, no IW."""
    if src is None or dst is None:
        return b1_random(rng, ben_pool, probs, threshold, time_step, n_select)
    deg = np.zeros(len(probs), dtype=np.int32)
    for s in src: deg[s] += 1
    for d in dst: deg[d] += 1
    deg_ben = deg[ben_pool]
    q33, q67 = np.percentile(deg_ben, [33, 67])
    strata = [
        deg_ben < q33,
        (deg_ben >= q33) & (deg_ben < q67),
        deg_ben >= q67,
    ]
    parts, ws = [], []
    n_per = max(1, n_select // 3)
    for mask in strata:
        pool = ben_pool[mask]
        if len(pool) == 0: continue
        n_take = min(n_per, len(pool))
        chosen = rng.choice(pool, n_take, replace=False)
        parts.append(chosen); ws.append(np.full(n_take, len(pool) / n_take))
    sel = np.concatenate(parts) if parts else ben_pool[:0]
    w = np.concatenate(ws)      if ws    else np.array([], dtype=float)
    return sel[:n_select], w[:n_select]


BASELINES = {
    "B1_random":      b1_random,
    "B2_stratified":  b2_stratified,
    "B2t_time":       b2t_stratified_time,
    "B3_no_iw":       b3_no_iw,
    "B4_no_hardneg":  b4_no_hardneg,
    "B5_fogliato":    b5_fogliato,
    "B6_leskovec":    b6_leskovec,
    "B7_gao":         b7_gao,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--ml-dir",   type=Path, default=ML_DIR)
    ap.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011])
    ap.add_argument("--benign-multiplier", type=float, default=2.0)
    ap.add_argument("--hard-neg-ratio",     type=float, default=0.3)
    ap.add_argument("--k-repeats", type=int, default=50)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test = np.load(args.proc_dir / "test.npz")
    edges = np.load(args.proc_dir / "edges.npz")
    src_g, dst_g = edges["src"], edges["dst"]

    y = test["y"].astype(np.int32)
    time_step = test["time_step"].astype(np.int32)
    n = len(y)
    ill_pool = np.where(y == 1)[0]
    ben_pool = np.where(y == 0)[0]
    n_select = int(round(args.benign_multiplier * len(ill_pool)))

    # Edges in `edges.npz` use *global* row IDs (the ~204k node universe).
    # Baselines (B6/B7) need them in test-local indices (0..n-1). Build a
    # global→local map and drop edges that don't connect two test nodes.
    test_row_id = test["row_id"].astype(np.int64)
    n_global = int(max(src_g.max(), dst_g.max(), test_row_id.max())) + 1
    global_to_local = -np.ones(n_global, dtype=np.int64)
    global_to_local[test_row_id] = np.arange(len(test_row_id), dtype=np.int64)
    src_l = global_to_local[src_g]
    dst_l = global_to_local[dst_g]
    keep = (src_l >= 0) & (dst_l >= 0)
    src, dst = src_l[keep].astype(np.int32), dst_l[keep].astype(np.int32)
    print(f"[coreset] test-local edges: {len(src):,} "
          f"(of {len(src_g):,} global edges)")

    print(f"[coreset] test n={n}  illicit={len(ill_pool)}  benign={len(ben_pool)}")
    print(f"          benign budget = {n_select} (multiplier {args.benign_multiplier}x)")

    fogliato_strata = None
    probs, thr = load_ensemble_probs(args.ml_dir, args.seeds)
    if len(probs) != n:
        raise ValueError(f"prob array length {len(probs)} != test size {n}; "
                         "did train_ml_baselines.py write predictions on the test split?")
    print(f"[coreset] avg ensemble threshold across seeds: {thr:.3f}")
    if "B5_fogliato" in args.baselines:
        fogliato_strata = precompute_fogliato_strata(probs, y)
        print(f"[coreset] Fogliato k-means strata: "
              f"{len(fogliato_strata['stratum_ids'])} clusters")

    # ── HT-Coreset: deterministic canonical draw, plus extras for variance ──
    rng = np.random.default_rng(args.seed_base)
    sel, w = ht_select(rng, ben_pool, probs, thr, time_step, n_select,
                       hard_neg_ratio=args.hard_neg_ratio)
    full_idx = np.concatenate([ill_pool, sel])
    full_w   = np.concatenate([np.ones(len(ill_pool)), w])
    np.savez(args.out_dir / "v2.npz",
             idx=full_idx, weights=full_w, threshold=thr,
             ensemble_probs=probs, y=y, time_step=time_step)
    print(f"[coreset] HT-Coreset → {args.out_dir/'v2.npz'} "
          f"(n={len(full_idx)}, illicit retained {len(ill_pool)}/{len(ill_pool)})")

    # ── Baselines: K random draws ──
    for name in args.baselines:
        if name not in BASELINES:
            print(f"  [coreset] skip unknown baseline {name}")
            continue
        fn = BASELINES[name]
        all_idx, all_w, all_y_ill_idx = [], [], []
        for k in range(args.k_repeats):
            rng_k = np.random.default_rng(args.seed_base + k * 7919)
            if name == "B5_fogliato":
                sel_ill, sel_ben, w_ill, w_ben = fn(
                    rng_k, ben_pool, probs, thr, time_step, n_select,
                    ill_pool=ill_pool, src=src, dst=dst,
                    strata=fogliato_strata)
                idx = np.concatenate([sel_ill, sel_ben])
                w_k = np.concatenate([w_ill, w_ben])
                all_y_ill_idx.append(sel_ill)
            else:
                if name in ("B1_random", "B2_stratified"):
                    # These two ablate minority retention, so they draw from
                    # the WHOLE test set and keep whatever illicit nodes the
                    # draw happens to contain. Drawing from ben_pool here was
                    # the defect that produced the 80.63 pp rows.
                    full_pool = np.arange(len(y), dtype=int)
                    sel_b, w_b = fn(rng_k, full_pool, probs, thr, time_step,
                                    n_select + len(ill_pool), y=y)
                    idx = sel_b
                    w_k = w_b
                    all_y_ill_idx.append(idx[y[idx] == 1])
                else:
                    sel_b, w_b = fn(rng_k, ben_pool, probs, thr, time_step,
                                    n_select, ill_pool=ill_pool, src=src, dst=dst)
                    idx = np.concatenate([ill_pool, sel_b])
                    w_k = np.concatenate([np.ones(len(ill_pool)), w_b])
                    all_y_ill_idx.append(ill_pool)
            all_idx.append(idx); all_w.append(w_k)

        np.savez(args.out_dir / f"{name}.npz",
                 draws_idx=np.array(all_idx, dtype=object),
                 draws_w  =np.array(all_w,   dtype=object),
                 draws_ill=np.array(all_y_ill_idx, dtype=object),
                 threshold=thr)
        print(f"[coreset] {name} → {args.k_repeats} draws written")

    meta = dict(
        n_test=int(n),
        n_illicit=int(len(ill_pool)),
        n_benign=int(len(ben_pool)),
        benign_budget=int(n_select),
        threshold=float(thr),
        seeds=args.seeds,
        k_repeats=int(args.k_repeats),
        hard_neg_ratio=float(args.hard_neg_ratio),
    )
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[coreset] wrote {args.out_dir/'meta.json'}")


if __name__ == "__main__":
    main()
