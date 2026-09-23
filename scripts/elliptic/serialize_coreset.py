#!/usr/bin/env python3
"""Serialize the Elliptic HT-Coreset for LLM evaluation.

Each test transaction is wrapped in its k-hop directed neighbourhood (default
k=2, max 30 neighbours per hop) and rendered as edge_list-style text — the
same format the LLM sees on AMLworld. Edges are *directed* (Bitcoin UTXO
spend), so we expose `from → to` and tag the focal transaction with
`(query_tx)`.

Output: `outputs/elliptic/llm_data/cases.jsonl`
    one JSON line per case:
      { case_id, label, time_step, weight, n_tokens, edge_list_text }

This is enough to drop straight into the existing run_all_baselines.py
ICL-AML pipeline by pointing `--cases-jsonl` at this file. (LLM hosting and
prediction are separate; this script only prepares the text.)

Tokens are estimated with `tiktoken cl100k_base` if available, otherwise a
~4 chars-per-token heuristic.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR  = REPO_ROOT / "data" / "elliptic" / "proc"
CORE_DIR  = REPO_ROOT / "outputs" / "elliptic" / "coreset"
OUT_DIR   = REPO_ROOT / "outputs" / "elliptic" / "llm_data"


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def build_adjacency(src: np.ndarray, dst: np.ndarray, n_nodes: int):
    """Return forward and reverse adjacency lists."""
    fwd = defaultdict(list); rev = defaultdict(list)
    for s, d in zip(src, dst):
        fwd[int(s)].append(int(d))
        rev[int(d)].append(int(s))
    return fwd, rev


def k_hop_neighbours(focal: int, fwd, rev, k: int, max_per_hop: int,
                     rng: np.random.Generator) -> set[int]:
    visited = {focal}; frontier = {focal}
    for _ in range(k):
        nxt = set()
        for v in frontier:
            cand = list(fwd.get(v, [])) + list(rev.get(v, []))
            cand = [c for c in cand if c not in visited]
            if len(cand) > max_per_hop:
                cand = list(rng.choice(cand, max_per_hop, replace=False))
            nxt.update(cand)
        if not nxt: break
        visited.update(nxt); frontier = nxt
    return visited


def render_edge_list(focal: int, nodes: set[int], fwd, rev,
                     time_step: np.ndarray, label: np.ndarray,
                     masked: set[int] | None = None) -> str:
    """Render directed edges among `nodes`, with the focal node tagged.

    ``masked`` holds nodes whose label must not appear in the prompt. Every
    node under evaluation belongs there, the focal node above all.

    Fixed 2026-08-10. The pre-release version printed ``label[u]`` for every
    node including the focal one, so a prompt read

        *tx_136285 (t=35,ill)  ->   tx_141053 (t=35,ill)

    with ``*`` marking the transaction the model was being asked to classify
    and ``ill`` giving away its ground truth. Any evaluation on that artefact
    would measure copying, not reasoning.

    No published number is affected: the Elliptic LLM stage was never run
    (``outputs/elliptic/`` holds only ``coreset/``, ``llm_data/`` and ``ml/``),
    and no Elliptic LLM or Doubt Triage result appears in the paper.
    """
    masked = masked or set()
    lines = [f"# query transaction: tx_{focal} (time_step {int(time_step[focal])})",
             "# edges (from → to) within the 2-hop neighbourhood:"]
    seen = set()
    for u in nodes:
        for v in fwd.get(u, []):
            if v in nodes and (u, v) not in seen:
                seen.add((u, v))
                tag_u = "*" if u == focal else " "
                tag_v = "*" if v == focal else " "
                lab_u = ("?" if u in masked
                         else {1: "ill", 0: "lic", -1: "?"}.get(int(label[u]), "?"))
                lab_v = ("?" if v in masked
                         else {1: "ill", 0: "lic", -1: "?"}.get(int(label[v]), "?"))
                lines.append(
                    f"  {tag_u}tx_{u} (t={int(time_step[u])},{lab_u})  →  "
                    f"{tag_v}tx_{v} (t={int(time_step[v])},{lab_v})"
                )
    if len(seen) == 0:
        lines.append("  (no edges within 2-hop neighbourhood; isolated transaction)")
    lines.append(f"# focal node degree:  in={len(rev.get(focal, []))}  "
                 f"out={len(fwd.get(focal, []))}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    ap.add_argument("--core-dir", type=Path, default=CORE_DIR)
    ap.add_argument("--out-dir",  type=Path, default=OUT_DIR)
    ap.add_argument("--k-hop", type=int, default=2)
    ap.add_argument("--max-neighbors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reveal-labels", action="store_true",
                    help="reproduce the pre-release artefact, which leaked the "
                         "focal node's ground truth into its own prompt")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    proc_test = np.load(args.proc_dir / "test.npz")
    proc_node = np.load(args.proc_dir / "node_index.npz")
    edges     = np.load(args.proc_dir / "edges.npz")
    coreset   = np.load(args.core_dir / "v2.npz")

    test_row_id = proc_test["row_id"]
    test_idx_in_test = coreset["idx"]                  # indices into test split
    weights = coreset["weights"]
    focal_row_ids = test_row_id[test_idx_in_test]      # global row ids

    src, dst = edges["src"], edges["dst"]
    n_nodes  = len(proc_node["txid"])
    fwd, rev = build_adjacency(src, dst, n_nodes)

    time_step = proc_node["time_step"].astype(np.int32)
    label     = proc_node["label"].astype(np.int8)

    # Labels of everything under evaluation are hidden from the prompt: the
    # focal transaction and every other node in the coreset. Neighbours outside
    # the evaluation set keep their label, which is information a deployed
    # system would legitimately hold about historically adjudicated
    # transactions.
    masked_nodes = set(int(r) for r in focal_row_ids)
    if args.reveal_labels:
        masked_nodes = set()
        print("[serialize] WARNING --reveal-labels: the focal node's ground "
              "truth will appear in its own prompt. Reproduction only.")
    else:
        print(f"[serialize] masking labels for {len(masked_nodes)} evaluated nodes")

    out_path = args.out_dir / "cases.jsonl"
    n_tokens_total = 0
    print(f"[serialize] writing {len(focal_row_ids)} cases → {out_path}")
    with out_path.open("w") as f:
        for i, (sub_idx, row_id, w) in enumerate(zip(test_idx_in_test,
                                                      focal_row_ids, weights)):
            nbrs = k_hop_neighbours(int(row_id), fwd, rev,
                                    args.k_hop, args.max_neighbors, rng)
            text = render_edge_list(int(row_id), nbrs, fwd, rev,
                                    time_step, label, masked=masked_nodes)
            n_tok = estimate_tokens(text)
            n_tokens_total += n_tok
            rec = dict(
                case_id=f"elliptic_test_{int(sub_idx)}",
                row_id=int(row_id),
                label=int(proc_test["y"][int(sub_idx)]),
                time_step=int(proc_test["time_step"][int(sub_idx)]),
                weight=float(w),
                n_tokens=int(n_tok),
                edge_list_text=text,
            )
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(focal_row_ids)} cases  "
                      f"(running mean tokens/case: {n_tokens_total/(i+1):.0f})")

    print(f"[serialize] done: {len(focal_row_ids)} cases, "
          f"avg {n_tokens_total/max(1,len(focal_row_ids)):.0f} tokens/case, "
          f"total {n_tokens_total/1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
