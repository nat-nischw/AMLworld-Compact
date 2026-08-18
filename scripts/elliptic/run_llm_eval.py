#!/usr/bin/env python3
"""LLM prediction on the Elliptic HT-Coreset.

Reads `outputs/elliptic/llm_data/cases.jsonl` (produced by
serialize_coreset.py), sends each case to a vLLM server via the OpenAI
chat-completions API, parses the verdict, and writes:

    outputs/elliptic/llm/{Model}/seed_{S}.json
        {
          "model":      "<name>",
          "seed":       <int>,
          "n_cases":    <int>,
          "predictions": [
              {"case_id": "elliptic_test_42",
               "pred_illicit": 0 or 1,
               "raw_response": "<text>",
               "reasoning_content": "<think> tokens, if exposed by vLLM>"},
              ...
          ]
        }

The output schema matches what `doubt_triage.py --llm-preds` expects, so the
two scripts compose without translation.

Prompt is intentionally simple (zero-shot, no AMLworld-specific typology
list): we are testing whether the *over-prediction* failure mode reported
in the main paper transfers to real Bitcoin data. ICL-style few-shot is
out of scope here; if richer prompting is needed, run the AMLworld LLM stage
(`amlc.llm.runner`, via `make llm-eval`) against this JSONL instead.

Usage:
    python scripts/elliptic/run_llm_eval.py \
        --model GPT-OSS-120B \
        --vllm-url http://<vllm-host>:18809/v1 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_DATA_DIR = REPO_ROOT / "outputs" / "elliptic" / "llm_data"
OUT_DIR      = REPO_ROOT / "outputs" / "elliptic" / "llm"

PROMPT_TEMPLATE = """You are an expert in Bitcoin financial crime.
You will see one Bitcoin transaction (the "query transaction") together
with its 2-hop neighbourhood of UTXO spend edges. Each line lists a
directed edge from one transaction to another, with each transaction's
time step (block-time bucket) and currently-known label
(ill = illicit, lic = licit, ? = unlabelled).

Decide whether the QUERY transaction is SUSPICIOUS (likely illicit:
ransomware proceeds, scams, mixers, dark-market settlement, etc.) or
NOT SUSPICIOUS (routine transfer, exchange operation, normal payment).

Subgraph:
{edge_list_text}

Answer with EXACTLY this format on one line:

VERDICT: <Suspicious or Not Suspicious>
REASON: <one sentence>
"""


VERDICT_RE = re.compile(r"VERDICT\s*:\s*(suspicious|not\s+suspicious)",
                        re.IGNORECASE)


def parse_verdict(text: str) -> int:
    """Return 1 if Suspicious, 0 if Not Suspicious, 0 fallback if unparseable."""
    if not text:
        return 0
    m = VERDICT_RE.search(text)
    if m:
        return 0 if "not" in m.group(1).lower() else 1
    # Fallback: if the model emits "Suspicious" anywhere on its own line
    for line in text.splitlines():
        s = line.strip().lower()
        if s == "suspicious":
            return 1
        if s in ("not suspicious", "not_suspicious"):
            return 0
    # Last-resort heuristic
    return 1 if "suspicious" in text.lower() and "not suspicious" not in text.lower() else 0


def call_vllm(client, model: str, prompt: str, max_tokens: int,
              temperature: float, top_p: float, timeout: int) -> dict:
    """Single vLLM chat call; returns {raw_response, reasoning_content}."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
    )
    msg = resp.choices[0].message
    raw = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", "") or ""
    return dict(raw_response=raw, reasoning_content=reasoning)


def predict_one(client, model: str, rec: dict, *, max_tokens: int,
                temperature: float, top_p: float, timeout: int,
                retries: int) -> dict:
    prompt = PROMPT_TEMPLATE.format(edge_list_text=rec["edge_list_text"])
    last_err = None
    for attempt in range(retries):
        try:
            out = call_vllm(client, model, prompt, max_tokens,
                            temperature, top_p, timeout)
            pred = parse_verdict(out["raw_response"])
            return dict(
                case_id=rec["case_id"], row_id=rec["row_id"],
                label=rec["label"], time_step=rec["time_step"],
                weight=rec["weight"],
                pred_illicit=int(pred),
                raw_response=out["raw_response"],
                reasoning_content=out["reasoning_content"],
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return dict(
        case_id=rec["case_id"], row_id=rec.get("row_id", -1),
        label=rec["label"], time_step=rec["time_step"], weight=rec["weight"],
        pred_illicit=0, raw_response="", reasoning_content="",
        error=str(last_err),
    )


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",     type=str, required=True,
                    help="served-model-name in vLLM (e.g. GPT-OSS-120B)")
    ap.add_argument("--vllm-url",  type=str, default="http://localhost:18809/v1")
    ap.add_argument("--cases",     type=Path,
                    default=LLM_DATA_DIR / "cases.jsonl")
    ap.add_argument("--out-dir",   type=Path, default=OUT_DIR)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--workers",   type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="LLM output cap (Bitcoin reasoning is short, 2K is plenty)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p",      type=float, default=0.95)
    ap.add_argument("--timeout",    type=int,   default=180)
    ap.add_argument("--retries",    type=int,   default=3)
    ap.add_argument("--limit",      type=int,   default=None,
                    help="optional: only process the first N cases (debug)")
    ap.add_argument("--api-key",    type=str,   default="EMPTY",
                    help="vLLM doesn't enforce auth; any string works")
    args = ap.parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package required: `pip install openai`")

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    print(f"[predict] {len(cases)} cases  model={args.model}  url={args.vllm_url}")

    client = OpenAI(base_url=args.vllm_url, api_key=args.api_key)
    out_dir = args.out_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(predict_one, client, args.model, c,
                      max_tokens=args.max_tokens, temperature=args.temperature,
                      top_p=args.top_p, timeout=args.timeout,
                      retries=args.retries): c["case_id"]
            for c in cases
        }
        n_done, n_err = 0, 0
        for fut in as_completed(futures):
            rec = fut.result()
            if rec.get("error"): n_err += 1
            predictions.append(rec)
            n_done += 1
            if n_done % 100 == 0:
                rate = n_done / (time.time() - t0)
                eta = (len(cases) - n_done) / max(rate, 1e-6)
                print(f"  {n_done}/{len(cases)}  err={n_err}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f} min")

    pred_illicit = sum(p["pred_illicit"] for p in predictions)
    print(f"[predict] done in {(time.time()-t0)/60:.1f} min  "
          f"errors={n_err}  predicted illicit={pred_illicit}/{len(predictions)} "
          f"({pred_illicit/len(predictions)*100:.1f}%)")

    out_path = out_dir / f"seed_{args.seed}.json"
    out_path.write_text(json.dumps(dict(
        model=args.model, seed=args.seed, n_cases=len(predictions),
        vllm_url=args.vllm_url, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens,
        predictions=predictions,
    ), indent=2))
    print(f"[predict] wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
