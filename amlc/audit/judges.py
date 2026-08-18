"""Score the audit slice with a frontier reasoning API as an independent judge.

The four-step rubric is applied to the same 1,000 traces by four annotators: the
keyword heuristic in :mod:`amlc.audit.sample_traces` and the three API
judges here. Agreement across them is what rules out the Conclude collapse being
an artefact of one annotator; :mod:`amlc.audit.iaa` computes it.

Three providers, one module. The pre-release tree carried a separate script per
provider, and they differed only in how the request is built: the rubric, the
retry policy, the response parsing, the truncation limit and the output schema
were copies. Only the request builders are provider-specific here.

    deepseek   DeepSeek reasoning model over the OpenAI-compatible endpoint,
               JSON response format, thinking enabled.
    opus       Anthropic Messages API, rubric as the system prompt.
    gemini     Google GenAI with a response schema and thinking level HIGH.
               Temperature 1.0, per Google's guidance for thinking mode.

Reads
    ``results/audit/trace_4step_annotations_n1000.csv`` for the slice, its
    ground truth and the model predictions, and the archived prediction JSONs
    for the trace text. The rubric comes from
    ``prompts/four_step_rubric_{system,user}.j2``.

Writes
    ``results/audit/trace_4step_<provider>_n1000.csv``: the four step scores
    under this judge's column prefix (``ds_``, ``op_``, ``gm_``), plus one error
    column recording any call that never returned usable JSON.

Prompt provenance
-----------------
The templates are byte-identical to the prompt the DeepSeek run sent. Two small
differences in the other two are dropped so that all three judges see the same
rubric: the Gemini script's system prompt ended without the trailing newline,
and the Opus script appended one further sentence, "Output ONLY the JSON object,
no preamble or commentary." That sentence guarded against a prose preamble
around the JSON, which the tolerant parser here handles for every provider.

Missing typology in the prompt
------------------------------
A ground-truth typology that AMLworld does not label arrives from the CSV as
NaN, and the archived scripts formatted it with ``str()`` before the
``or "(none)"`` fallback, so the judges were shown ``typology=nan``. Preserved
in :func:`_format_field`, because the released judge annotations were produced
with that prompt, and quietly changing it would leave them unreproducible.

Keys come from the environment: ``DEEPSEEK_API_KEY``, ``ANTHROPIC_API_KEY``,
``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``). See ``.env.example``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import config
from ..triage.doubt_triage import resolve_archive
from .sample_traces import load_traces

PROVIDERS = config.JUDGE_PROVIDERS

#: Column prefix each judge's scores are written under. The released CSVs and
#: amlc.audit.iaa join on these.
COLUMN_PREFIX = config.JUDGE_COLUMN_PREFIX

#: The model that produced the released annotations for each provider. This is
#: provenance, not a tuning knob; override per provider with
#: ``AMLC_JUDGE_MODEL_DEEPSEEK`` and so on, or with ``--model``.
RELEASED_JUDGE_MODEL = config.RELEASED_JUDGE_MODEL

_API_KEY_ENV = config.JUDGE_API_KEY_ENV

#: Characters of reasoning trace shown to a judge. Longer traces are cut.
MAX_TRACE_CHARS = config.JUDGE_MAX_TRACE_CHARS

#: Attempts per trace, with exponential backoff between them.
MAX_ATTEMPTS = config.JUDGE_MAX_ATTEMPTS

_FAILED = {step: False for step in config.RUBRIC_STEPS}

#: A JSON object holding the rubric keys, for pulling the verdict out of prose.
_JSON_RE = re.compile(r"\{[^{}]*\"parse\"[^{}]*\}", re.DOTALL)


# ── Prompt ───────────────────────────────────────────────────────────────

def render_rubric(gt_label: int, gt_typology, pred_illicit: bool, pred_typology,
                  trace: str, template_root: Optional[Path] = None,
                  max_chars: int = MAX_TRACE_CHARS) -> tuple[str, str]:
    """Render the (system, user) rubric prompt for one trace."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    from ..paths import prompts

    env = Environment(
        loader=FileSystemLoader(str(Path(template_root) if template_root else prompts())),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    system = env.get_template("four_step_rubric_system.j2").render()
    user = env.get_template("four_step_rubric_user.j2").render(
        gt_label="illicit" if int(gt_label) == 1 else "benign",
        gt_typology=_format_field(gt_typology),
        pred_illicit="True" if pred_illicit else "False",
        pred_typology=_format_field(pred_typology),
        trace=trace[:max_chars],
    )
    return system, user


def _format_field(value) -> str:
    """Format a ground-truth or prediction field, as the archived judges saw it.

    A missing value that reached the archived scripts as ``None`` became
    ``(none)``; one that reached them as a pandas NaN became the string ``nan``,
    because ``str(nan)`` is truthy and the fallback never fired. Both behaviours
    are kept so a rerun sends the prompt the released annotations came from.
    """
    if value is None:
        return "(none)"
    return str(value) or "(none)"


def parse_verdict(text: str) -> dict:
    """Pull the four booleans out of a judge response.

    Tries the whole response as JSON, then a JSON object embedded in prose, then
    a fenced code block. Raises when none of the three yields an object.
    """
    text = (text or "").strip()
    for candidate in (text, _match_or_none(text), _unfenced_or_none(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError(f"unparseable judge response: {text[:200]}")


def _match_or_none(text: str) -> Optional[str]:
    m = _JSON_RE.search(text)
    return m.group(0) if m else None


def _unfenced_or_none(text: str) -> Optional[str]:
    if "```" not in text:
        return None
    return re.sub(r"```(?:json)?", "", text).replace("```", "").strip()


# ── Providers ────────────────────────────────────────────────────────────

def judge_model(provider: str) -> str:
    """Model id for a provider: environment override, else the released one."""
    return os.environ.get(f"AMLC_JUDGE_MODEL_{provider.upper()}",
                          RELEASED_JUDGE_MODEL[provider])


def api_key(provider: str) -> str:
    for name in _API_KEY_ENV[provider]:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        f"{provider} judge needs one of {', '.join(_API_KEY_ENV[provider])} in the "
        "environment; see .env.example"
    )


def client_for(provider: str) -> object:
    """Build the provider's client. Imported here so the module stays light."""
    key = api_key(provider)
    if provider == "deepseek":
        from openai import OpenAI

        base_url = config.DEEPSEEK_BASE_URL
        return OpenAI(api_key=key, base_url=base_url)
    if provider == "opus":
        import anthropic

        return anthropic.Anthropic(api_key=key)
    if provider == "gemini":
        from google import genai

        return genai.Client(api_key=key)
    raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")


def _call_deepseek(client, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_tokens=2000,
        response_format={"type": "json_object"},
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )
    return resp.choices[0].message.content


def _call_opus(client, model: str, system: str, user: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content
                   if getattr(b, "type", None) == "text")


def _call_gemini(client, model: str, system: str, user: str) -> str:
    from google.genai import types

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={step: types.Schema(type=types.Type.BOOLEAN)
                    for step in config.RUBRIC_STEPS},
        required=list(config.RUBRIC_STEPS),
    )
    # Thinking-mode settings from Google's guidance: temperature 1.0, top_p at
    # the default 0.95, thinking level HIGH, and the model's maximum output.
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        max_output_tokens=8192,
        temperature=1.0,
        top_p=0.95,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
    )
    return client.models.generate_content(model=model, contents=user, config=cfg).text


_CALL = {"deepseek": _call_deepseek, "opus": _call_opus, "gemini": _call_gemini}


# ── Scoring ──────────────────────────────────────────────────────────────

def score_trace(provider: str, client, model: str, gt_label: int, gt_typology,
                pred_illicit: bool, pred_typology, trace: Optional[str],
                max_chars: int = MAX_TRACE_CHARS) -> dict:
    """Score one trace. Never raises: a failed call returns four failures.

    A trace that is missing or empty is a failure of all four steps, not a
    missing observation, because the model produced no reasoning to score. This
    keeps the judge tables the same length as the slice.
    """
    if not trace:
        return {**_FAILED, "_error": "empty_trace"}

    system, user = render_rubric(gt_label, gt_typology, pred_illicit,
                                pred_typology, trace, max_chars=max_chars)
    for attempt in range(MAX_ATTEMPTS):
        try:
            obj = parse_verdict(_CALL[provider](client, model, system, user))
            return {step: bool(obj.get(step, False)) for step in config.RUBRIC_STEPS}
        except Exception as exc:  # provider errors, rate limits, bad JSON
            if attempt == MAX_ATTEMPTS - 1:
                return {**_FAILED, "_error": str(exc)[:200]}
            time.sleep(2 ** attempt)
    return {**_FAILED, "_error": "retry_exhausted"}


def annotate(provider: str, slice_df: pd.DataFrame, archive: Path,
             dataset: str = "HI-Small", prompting: str = "ICL-FS",
             seed: int = config.RUBRIC_SEED, model: Optional[str] = None,
             workers: int = 4, max_chars: int = MAX_TRACE_CHARS,
             progress_every: int = 25) -> pd.DataFrame:
    """Score every row of the slice with one judge."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}, got {provider!r}")
    model = model or judge_model(provider)
    client = client_for(provider)

    traces: dict[str, dict] = {}

    def trace_for(model_name: str, case_id: str) -> Optional[str]:
        if model_name not in traces:
            traces[model_name] = load_traces(archive, dataset, model_name,
                                             prompting, seed) or {}
        rec = traces[model_name].get(case_id)
        return None if rec is None else (rec.get("reasoning_content", "") or "")

    def task(i: int, row) -> tuple[int, dict]:
        return i, score_trace(
            provider, client, model,
            int(row.gt_label), str(row.gt_typology),
            bool(row.pred_illicit),
            str(row.pred_typology) if pd.notna(row.pred_typology) else None,
            trace_for(row.model, row.case_id), max_chars,
        )

    results: list[Optional[dict]] = [None] * len(slice_df)
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, i, row)
                   for i, row in enumerate(slice_df.itertuples())]
        done = 0
        for future in as_completed(futures):
            i, scores = future.result()
            results[i] = scores
            done += 1
            if done % progress_every == 0 or done == len(slice_df):
                rate = done / max(time.time() - started, 1e-3)
                print(f"  [{done}/{len(slice_df)}] {rate:.2f}/s "
                      f"ETA {(len(slice_df) - done) / max(rate, 1e-3):.0f}s")

    prefix = COLUMN_PREFIX[provider]
    scored = pd.DataFrame(results)
    out = pd.concat(
        [slice_df[["model", "case_id", "outcome"]].reset_index(drop=True),
         scored[list(config.RUBRIC_STEPS)].rename(
             columns={s: f"{prefix}{s}" for s in config.RUBRIC_STEPS})],
        axis=1)
    # One error column, named for this judge. The pre-release scripts emitted
    # both the raw "_error" and the prefixed copy of it.
    out[f"{prefix}error"] = scored["_error"] if "_error" in scored.columns else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("provider", choices=PROVIDERS)
    ap.add_argument("--archive", type=Path, default=None,
                    help="archived experiment tree (default: $AMLC_ARCHIVE)")
    ap.add_argument("--slice", type=Path, default=None,
                    help="regex annotation CSV naming the slice")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default=None,
                    help=f"judge model id (default: {RELEASED_JUDGE_MODEL})")
    ap.add_argument("--dataset", default=config.DATASETS[0], choices=config.DATASETS)
    ap.add_argument("--prompting", default="ICL-FS", choices=config.PROMPTINGS)
    ap.add_argument("--seed", type=int, default=config.RUBRIC_SEED)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=MAX_TRACE_CHARS)
    args = ap.parse_args()

    from ..paths import ensure, results

    audit = ensure(results() / "audit")
    n = config.RUBRIC_N_TRACES
    slice_path = args.slice or audit / f"trace_4step_annotations_n{n}.csv"
    slice_df = pd.read_csv(slice_path)
    print(f"Loaded {len(slice_df)} traces from {slice_path}")

    model = args.model or judge_model(args.provider)
    print(f"Judging with {args.provider} ({model}), {args.workers} workers")
    out = annotate(args.provider, slice_df, resolve_archive(args.archive),
                   dataset=args.dataset, prompting=args.prompting, seed=args.seed,
                   model=model, workers=args.workers, max_chars=args.max_chars)

    path = args.out or audit / f"trace_4step_{args.provider}_n{n}.csv"
    out.to_csv(path, index=False)
    print(f"\nSaved {path} ({len(out)} rows)")

    prefix = COLUMN_PREFIX[args.provider]
    n_failed = int(out[f"{prefix}error"].notna().sum())
    print(f"\n{args.provider} pass rates ({n_failed} call failures)")
    for step in config.RUBRIC_STEPS:
        print(f"  {step:10s} {out[f'{prefix}{step}'].astype(float).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
