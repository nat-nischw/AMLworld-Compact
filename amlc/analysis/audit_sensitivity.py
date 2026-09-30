"""Offline, descriptive sensitivity of the frozen outcome-stratified trace audit.

The archived prediction fields and API judgements are inputs, never rewritten.
Portable features contain no trace text. Original lexical rules are imported
unchanged; alternative counts describe text presence, not factual correctness.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from ..audit import sample_traces
from ..case_ids import is_case_id, to_canonical
from ..llm.methods import PATTERN_MAP

PREFIX_CHARS = 8000
KEY = ["model", "case_id"]
STEPS = ("parse", "recall", "match")
JUDGES = {"deepseek": "ds_", "opus": "op_", "gemini": "gm_"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(value) -> bool:
    """Read CSV booleans explicitly; bool('False') would silently be wrong."""
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def read_annotations(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate annotation model/case keys")
    for col in ("pred_illicit", "parse", "recall", "match", "conclude"):
        frame[col] = frame[col].map(as_bool)
    frame["gt_label"] = pd.to_numeric(frame["gt_label"])
    if not frame["gt_label"].isin([0, 1]).all():
        raise ValueError("ground-truth binary labels must be 0 or 1")
    frame["outcome"] = frame["outcome"].map(sample_traces.normalise_outcome)
    return frame


def lexical_features(trace: str, pred_illicit: bool) -> dict:
    """Apply the original scorer and count its categories without the bypass.

    Canonical Recall is an additional, ground-truth-free lexical diagnostic:
    count distinct classes in the pre-existing final-parser PATTERN_MAP using
    case-insensitive substring presence. It is not a new final-answer parser.
    No new synonyms, Unicode folding, or judge-dependent tuning are introduced.
    """
    low = trace.lower()
    markers = {
        name: any(re.search(pattern, low, re.IGNORECASE) for pattern in patterns)
        for name, patterns in sample_traces.MARKER_PATTERNS.items()
    }
    raw_names = {name.replace(" ", "-").replace("_", "-")
                 for name in sample_traces.TYPOLOGY_NAMES if name in low}
    canonical_names = {canonical for surface, canonical in PATTERN_MAP if surface in low}
    scores = sample_traces.annotate_trace(
        trace, {"illicit": pred_illicit, "typology": None}, 0, "")
    return {
        **{step: int(scores[step]) for step in STEPS},
        "marker_count": sum(markers.values()),
        **{f"marker_{name}": int(value) for name, value in markers.items()},
        "marker_ge2": int(sum(markers.values()) >= 2),
        "original_typology_name_count": len(raw_names),
        "canonical_typology_count": len(canonical_names),
        "canonical_recall": int(len(canonical_names) >= 2),
        "benign_match_bypass": int(bool(trace) and not pred_illicit),
        "match_without_two_markers": int(scores["match"] and sum(markers.values()) < 2),
    }


def source_contract() -> dict:
    """Fingerprint the frozen lexical definitions and predefined class map."""
    from ..llm import methods

    definitions = {
        "parse_keywords": sample_traces.PARSE_KEYWORDS,
        "typology_names": sample_traces.TYPOLOGY_NAMES,
        "marker_patterns": sample_traces.MARKER_PATTERNS,
        "canonical_typology_map": list(PATTERN_MAP),
        "prefix_chars": PREFIX_CHARS,
    }
    return {
        "definitions": definitions,
        "definitions_sha256": hashlib.sha256(
            json.dumps(definitions, sort_keys=True).encode()).hexdigest(),
        "original_scorer_sha256": sha256(Path(sample_traces.__file__)),
        "predefined_parser_map_source_sha256": sha256(Path(methods.__file__)),
    }


def prepare_features(annotations: pd.DataFrame, archive_root: Path) -> tuple[pd.DataFrame, dict]:
    """Extract full/prefix features from archived HI-Small few-shot seed 42.

    archive_root can name outputs/ or its parent. Case identifiers are mapped
    in keys only; the actual archived text is neither retagged nor rewritten.
    """
    archive_root = Path(archive_root)
    if (archive_root / "outputs").is_dir():
        archive_root = archive_root / "outputs"
    features, files = [], []
    for model, rows in annotations.groupby("model", sort=True):
        relative = Path(model) / "LLM+ICL-AML/HI-Small/seed_42.json"
        path = archive_root / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        records = {}
        for rec in data["predictions"]:
            if is_case_id(rec.get("case_id", "")):
                key = to_canonical(rec["case_id"])
                if key in records:
                    raise ValueError(f"duplicate archive case: {model}/{key}")
                records[key] = rec
        for row in rows.to_dict("records"):
            rec = records[row["case_id"]]
            trace = rec.get("reasoning_content") or ""
            if not isinstance(trace, str):
                raise TypeError("reasoning_content must be text")
            # Predictions in the published annotation remain authoritative.
            # A mismatch is recorded rather than reparsing any final answer.
            prediction_agrees = (as_bool(rec["illicit"]) == row["pred_illicit"]
                                  and (rec.get("typology") or "") == row["pred_typology"])
            out = {"model": model, "case_id": row["case_id"],
                   "trace_chars": len(trace),
                   "trace_sha256": hashlib.sha256(trace.encode("utf-8")).hexdigest(),
                   "prefix_sha256": hashlib.sha256(
                       trace[:PREFIX_CHARS].encode("utf-8")).hexdigest(),
                   "archive_prediction_agrees": int(prediction_agrees)}
            for window, text in (("full", trace), ("prefix8000", trace[:PREFIX_CHARS])):
                out.update({f"{window}_{name}": value for name, value in
                            lexical_features(text, row["pred_illicit"]).items()})
            features.append(out)
        files.append({"path_relative_to_outputs": relative.as_posix(),
                      "sha256": sha256(path), "bytes": path.stat().st_size,
                      "audited_rows": len(rows)})
    return pd.DataFrame(features), {"archive_files": files, **source_contract()}


def attach_labels(annotations: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join portable features and frozen labels with strict one-to-one keys."""
    if features.duplicated(KEY).any():
        raise ValueError("duplicate feature model/case keys")
    merged = annotations.merge(features, on=KEY, how="outer", validate="one_to_one",
                               indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("annotation and feature keys differ")
    merged = merged.drop(columns="_merge")
    illicit = merged["gt_label"].eq(1)
    typed = illicit & merged["gt_typology"].ne("")
    missing = illicit & ~typed
    merged["reference_group"] = "benign"
    merged.loc[typed, "reference_group"] = "illicit_typed"
    merged.loc[missing, "reference_group"] = "illicit_reference_absent"
    merged["binary_correct"] = merged["pred_illicit"].eq(illicit)
    merged["binary_outcome"] = merged["binary_correct"].map({True: "correct", False: "incorrect"})
    merged["reference_evaluable"] = ~missing
    merged["typed_eligible"] = typed
    merged["typed_correct"] = typed & merged["pred_illicit"] & merged[
        "pred_typology"].eq(merged["gt_typology"])
    merged["evaluable_correct"] = ((~illicit & ~merged["pred_illicit"])
                                     | merged["typed_correct"])
    merged["error_class"] = merged["outcome"]
    merged.loc[missing & merged["pred_illicit"], "error_class"] = "reference_absent_flagged"
    merged.loc[missing & ~merged["pred_illicit"], "error_class"] = "reference_absent_missed"
    merged["truncated"] = merged["trace_chars"].gt(PREFIX_CHARS)
    merged["empty_trace"] = merged["trace_chars"].eq(0)
    for window in ("full", "prefix8000"):
        # Minimal synthetic label-only fixtures need no lexical columns.
        if f"{window}_parse" not in merged:
            continue
        parse = merged[f"{window}_parse"].astype(bool)
        recall = merged[f"{window}_recall"].astype(bool)
        markers = merged[f"{window}_marker_ge2"].astype(bool)
        merged[f"{window}_original_PRM"] = parse & recall & merged[f"{window}_match"].astype(bool)
        merged[f"{window}_nonvacuous_PRM"] = parse & recall & markers
        merged[f"{window}_canonical_nonvacuous_PRM"] = (
            parse & merged[f"{window}_canonical_recall"].astype(bool) & markers)
    return merged


def groups(frame: pd.DataFrame):
    """Overall, each model, each outcome/error/reference group, and their crosses."""
    yield {"model": "ALL", "group_kind": "all", "group": "all"}, frame
    for model, subset in frame.groupby("model", sort=True):
        yield {"model": model, "group_kind": "all", "group": "all"}, subset
    for kind in ("outcome", "error_class", "reference_group", "binary_outcome"):
        for label, subset in frame.groupby(kind, sort=True):
            yield {"model": "ALL", "group_kind": kind, "group": label}, subset
        for (model, label), subset in frame.groupby(["model", kind], sort=True):
            yield {"model": model, "group_kind": kind, "group": label}, subset


def correctness_definitions(frame: pd.DataFrame):
    all_rows = pd.Series(True, index=frame.index)
    return {
        "strict_archived": (all_rows, frame["conclude"]),
        "binary": (all_rows, frame["binary_correct"]),
        "exclude_illicit_reference_absent": (frame["reference_evaluable"],
                                               frame["evaluable_correct"]),
        "typed_illicit_only": (frame["typed_eligible"], frame["typed_correct"]),
    }


def rate_record(mask: pd.Series, eligible: pd.Series) -> dict:
    n = int(eligible.sum())
    k = int((mask & eligible).sum())
    return {"count": k, "denominator": n, "rate": k / n if n else None}


def summarize(frame: pd.DataFrame, judge_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Pure report aggregation; never requires traces or a model/API client."""
    rates, conjunctions, truncation, discordance = [], [], [], []
    feature_metrics = [name for name in frame if name.startswith(("full_", "prefix8000_"))
                       and not name.endswith("count")]
    for key, subset in groups(frame):
        all_rows = pd.Series(True, index=subset.index)
        for name in feature_metrics + ["truncated", "empty_trace"]:
            rates.append({**key, "metric": name,
                          **rate_record(subset[name].astype(bool), all_rows)})
        for name, (eligible, correct) in correctness_definitions(subset).items():
            rates.append({**key, "metric": f"correctness_{name}",
                          **rate_record(correct, eligible)})
        for window in ("archived", "full", "prefix8000"):
            prefix = "" if window == "archived" else f"{window}_"
            parse = subset[f"{prefix}parse"].astype(bool)
            recall = subset[f"{prefix}recall"].astype(bool)
            match = subset[f"{prefix}match"].astype(bool)
            rules = {"original_PRM": parse & recall & match}
            if window != "archived":
                marker_ge2 = subset[f"{prefix}marker_ge2"].astype(bool)
                rules["PR_and_two_marker_categories"] = parse & recall & marker_ge2
                rules["P_canonical_R_and_two_marker_categories"] = (
                    parse & subset[f"{prefix}canonical_recall"].astype(bool) & marker_ge2)
            for rule, upstream in rules.items():
                for definition, (eligible, correct) in correctness_definitions(subset).items():
                    conjunctions.append({**key, "window": window, "rule": rule,
                                         "correctness": definition,
                                         "upstream_count": int((upstream & eligible).sum()),
                                         "incorrect_count": int((~correct & eligible).sum()),
                                         **rate_record(upstream & ~correct, eligible)})
        for step in STEPS + ("marker_ge2", "canonical_recall"):
            full = subset[f"full_{step}"].astype(bool)
            prefix = subset[f"prefix8000_{step}"].astype(bool)
            truncation.append({**key, "metric": step, "denominator": len(subset),
                               "truncated_count": int(subset["truncated"].sum()),
                               "full_count": int(full.sum()), "prefix8000_count": int(prefix.sum()),
                               "full_only_count": int((full & ~prefix).sum()),
                               "prefix_only_count": int((~full & prefix).sum()),
                               "archived_full_disagreement_count": (
                                   int(subset[step].ne(full).sum()) if step in STEPS else None)})
    for judge, data in judge_frames.items():
        prefix = JUDGES[judge]
        if data.duplicated(KEY).any():
            raise ValueError(f"duplicate {judge} model/case keys")
        cols = KEY + [f"{prefix}{step}" for step in (*STEPS, "conclude")]
        judge_data = data[cols].copy()
        for step in (*STEPS, "conclude"):
            judge_data[f"{prefix}{step}"] = judge_data[f"{prefix}{step}"].map(as_bool)
        error_cols = [c for c in ("_error", f"{prefix}error") if c in data]
        judge_data["judge_error"] = (data[error_cols].fillna("").ne("").any(axis=1)
                                     if error_cols else False)
        joined = frame.merge(judge_data, on=KEY, how="outer", validate="one_to_one",
                             indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"{judge} and audit keys differ")
        for key, subset in groups(joined):
            valid = ~subset["judge_error"]
            judge_conclude = subset[f"{prefix}conclude"]
            for scope, selected in (("all_rows", pd.Series(True, index=subset.index)),
                                    ("scored_only", valid)):
                for definition, (eligible, parser_correct) in correctness_definitions(subset).items():
                    use = eligible & selected
                    discordance.append({
                        **key, "judge": judge, "judge_score_scope": scope,
                        "parser_definition": definition,
                        "denominator": int(use.sum()),
                        "judge_error_count": int((eligible & ~valid).sum()),
                        "judge_error_excluded_count": int((eligible & ~selected).sum()),
                        "both_pass": int((use & parser_correct & judge_conclude).sum()),
                        "both_fail": int((use & ~parser_correct & ~judge_conclude).sum()),
                        "parser_pass_judge_fail": int((use & parser_correct & ~judge_conclude).sum()),
                        "parser_fail_judge_pass": int((use & ~parser_correct & judge_conclude).sum()),
                    })
                upstream = subset[[f"{prefix}{step}" for step in STEPS]].all(axis=1)
                conjunctions.append({**key, "window": f"archived_{judge}_prefix8000",
                                     "rule": f"judge_PRM_{scope}", "correctness": "judge_conclude",
                                     "upstream_count": int((upstream & selected).sum()),
                                     "incorrect_count": int((~judge_conclude & selected).sum()),
                                     **rate_record(upstream & ~judge_conclude, selected)})
    return {"rates": pd.DataFrame(rates), "conjunctions": pd.DataFrame(conjunctions),
            "truncation": pd.DataFrame(truncation),
            "judge_discordance": pd.DataFrame(discordance)}


def render_readme(frame: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> str:
    overall = lambda table: table[table["model"].eq("ALL") & table["group_kind"].eq("all")]
    lines = [
        "# Offline audit scoring sensitivity", "",
        ("This is a descriptive reanalysis of the frozen 1,000 model/case audit rows: "
        "HI-Small, ICL-FS, seed 42, seven models. The sample was stratified by model "
        "and prediction outcome and underrepresents correct benign predictions. "
        "Rates describe this audit slice; they are not benchmark prevalence estimates, "
        "population-weighted error rates, or causal evidence about reasoning stages. "
        "No new model calls or semantic judgements were made."), "",
        "Reproduce all reports using only the released annotation CSVs and compact features:", "",
        "```sh", "python scripts/27_audit_scoring_sensitivity.py", "```", "",
        "Optionally recompute the compact features from the original local archive:", "",
        "```sh", "python scripts/27_audit_scoring_sensitivity.py --archive-root /path/to/outputs", "```", "",
        ("The archive must contain `<model>/LLM+ICL-AML/HI-Small/seed_42.json`. "
        "The script reads archived reasoning_content, maps v2_ case keys to amlc_ keys, "
        "and does not change trace text, final-answer parsing, annotations, or judge scores. "
        "No raw reasoning text is distributed here. SHA-256 hashes cover source archives, "
        "each full/prefix trace, annotation inputs, scoring definitions, source code, and features."), "",
        "## Definitions and denominators", "",
        ("- `strict_archived`: frozen heuristic Conclude column, denominator all rows in the group. "
        "Untyped illicit references are failures under that archived convention."),
        "- `binary`: extracted suspicious/benign verdict equals the binary label; denominator all group rows.",
        ("- `exclude_illicit_reference_absent`: correct benign verdict or correct typed illicit verdict; "
        "denominator benign plus typed illicit rows. Untyped illicit rows are excluded, not relabelled as correct."),
        ("- `typed_illicit_only`: suspicious verdict and exact canonical typology match; "
        "denominator typed illicit rows only, including binary misses. Typed correctness is undefined for "
        "benign or missing-reference rows; their false indicator is always masked by eligibility."),
        ("- `full` and `prefix8000` reuse the same original Parse, Recall, and Match functions. "
        "Prefix means the first 8,000 Python characters, matching the archived judge input limit. "
        "Final prediction fields are held fixed at both lengths. Empty traces fail all three lexical steps."),
        ("- `marker_*`, `marker_count`, and `marker_ge2` count the original six regex marker categories "
        "regardless of the verdict. These are lexical presence indicators; negated, hypothetical, or "
        "incorrect statements can match. They do not establish factual graph correctness or entailment."),
        ("- Original Match automatically passes a nonempty trace with a benign prediction. "
        "`benign_match_bypass` marks that branch, and `match_without_two_markers` counts passes with fewer "
        "than two marker categories. `PR_and_two_marker_categories` removes that vacuity."),
        ("- Original Recall counts matched surface strings, so synonyms can count separately. "
        "`canonical_recall` counts distinct canonical classes using the already defined "
        "`amlc.llm.methods.PATTERN_MAP` with case-insensitive substring matching and no new aliases. "
        "This diagnostic changes the lexicon as well as collapsing aliases (e.g. mule is present in "
        "PATTERN_MAP, cyclic is not). It is not a pure deduplication experiment or semantic Recall score. "
        "No judge or ground truth was used to choose the mapping. The final-answer parser remains frozen."),
        ("- A conjunction is the stated upstream lexical rule AND failure under the stated correctness "
        "definition. Its denominator is all eligible rows in that group, not only upstream passes or errors. "
        "`upstream_count` and `incorrect_count` give those alternate denominators explicitly. "
        "This co-occurrence is not an attribution of an error to Conclude or evidence of stage order."),
        ("- `judge_discordance.csv` is a 2-by-2 comparison of each archived API Conclude judgement "
        "with each fixed parser-based definition. `all_rows` retains the original convention: empty "
        "traces and recorded judge errors have all-failure labels, and stay in the denominator. "
        "`scored_only` excludes recorded error rows and gives their count. In these tables the 148 "
        "error rows for each judge all represent empty traces, not failed API calls. "
        "The archived prompts exposed missing illicit typology as the string nan; the sensitivity "
        "analysis changes eligibility masks only and cannot remove that influence from judge labels. "
        "Judge PRM/Conclude conjunctions are reported under each judge's own unchanged labels. "
        "Disagreement is not a corrected semantic accuracy estimate."),
        ("- Every rate CSV row includes its integer numerator and denominator. A zero denominator "
        "has an empty rate. `ALL` rows pool the selected audit cases without weighting. "
        "`outcome` preserves archived classes; `error_class` splits missing-reference flagged/missed cases "
        "from typology errors and under-predictions; `reference_group` separates reference availability. "
        "`binary_outcome` conditions on binary correctness (470 correct, 530 incorrect). "
        "Every grouping is also reported separately by model. These overlapping grouping tables must not be summed."), "",
        "## Overall checks", "",
        (f"Rows: {len(frame)}. Binary-correct: {int(frame.binary_correct.sum())}/{len(frame)}. "
        f"Typed-correct: {int(frame.typed_correct.sum())}/{int(frame.typed_eligible.sum())}. "
        f"Illicit reference absent: {int((~frame.reference_evaluable).sum())}/{len(frame)}. "
        f"Truncated at 8,000 characters: {int(frame.truncated.sum())}/{len(frame)}. "
        f"Empty reasoning traces: {int(frame.empty_trace.sum())}/{len(frame)}."), "",
        "| Window | Lexical rule | Correctness convention | Conjunction / eligible rows |", "|---|---|---|---|",
    ]
    for row in overall(tables["conjunctions"]).to_dict("records"):
        lines.append(f"| {row['window']} | {row['rule']} | {row['correctness']} | "
                     f"{row['count']}/{row['denominator']} |")
    lines += ["", "| Step | Full passes | Prefix passes | Full-only | Prefix-only |",
              "|---|---:|---:|---:|---:|"]
    for row in overall(tables["truncation"]).to_dict("records"):
        lines.append(f"| {row['metric']} | {row['full_count']} | {row['prefix8000_count']} | "
                     f"{row['full_only_count']} | {row['prefix_only_count']} |")
    lines += ["", "## Files", "",
              ("`inputs/trace_features.csv` and `inputs/provenance.json` are the portable archived-text "
              "derivatives and their provenance. `items.csv` joins those features to fixed labels, "
              "eligibility indicators, and reference groups. `rates.csv` contains outcome-conditioned "
              "lexical marker and correctness rates; `conjunctions.csv` contains joint descriptive "
              "counts; `truncation.csv` compares paired full/prefix scores; `judge_discordance.csv` "
              "contains parser/judge contingency counts. `analysis.json` records hashes, integrity "
              "checks, and overall tables. Portable reproduction verifies input hashes before reporting."), ""]
    return "\n".join(lines)
