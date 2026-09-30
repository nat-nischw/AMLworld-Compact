#!/usr/bin/env python3
"""Build an unevaluated target-preserving resource without changing old inputs.

Requires original AMLworld CSVs, archived JSON and typed-graph companions, and
the existing dataset release. No model calls, predictions, or scores are made.
The output directory must not exist; successful builds are published atomically.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa
import pyarrow.parquet as pq

from amlc.serialize.targeted_graph import (
    VERSION,
    ObservableTransaction,
    evaluation_prompt,
    preserve_target,
    recover_archived_transactions,
    serialise_targeted_graph,
)

CSV_HEADER = ["Timestamp", "From Bank", "Account", "To Bank", "Account",
              "Amount Received", "Receiving Currency", "Amount Paid",
              "Payment Currency", "Payment Format", "Is Laundering"]
DATASETS = ("HI-Small", "LI-Small")
SCHEMA = pa.schema([
    ("serialization_version", pa.string()), ("dataset", pa.string()),
    ("case_id", pa.string()),
    ("center_edge_id", pa.string()), ("subset_index", pa.int64()),
    ("ht_weight", pa.float64()), ("source_csv_row_zero_based", pa.int64()),
    ("target_source", pa.string()), ("target_destination", pa.string()),
    ("original_n_transactions", pa.int64()), ("n_transactions", pa.int64()),
    ("original_target_present", pa.bool_()), ("removed_edge_ids", pa.list_(pa.string())),
    ("n_tokens_approx", pa.int64()), ("targeted_graph_text", pa.string()),
    ("evaluation_prompt", pa.string()),
])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parquet_rows(path: Path):
    columns = ["case_id", "center_edge_id", "subset_index", "ht_weight",
               "n_transactions", "typed_graph_text"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
        yield from batch.to_pylist()


def load_targets(path: Path, indexes: list[dict]) -> tuple[dict, int]:
    wanted = {int(row["center_edge_id"].removeprefix("e_")) for row in indexes}
    if len(wanted) != len(indexes):
        raise ValueError("Duplicate source target rows in case index")
    targets = {}
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader) != CSV_HEADER:
            raise ValueError("Unexpected AMLworld CSV header")
        for row_index, row in enumerate(reader):
            count += 1
            if row_index in wanted:
                targets[f"e_{row_index}"] = ObservableTransaction.from_csv_row(row_index, row)
    if set(targets) != {row["center_edge_id"] for row in indexes}:
        raise ValueError("Some target source rows could not be resolved")
    test_start = int(count * (0.6 + 0.2))
    for row in indexes:
        if row["center_edge_id"] != f"e_{test_start + int(row['subset_index'])}":
            raise ValueError("Source row and published subset position disagree")
    return targets, count


def source_record(target: ObservableTransaction) -> dict:
    """The ten observable CSV fields, preserving original numeric strings."""
    src_bank, src_account = target.source.split("_", 1)
    dst_bank, dst_account = target.destination.split("_", 1)
    return {"timestamp": target.timestamp, "from_bank": src_bank,
            "from_account": src_account, "to_bank": dst_bank, "to_account": dst_account,
            "amount_received": target.amount_received,
            "receiving_currency": target.receiving_currency,
            "amount_paid": target.amount, "payment_currency": target.currency,
            "payment_format": target.payment_format}


def build_dataset(dataset: str, args, out: Path, analysis: Path) -> dict:
    index_path = args.dataset_release / "extras" / dataset / "case_index.csv"
    released_path = args.dataset_release / "data" / dataset / "test-00000-of-00001.parquet"
    archive_dir = args.archive_root / dataset / "v2_subset"
    json_path = archive_dir / "cases_json.jsonl"
    typed_path = archive_dir / "cases_paper_format.jsonl"
    csv_path = args.source_data / f"{dataset}_Trans.csv"
    sources = {
        f"dataset_release/extras/{dataset}/case_index.csv": index_path,
        f"dataset_release/data/{dataset}/test-00000-of-00001.parquet": released_path,
        f"archive/{dataset}/v2_subset/cases_json.jsonl": json_path,
        f"archive/{dataset}/v2_subset/cases_paper_format.jsonl": typed_path,
        f"AMLworld/{dataset}_Trans.csv": csv_path,
    }
    provenance = {logical: {"sha256": sha256(path), "bytes": path.stat().st_size}
                  for logical, path in sources.items()}
    with index_path.open(newline="", encoding="utf-8") as handle:
        indexes = list(csv.DictReader(handle))
    if len({row["case_id"] for row in indexes}) != len(indexes):
        raise ValueError("Duplicate case IDs")
    print(f"{dataset}: checking {len(indexes)} targets against original CSV", flush=True)
    targets, csv_count = load_targets(csv_path, indexes)
    destination = out / dataset
    destination.mkdir()
    output_path = destination / "test-00000-of-00001.parquet"
    mapping_path = analysis / f"{dataset}_case_mapping.jsonl.gz"
    summary = {"dataset": dataset, "cases": 0, "source_csv_rows": csv_count,
               "targets_present_in_archive": 0, "targets_restored": 0,
               "source_endpoints_absent_in_archive": 0,
               "destination_endpoints_absent_in_archive": 0,
               "cases_with_replaced_context_edge": 0,
               "original_transactions": 0, "new_transactions": 0,
               "validated_targets_against_original_csv": 0,
               "archive_companion_pairs_verified": 0,
               "released_historical_texts_verified": 0,
               "evaluation_prompts_validated": 0,
               "parquet_roundtrip_verified": 0,
               "original_to_new_mapping": str(mapping_path.name),
               "input_provenance": provenance}
    expected_output_hashes = []
    buffer = []
    with json_path.open(encoding="utf-8") as json_handle, \
            typed_path.open(encoding="utf-8") as typed_handle, \
            gzip.open(mapping_path, "wt", encoding="utf-8") as mapping_handle, \
            pq.ParquetWriter(output_path, SCHEMA, compression="zstd") as writer:
        rows = zip_longest(indexes, parquet_rows(released_path), json_handle, typed_handle)
        for index, released, json_line, typed_line in rows:
            if any(value is None for value in (index, released, json_line, typed_line)):
                raise ValueError("Case index, release and archives have different row counts")
            companion, archived = json.loads(json_line), json.loads(typed_line)
            cid, edge = index["case_id"], index["center_edge_id"]
            archived_cid = "v2_" + cid.removeprefix("amlc_")
            for record in (companion, archived):
                if (record["case_id"] != archived_cid or record["center_edge_id"] != edge
                        or record["subset_index"] != int(index["subset_index"])
                        or record["n_transactions"] != int(index["n_transactions"])):
                    raise ValueError("Archive-to-index mapping mismatch")
            if (released["case_id"] != cid or released["center_edge_id"] != edge
                    or released["subset_index"] != int(index["subset_index"])
                    or released["n_transactions"] != int(index["n_transactions"])):
                raise ValueError("Release-to-index mapping mismatch")
            canonical_old = archived["serialized_text"].replace(
                f"(Case: {archived_cid})", f"(Case: {cid})", 1)
            if canonical_old != released["typed_graph_text"]:
                raise ValueError("Released historical text differs from archived graph")
            original = recover_archived_transactions(
                companion["serialized_text"], archived["serialized_text"], archived_cid)
            if len(original) != int(index["n_transactions"]):
                raise ValueError("Extracted archive count differs from case index")
            target = targets[edge]
            graph = preserve_target(cid, original, target)
            graph_text = serialise_targeted_graph(graph)
            prompt = evaluation_prompt(graph, graph_text)
            original_nodes = {n for t in original for n in (t.source, t.destination)}
            if len(graph.transactions) != len(original):
                raise ValueError("Published transaction budget changed")
            record = {
                "serialization_version": VERSION, "dataset": dataset,
                "case_id": cid, "center_edge_id": edge,
                "subset_index": int(index["subset_index"]),
                "ht_weight": released["ht_weight"],
                "source_csv_row_zero_based": int(edge[2:]),
                "target_source": target.source, "target_destination": target.destination,
                "original_n_transactions": len(original), "n_transactions": len(graph.transactions),
                "original_target_present": graph.target_was_present,
                "removed_edge_ids": list(graph.removed_edge_ids),
                "n_tokens_approx": max(1, len(prompt) // 4),
                "targeted_graph_text": graph_text, "evaluation_prompt": prompt,
            }
            buffer.append(record)
            expected_output_hashes.append((cid, text_sha256(graph_text), text_sha256(prompt)))
            if len(buffer) >= 32:
                writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                buffer.clear()
            mapping = {
                "dataset": dataset, "case_id": cid,
                "center_edge_id": edge, "subset_index": record["subset_index"],
                "source_csv_row_zero_based": int(edge[2:]),
                "original_source_record": source_record(target),
                "original_target_present": graph.target_was_present,
                "original_source_endpoint_present": target.source in original_nodes,
                "original_destination_endpoint_present": target.destination in original_nodes,
                "original_n_transactions": len(original), "new_n_transactions": len(graph.transactions),
                "original_graph_char_count": len(canonical_old),
                "new_graph_char_count": len(graph_text),
                "original_edge_ids": [t.edge_id for t in original],
                "new_edge_ids": [t.edge_id for t in graph.transactions],
                "removed_edge_ids": list(graph.removed_edge_ids),
                "added_edge_ids": [] if graph.target_was_present else [edge],
                "archived_typed_text_sha256": text_sha256(archived["serialized_text"]),
                "released_historical_text_sha256": text_sha256(canonical_old),
                "targeted_graph_text_sha256": text_sha256(graph_text),
                "evaluation_prompt_sha256": text_sha256(prompt),
            }
            mapping_handle.write(json.dumps(mapping, separators=(",", ":")) + "\n")
            summary["cases"] += 1
            summary["targets_present_in_archive"] += int(graph.target_was_present)
            summary["targets_restored"] += int(not graph.target_was_present)
            summary["source_endpoints_absent_in_archive"] += int(target.source not in original_nodes)
            summary["destination_endpoints_absent_in_archive"] += int(target.destination not in original_nodes)
            summary["cases_with_replaced_context_edge"] += int(bool(graph.removed_edge_ids))
            summary["original_transactions"] += len(original)
            summary["new_transactions"] += len(graph.transactions)
            for key in ("validated_targets_against_original_csv", "archive_companion_pairs_verified",
                        "released_historical_texts_verified", "evaluation_prompts_validated"):
                summary[key] += 1
            if summary["cases"] % 500 == 0:
                print(f"{dataset}: verified and rendered {summary['cases']} cases", flush=True)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
    written = pq.ParquetFile(output_path)
    if written.schema_arrow != SCHEMA:
        raise ValueError("Written parquet schema mismatch")
    offset = 0
    for batch in written.iter_batches(batch_size=16):
        for row in batch.to_pylist():
            cid, graph_hash, prompt_hash = expected_output_hashes[offset]
            if (cid != row["case_id"] or graph_hash != text_sha256(row["targeted_graph_text"])
                    or prompt_hash != text_sha256(row["evaluation_prompt"])):
                raise ValueError("Parquet prompt roundtrip mismatch")
            offset += 1
    if offset != len(indexes):
        raise ValueError("Parquet output row count mismatch")
    summary["parquet_roundtrip_verified"] = offset
    for logical, path in sources.items():
        if sha256(path) != provenance[logical]["sha256"]:
            raise ValueError("Historical input changed while building resource")
    summary["historical_inputs_unchanged"] = True
    summary["output"] = {"path": f"{dataset}/test-00000-of-00001.parquet",
                         "sha256": sha256(output_path), "bytes": output_path.stat().st_size}
    summary["mapping"] = {"path": mapping_path.name, "sha256": sha256(mapping_path),
                          "bytes": mapping_path.stat().st_size}
    shutil.copy2(mapping_path, out / mapping_path.name)
    print(f"{dataset}: complete; {summary['targets_restored']} targets restored", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True,
                        help="Directory containing original HI/LI-Small_Trans.csv")
    parser.add_argument("--archive-root", type=Path, required=True,
                        help="Archive containing DATASET/v2_subset/cases_{json,paper_format}.jsonl")
    parser.add_argument("--dataset-release", type=Path,
                        default=package_root.parent / "amlcompact-dataset")
    parser.add_argument("--out", type=Path,
                        help="New output directory; default dataset release extras/targeted_prompts_v1")
    parser.add_argument("--analysis-out", type=Path,
                        default=package_root / "results" / "analysis" / "target_integrity")
    args = parser.parse_args(argv)
    args.out = args.out or args.dataset_release / "extras" / "targeted_prompts_v1"
    for path in (args.out, args.analysis_out):
        if path.exists():
            parser.error(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".targeted-prompts-", dir=args.out.parent) as temp_out, \
            tempfile.TemporaryDirectory(prefix=".target-integrity-", dir=args.analysis_out.parent) as temp_analysis:
        out, analysis = Path(temp_out), Path(temp_analysis)
        reports = [build_dataset(dataset, args, out, analysis) for dataset in DATASETS]
        total = sum(row["cases"] for row in reports)
        if total != 6021:
            raise ValueError(f"Expected the 6,021 published cases; found {total}")
        integrity_columns = ["dataset", "case_id", "target_edge_id",
                             "subset_index", "original_target_present",
                             "original_transaction_count", "original_graph_char_count",
                             "original_source_endpoint_present", "original_destination_endpoint_present",
                             "new_transaction_count", "removed_edge_id", "source_target_verified",
                             "unique_target_verified", "endpoints_verified", "duration_verified",
                             "label_fields_absent", "placeholders_absent"]
        with (analysis / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=integrity_columns)
            writer.writeheader()
            for dataset in DATASETS:
                with gzip.open(analysis / f"{dataset}_case_mapping.jsonl.gz", "rt", encoding="utf-8") as mappings:
                    for line in mappings:
                        mapping = json.loads(line)
                        row = {key: mapping[key] for key in integrity_columns if key in mapping}
                        row.update(target_edge_id=mapping["center_edge_id"],
                                   original_transaction_count=mapping["original_n_transactions"],
                                   new_transaction_count=mapping["new_n_transactions"],
                                   removed_edge_id=";".join(mapping["removed_edge_ids"]))
                        for key in ("source_target_verified", "unique_target_verified", "endpoints_verified",
                                    "duration_verified", "label_fields_absent", "placeholders_absent"):
                            row[key] = True
                        writer.writerow(row)
        shutil.copy2(analysis / "cases.csv", out / "cases.csv")
        report = {
            "serialization_version": VERSION, "status": "validated_unevaluated_resource",
            "model_calls": 0, "new_predictions": 0, "historical_scores_recomputed": False,
            "total_cases": total, "datasets": reports,
            "case_integrity_csv": "cases.csv",
            "transaction_budget_policy": "Retain original edge count. Restore a missing target by replacing the latest non-target edge; break timestamp ties by numeric edge ID. Retain both target endpoint accounts.",
            "account_identity": "bank_account; bank prefixes are retained to disambiguate account nodes",
            "target_fields": "All ten observable original CSV fields are included; binary outcome and typology are excluded.",
            "timestamp_semantics": "Timezone-unspecified AMLworld local timestamps parsed as naive datetimes. Start, end, and elapsed seconds refer only to the included edges, not the entire source graph.",
            "prompt_scope": "New standalone zero-shot instruction plus targeted graph; historical runners and old predictions remain tied to archived prompts.",
            "labels": "No ground-truth columns in this resource. Join original labels separately using dataset and case_id; never add them to evaluation_prompt.",
            "limitations": [
                "Not a rerun or re-evaluation of historical predictions; no performance claim is supported for these inputs.",
                "Archived context may contain future transactions; this version does not repair temporal leakage or make the file-order split chronological.",
                "The original edge count is preserved, not the original token budget; target details and identifiers increase text length.",
                "All target attributes are verified against the original CSV. Non-target edges are verified between archived companion formats, not independently rechecked against every source CSV row.",
                "A cap of one permits only the target edge. Both endpoints are always retained; a self-transfer has one distinct endpoint account.",
            ],
            "schema": [{"name": field.name, "type": str(field.type)} for field in SCHEMA],
        }
        write_json(out / "manifest.json", report)
        write_json(analysis / "validation_report.json", report)
        (out / "schema.json").write_text(json.dumps(report["schema"], indent=2) + "\n")
        # Both final paths were required to be new. No historical file is a write target.
        out.rename(args.out)
        analysis.rename(args.analysis_out)
    print(json.dumps({"status": "validated", "cases": total,
                      "targets_restored": sum(row["targets_restored"] for row in reports),
                      "model_calls": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
