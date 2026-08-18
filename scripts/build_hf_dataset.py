#!/usr/bin/env python3
"""Build the Hugging Face dataset repo from the archived run directory.

Produces the released evaluation set as parquet, plus the arrays and the
supervised-baseline artefacts that go with it.

    data/<dataset>/test-*.parquet   one row per case, with the typed k-hop
                                    prompt the LLMs were actually given
    extras/<dataset>/               subset indices, HT weights, labels,
                                    typologies, GFP tensors, case index
    ml_baselines/weights/           LightGBM and XGBoost boosters, per seed
    ml_baselines/test_probs/        full-test-split probabilities, 3 members

Weights come from ``data/coreset/<dataset>/ht_weights.npy`` in this repo, never
from the archived column, which double-counts the LI-Small benign population.
The build refuses to write unless they sum to the full test-split size.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from amlc.case_ids import case_id, retag
from amlc.coreset.ht_weights import repair_archived_weights
from amlc.config import DATASETS, ENSEMBLE_MEMBERS, N_TEST_FULL, SEEDS
from amlc.archive import (
    SERIALISED_DIR, legacy_path, member_dir, member_weights_dir,
)
from amlc.typology import TYPOLOGY_TO_IDX



# Only the members the paper reports. PNA appears in no table and is not shipped.
#
# The two boosters keep their weights under saved_models/. GCPAL does not: the
# reported member GCPAL_knn_temporal is not separately trained, it is the
# GCPAL_knn checkpoints re-inferred on the temporal split by
# gcpal_infer_temporal.py. So its weights live under models/<ds>/GCPAL_knn/ and
# its probabilities under test_probs/<ds>/GCPAL_knn_temporal/.
ML_MEMBERS = {m: member_dir(m) for m in ENSEMBLE_MEMBERS}

#: paper name -> (directory under outputs/, subdirectory holding the weights)
ML_WEIGHT_SOURCE = {m: member_weights_dir(m) for m in ML_MEMBERS}

# One file per config: 20 MB and 13 MB after zstd, far below any sharding
# threshold, and a single file is what the viewer handles most simply.
SHARD_ROWS = 100_000

# Row group size drives the dataset viewer. The prompt column averages ~76 KB
# per row, so the default of one row group per file made the viewer decompress
# 150 MB to render the first screen. At 100 rows a group is 7.8 MB, the first
# 100 rows read 7.7x faster (138 ms -> 18 ms), and the file grows 1.0%.
ROW_GROUP_ROWS = 100


def build_split(archive: Path, dataset: str, out: Path) -> dict:
    # Weights are recomputed from the archive rather than read from a vendored
    # copy: the archived LI-Small vector double-counts the benign population,
    # and there is no second place for the corrected one to go stale.
    repair = repair_archived_weights(archive, dataset, "ht-coreset")
    idx, weights = repair["subset_idx"], repair["fixed"]
    labels = np.load(legacy_path(archive, "test_labels", dataset))[idx]
    typologies = np.load(legacy_path(archive, "test_typologies", dataset))[idx]
    index = pd.read_csv(archive / "llm_datasets" / dataset / SERIALISED_DIR
                        / "case_index.csv")
    index = index.drop(columns=[c for c in ("legacy_case_id",) if c in index.columns])
    index["case_id"] = [case_id(i) for i in range(len(index))]

    if not np.isclose(weights.sum(), N_TEST_FULL[dataset], rtol=0, atol=1e-6):
        raise SystemExit(
            f"{dataset}: HT weights sum to {weights.sum():,.1f}, expected "
            f"{N_TEST_FULL[dataset]:,}. Refusing to publish."
        )

    # The archived case index carries its own copy of the weight column, written
    # before the LI-Small spare-fill double-count was found. Checking the repaired
    # vector is not enough, because that copy rides along untouched: the first
    # release shipped an LI-Small case_index.csv summing to 2,767,353 against a
    # population of 1,384,810 while ht_weights.npy beside it was correct. Every
    # column here that duplicates a shipped array is now overwritten from that
    # array and then checked, so there is one source per quantity and no second
    # place for one to go stale.
    duplicated = {"subset_index": idx, "label": labels, "weight": weights}
    for column, canonical in duplicated.items():
        if column in index.columns:
            index[column] = canonical
    if len(index) != len(idx):
        raise SystemExit(
            f"{dataset}: case index has {len(index):,} rows, coreset has {len(idx):,}"
        )
    for column, canonical in duplicated.items():
        if column in index.columns and not np.array_equal(
            index[column].to_numpy(), np.asarray(canonical)
        ):
            raise SystemExit(f"{dataset}: case_index.{column} disagrees with the shipped array")
    if not np.isclose(index["weight"].sum(), N_TEST_FULL[dataset], rtol=0, atol=1e-6):
        raise SystemExit(
            f"{dataset}: case_index weights sum to {index['weight'].sum():,.1f}, "
            f"expected {N_TEST_FULL[dataset]:,}. Refusing to publish."
        )

    jsonl = (archive / "llm_datasets" / dataset / SERIALISED_DIR
             / "cases_paper_format.jsonl")
    rows = [json.loads(line) for line in jsonl.open()]
    if len(rows) != len(idx):
        raise SystemExit(f"{dataset}: {len(rows)} serialised cases vs {len(idx)} coreset rows")

    # The prompt header embeds the case identifier. It is retagged so the whole
    # release speaks one vocabulary; see amlc.case_ids for why that is
    # safe and what it costs.
    prompts, n_retag = [], 0
    for r in rows:
        t, k = retag(r["serialized_text"])
        prompts.append(t)
        n_retag += k

    df = pd.DataFrame({
        "case_id": [case_id(i) for i in range(len(rows))],
        "center_edge_id": [r["center_edge_id"] for r in rows],
        "subset_index": idx.astype("int64"),
        "label": labels.astype("int8"),
        "illicit": labels.astype(bool),
        "typology": [None if t < 0 else
                     [k for k, v in TYPOLOGY_TO_IDX.items() if v == t][0]
                     for t in typologies],
        "typology_id": typologies.astype("int8"),
        "ht_weight": weights.astype("float64"),
        "n_transactions": [r["n_transactions"] for r in rows],
        "n_tokens_approx": [r["n_tokens_approx"] for r in rows],
        "typed_graph_text": prompts,
    })

    # Cross-checks before anything is written.
    assert (df.subset_index.to_numpy() == idx).all()
    assert (df.label.to_numpy() == labels).all()
    assert np.array_equal(df.ht_weight.to_numpy(), weights)
    assert list(df.case_id) == list(index.case_id)
    assert n_retag == len(df), f"expected one identifier per prompt, retagged {n_retag}"
    assert not any("v2_" in t for t in df.typed_graph_text)
    assert (df.typology.isna().to_numpy() == (typologies < 0)).all()

    split_dir = out / "data" / dataset
    split_dir.mkdir(parents=True, exist_ok=True)
    for f in split_dir.glob("*.parquet"):
        f.unlink()
    n_shards = max(1, -(-len(df) // SHARD_ROWS))
    for s in range(n_shards):
        chunk = df.iloc[s * SHARD_ROWS:(s + 1) * SHARD_ROWS]
        pq.write_table(
            pa.Table.from_pandas(chunk, preserve_index=False),
            split_dir / f"test-{s:05d}-of-{n_shards:05d}.parquet",
            compression="zstd", compression_level=12,
            row_group_size=ROW_GROUP_ROWS,
        )

    # The ensemble probability per row, so the headline supervised number can
    # be reproduced without pulling the 458 MB of full-split probabilities.
    probs = [
        np.load(legacy_path(archive, "member_probs", dataset,
                            member=member_dir(m), seed=seed))[idx]
        for m in ENSEMBLE_MEMBERS for seed in SEEDS
        if legacy_path(archive, "member_probs", dataset,
                       member=member_dir(m), seed=seed).exists()
    ]
    ensemble = np.mean(probs, axis=0)

    extras = out / "extras" / dataset
    extras.mkdir(parents=True, exist_ok=True)
    np.save(extras / "ensemble_probs_coreset.npy", ensemble)
    np.save(extras / "ht_subset_indices.npy", idx)
    np.save(extras / "ht_weights.npy", weights)
    np.save(extras / "labels.npy", labels)
    np.save(extras / "typologies.npy", typologies)
    index.to_parquet(extras / "case_index.parquet", index=False)
    shutil.copy2(
        archive / "llm_datasets" / dataset / SERIALISED_DIR / "features_gfp.npz",
        extras / "features_gfp.npz")

    size = sum(f.stat().st_size for f in split_dir.glob("*.parquet"))
    return {"dataset": dataset, "rows": len(df), "shards": n_shards,
            "parquet_bytes": size, "jsonl_bytes": jsonl.stat().st_size,
            "weights_sum": float(weights.sum()), "prompts_retagged": n_retag,
            "illicit": int(labels.sum()), "typed": int((typologies >= 0).sum())}


def copy_ml_baselines(archive: Path, out: Path) -> dict:
    w_out = out / "ml_baselines" / "weights"
    p_out = out / "ml_baselines" / "test_probs"
    n_w = n_p = 0
    for ds in DATASETS:
        for paper_name, dirname in ML_MEMBERS.items():
            root, sub = ML_WEIGHT_SOURCE[paper_name]
            srcw = archive / root / ds / sub
            if srcw.is_dir():
                dst = w_out / ds / paper_name
                dst.mkdir(parents=True, exist_ok=True)
                for f in srcw.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
                        n_w += 1
            srcp = archive / "test_probs" / ds / dirname
            if srcp.is_dir():
                dst = p_out / ds / paper_name
                dst.mkdir(parents=True, exist_ok=True)
                for seed in SEEDS:
                    for suffix in ("", "_typ"):
                        f = srcp / f"seed_{seed}{suffix}.npy"
                        if f.exists():
                            shutil.copy2(f, dst / f.name)
                            n_p += 1
        for name in ("test_labels.npy", "test_typologies.npy"):
            f = archive / "test_probs" / ds / name
            if f.exists():
                (p_out / ds).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, p_out / ds / name)
                n_p += 1
    return {"weight_files": n_w, "prob_files": n_p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-ml", action="store_true")
    args = ap.parse_args()

    report = [build_split(args.archive, ds, args.out) for ds in DATASETS]
    for r in report:
        print(f"{r['dataset']}: {r['rows']} rows in {r['shards']} shard(s), "
              f"{r['illicit']} illicit, {r['typed']} typed, "
              f"sum(w)={r['weights_sum']:,.0f}, "
              f"parquet {r['parquet_bytes']/1e6:.1f} MB "
              f"from {r['jsonl_bytes']/1e6:.1f} MB jsonl "
              f"({r['parquet_bytes']/r['jsonl_bytes']*100:.1f}%)")

    if not args.skip_ml:
        ml = copy_ml_baselines(args.archive, args.out)
        print(f"ml_baselines: {ml['weight_files']} weight files, "
              f"{ml['prob_files']} probability files")

    (args.out / "build_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
