# AMLworld-Compact

[![CI](https://github.com/nat-nischw/AMLCompact/actions/workflows/ci.yml/badge.svg)](https://github.com/nat-nischw/AMLCompact/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-AMLworldCompactEval-yellow)](https://huggingface.co/datasets/natnitaract/AMLworldCompactEval)
[![Code MIT](https://img.shields.io/badge/code-MIT-blue)](LICENSE)

[Install](#install) · [Evaluation](#evaluation) · [Baselines](#baselines) ·
[Dataset](#dataset) · [Citation](#citation)

AMLworld-Compact provides importance-weighted subsets of AMLworld for evaluating
transaction classification and laundering-typology prediction. It retains all
illicit test edges and samples benign edges, reducing HI-Small and LI-Small by
271× and 611×. Each retained edge has graph features, a serialised local graph,
and an inverse-inclusion-probability weight.

Code and evaluation tools for *AMLworld-Compact: Importance-Weighted Downsampling
for LLM Evaluation and Error Diagnosis*. The Python package is `amlc`; the dataset
is [`natnitaract/AMLworldCompactEval`](https://huggingface.co/datasets/natnitaract/AMLworldCompactEval).

## Install

Use Python 3.11 or later. Run commands from the code repository root:

```bash
git clone https://github.com/nat-nischw/AMLCompact.git
cd AMLCompact
python -m pip install -e ".[hub]"
```

This installs the dataset loaders and scoring dependencies. To generate predictions
through an existing vLLM endpoint, also install the LLM client dependencies:

```bash
python -m pip install -e ".[llm]"
```

## Evaluation

### Score the released supervised ensemble

The following example loads the evaluation arrays and scores the released ensemble
without training or generating new predictions. The first Hub load downloads the
arrays; subsequent loads use the cache.

The dataset is currently private. Set `HF_TOKEN` in your environment using a
Hugging Face token with read access before loading it from the Hub.

<!--quick-start-begin-->
```python
import numpy as np
from amlc import hub
from amlc.triage.doubt_triage import ht_weighted_prf

d = hub.load_coreset("HI-Small")
predictions = (d["ensemble_probs"] >= d["ml_threshold"]).astype(int)

for name, weights in [
    ("HT-weighted", d["weights"]),
    ("Compact", np.ones(d["n"])),
]:
    precision, recall, f1 = ht_weighted_prf(predictions, d["labels"], weights)
    print(f"{name}: P={100 * precision:.4f}% R={100 * recall:.4f}% F1={100 * f1:.4f}%")
```
<!--quick-start-end-->

For your own detector, replace `predictions` with a one-dimensional binary array:
`1` means illicit and `0` means benign. It must contain exactly one prediction per
released row, in the same order as `d["labels"]`. Join unordered outputs to the
`case_id` column of `hub.load_table("HI-Small")` before scoring. If probabilities
cover the full temporal test split, select `probabilities[d["subset_idx"]]` first,
then apply your chosen decision threshold.

To check both released ensembles and their weight sums:

```bash
python -m amlc.selftest
```

With a local copy of the dataset, the array loader can run offline. Point it at
`extras/`, which contains one directory per split:

```bash
AMLC_CORESET_DIR=/path/to/amlcompact-dataset/extras python -m amlc.selftest
```

`AMLC_CORESET_DIR` overrides array loading; `hub.load_table()` still reads the Hub
Parquet table. To load local prompts, use `pandas.read_parquet()` on
`data/HI-Small/test-00000-of-00001.parquet` in the dataset copy.

### Generate LLM predictions

Use an existing vLLM server with a served model name matching a runner ID in
[Baselines](#baselines). The hosting configurations used for the released runs are
in [`scripts/slurm/host_vllm/`](scripts/slurm/host_vllm/). Set `--model-id` if your
endpoint exposes a different served name.

```bash
python scripts/13_run_llm_eval.py \
  --model GPT-OSS-20B \
  --vllm-url http://localhost:8000/v1 \
  --datasets HI-Small LI-Small \
  --promptings ICL-ZS ICL-FS \
  --seeds 42 123 456 789 1011 \
  --workers 16 \
  --out runs/gpt-oss-20b
```

`ICL-ZS` uses the task instructions without demonstrations; `ICL-FS` adds the
released demonstration pool. `ICL-V` is an additional supported condition and is
not included in the baseline table below. The coreset stays fixed across seeds;
the runner sends each requested sampling seed to vLLM. Server versions and
batching can still affect reproducibility.
Use fewer values in `--datasets`, `--promptings`, or `--seeds` to evaluate a smaller
configuration. Each selected combination evaluates every row in that split.

The runner saves per-case verdicts, typologies, responses, available reasoning,
and token counts in per-seed JSON files under `--out`. It also writes
`metrics.json` beside those files and `summary_all.csv` / `summary_all.json` at
the output root. **The runner's detection metrics are unweighted compact-set
metrics.** Run the next step to obtain HT-weighted metrics.

### Score saved LLM predictions

This step reads predictions already saved by the runner and the released coreset
arrays. It does not call an LLM:

```bash
python scripts/23_score_llm_ht.py \
  --runs-dir runs/gpt-oss-20b \
  --models GPT-OSS-20B \
  --promptings ICL-ZS ICL-FS \
  --seeds 42 123 456 789 1011 \
  --dataset HI-Small --dataset LI-Small \
  --out runs/gpt-oss-20b/evaluation
```

Set `AMLC_CORESET_DIR` as above to score with local arrays. Match the model,
prompting, dataset, and seed selections to the prediction files you generated.

| Output in the evaluation directory | Contents |
|---|---|
| `llm_ht_weighted.csv` | One row per dataset/model/prompting/seed with both `ht_*` and `subset_*` detection metrics; also includes predict-all-illicit reference rows. |
| `llm_ht_weighted_summary.csv` | Ranges over model/prompting means within each dataset and the predict-all-illicit reference. |

### Metrics and reporting

For binary labels `y`, predictions `p`, and weights `w`, the scorer computes
`TP = sum(w * (y == 1) * (p == 1))`, with analogous weighted FP and FN counts.
Precision is `TP / (TP + FP)`, recall is `TP / (TP + FN)`, and F1 is
`2 * TP / (2 * TP + FP + FN)`. A zero denominator returns zero.

| Metric | Population / denominator | Where to find it |
|---|---|---|
| HT-weighted detection precision, recall, F1 | Released coreset with `ht_weight`; estimates the full temporal test split. | `ht_p`, `ht_r`, `ht_f1` |
| Compact detection precision, recall, F1 | Every retained row has weight one; describes the 1:2 compact-set class ratio. | `subset_p`, `subset_r`, `subset_f1`; runner detection columns |
| Detection accuracy | Fraction of all compact-set rows classified correctly. | Runner `detection_accuracy` |
| Typology macro-F1 | Unweighted mean of class F1 values on illicit rows with a known ground-truth typology. The runner averages over labels present in the ground truth or predictions, including `none` for missing predictions. | Runner `typology_macro_f1` |
| Typology accuracy | Fraction of labelled illicit rows whose predicted typology matches the ground truth, scored independently of the detection verdict. | Runner `typology_accuracy` |
| Valid-response ratio | Percentage of attempted cases with recorded token usage; this is not classification accuracy. | Runner `valid_ratio` |

`ht_weighted_prf()` returns fractions in `[0, 1]`. Detection and typology scores in
the stage-23 and runner summary CSV files are percentages in `[0, 100]`; their
values in per-seed metric JSON are fractions. Counts, token usage, and latency
retain their own units; `token_stats.valid_ratio` is already a percentage.
Runner seed summaries report mean and standard deviation (`ddof=0`),
with zero standard deviation for a single run. The stage-23 summary reports ranges
of seed means, not confidence intervals. Evidence/verifier fields are not computed
by the standard ICL evaluation.

Report HT-weighted detection metrics as the primary full-split estimates, and label
compact metrics separately. Predicting every edge illicit gives compact precision
33.333% and F1 50.000%; after weighting, its F1 is 0.246% on HI-Small and 0.109% on
LI-Small. These are different evaluation distributions.

With fixed per-edge predictions and contexts, positive known inclusion
probabilities make HT confusion counts unbiased in expectation. Precision and F1
are ratios and need not be unbiased in finite samples. Recall is exact for the
fixed predictor because every illicit edge is retained with weight one. Variation
across inference seeds does not measure variation across newly sampled coresets.

## Baselines

### Supervised models

GFP means **Graph Feature Preprocessor**. The primary ensemble averages
LightGBM and XGBoost probabilities across five seeds. It retains thresholds
0.80 for HI-Small and 0.48 for LI-Small from the original construction study;
these operating points were selected on the full test split.

| Full baseline name | Loader ID |
|---|---|
| LightGBM + Graph Feature Preprocessor | `LightGBM+GFP` |
| XGBoost + Graph Feature Preprocessor | `XGBoost+GFP` |

The released ensemble's HT-weighted P / R / F1 (%) is
**84.8449 / 56.8345 / 68.0708** on HI-Small and
**60.1695 / 18.7831 / 28.6290** on LI-Small. Its full-split and coreset values
coincide because all edges contributing to its TP, FP, and FN counts are retained
with weight one. This exact equality does not extend to arbitrary predictors.

The frozen sampling design originally used a third scorer, **Graph Contrastive
Pre-training for Anti-money Laundering + GFP (GCPAL+GFP)**. Its random fine-tuning
split overlaps the temporal test set. We preserve that construction history and
its original inclusion weights, while the primary evaluation ensemble uses only
the two temporally trained boosters. GCPAL checkpoints remain available as
construction assets.

Checkpoints and full-test probabilities are available through `hub.load_ml_weights()`
and `hub.load_test_probs()`. Use `d["subset_idx"]` to align full-test predictions
with coreset rows.

### Language models

| Model | Runner ID (`--model`) | Checkpoint |
|---|---|---|
| OpenAI GPT-OSS-20B | `GPT-OSS-20B` | `openai/gpt-oss-20b` |
| Qwen3.5-27B | `Qwen3.5-27B` | `Qwen/Qwen3.5-27B-FP8` |
| NVIDIA Nemotron-3-Nano-30B-A3B | `Nemotron-3-Nano-30B` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |
| Qwen3.5-35B-A3B | `Qwen3.5-35B-A3B` | `Qwen/Qwen3.5-35B-A3B-FP8` |
| OpenAI GPT-OSS-120B | `GPT-OSS-120B` | `openai/gpt-oss-120b` |
| NVIDIA Nemotron-3-Super-120B-A12B | `Nemotron-3-Super-120B` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` |
| Qwen3.5-397B-A17B | `Qwen3.5-397B-A17B` | `Qwen/Qwen3.5-397B-A17B-FP8` |

The following values are **HT-weighted detection F1 (%)**, averaged over five seeds
for each model and prompting condition. ZS means `ICL-ZS`; FS means `ICL-FS`.

| Model | HI-Small ZS | HI-Small FS | LI-Small ZS | LI-Small FS |
|---|---:|---:|---:|---:|
| OpenAI GPT-OSS-20B | 0.246 | 0.236 | 0.109 | 0.098 |
| Qwen3.5-27B | 0.230 | 0.240 | 0.088 | 0.096 |
| NVIDIA Nemotron-3-Nano-30B-A3B | 0.246 | 0.245 | 0.109 | 0.108 |
| Qwen3.5-35B-A3B | 0.246 | 0.253 | 0.109 | 0.107 |
| OpenAI GPT-OSS-120B | 0.246 | 0.249 | 0.109 | 0.108 |
| NVIDIA Nemotron-3-Super-120B-A12B | 0.247 | 0.247 | 0.110 | 0.109 |
| Qwen3.5-397B-A17B | 0.248 | 0.251 | 0.110 | 0.110 |

Source: [`results/analysis/llm_ht_weighted.csv`](results/analysis/llm_ht_weighted.csv).
These values are estimates at native test-split prevalence. Compact-set metrics
are available as `subset_*` columns in the same file.

## Dataset

| | HI-Small | LI-Small |
|---|---:|---:|
| Full temporal test edges | 1,015,669 | 1,384,810 |
| Illicit edges retained | 1,251 | 756 |
| Compact evaluation rows | 3,753 | 2,268 |
| Illicit rows with a known typology | 791 | 174 |

The [dataset card](https://huggingface.co/datasets/natnitaract/AMLworldCompactEval)
describes all columns, feature arrays, weights, and checkpoint files. Graph texts
use two-hop neighbourhoods with a cap of 50 neighbours per hop. The released data
contains test rows; use separate data when training and tuning a new model.

## Repository layout

| Path | Contents |
|---|---|
| `amlc/` | Data loaders, coreset construction, baseline models, evaluation, triage, and audit code |
| `scripts/` | Command-line entry points and optional cluster configurations |
| `prompts/` | Task, demonstration, verification, and audit templates |
| `data/` | Tuned hyperparameters and demonstration pools |
| `results/` | Released metrics and analysis outputs |

Run `make help` to list the available commands. For local checks, run
`make install-dev`, then `make test` and `make lint`.
The released detailed reasoning audit uses the available seed-42 traces; task
metrics use five inference seeds.

## Uses

This synthetic benchmark supports research on transaction classification,
graph-to-text prompting, and error analysis. Use the released test split for
evaluation and separate data for training and tuning. Results describe the
released AMLworld splits and do not establish performance on real banking
transactions.

## Licences

Code is [MIT](LICENSE). AMLworld-derived evaluation data is CDLA-Sharing-1.0;
released supervised model parameters and outputs are labelled MIT in the dataset
release. The Elliptic pipeline is provided without derived Elliptic data. See
[`NOTICE.md`](NOTICE.md) and the dataset's licence files for attribution and
asset-specific terms.

## Citation

```bibtex
@misc{nitarach2026amlcompact,
  title  = {AMLworld-Compact: Importance-Weighted Downsampling for LLM Evaluation and Error Diagnosis},
  author = {Nitarach, Natapong and Ngampornsukswadi, Phume and
            Taveekitworachai, Pittawat and Nonesung, Surapon and
            Sirichotedumrong, Warit and Halverson, Duncan and
            Pipatanakul, Kunat},
  year   = {2026},
  url    = {https://openreview.net/forum?id=VouFZFf8Ph}
}
```

Please also cite AMLworld, the source dataset:

```bibtex
@inproceedings{altman2023realistic,
  title     = {Realistic Synthetic Financial Transactions for Anti-Money Laundering Models},
  author    = {Altman, Erik and Blanu{\v{s}}a, Jovan and von Niederh{\"a}usern, Luc and
               Egressy, B{\'e}ni and Anghel, Andreea and Atasu, Kubilay},
  booktitle = {Advances in Neural Information Processing Systems 36,
               Datasets and Benchmarks Track},
  year      = {2023}
}
```

Paper metadata is maintained in [CITATION.cff](CITATION.cff). Run
`make check-citation` to check the README citation against that file.
