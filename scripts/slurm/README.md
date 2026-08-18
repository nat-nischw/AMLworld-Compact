# Cluster jobs

Numbered to match `scripts/`. Every credential and node name is an environment
variable; see `.env.example`. `#SBATCH` directives cannot expand shell variables,
so pass `--nodelist` and the log paths on the `sbatch` command line.

| script | stage |
|---|---|
| `03_train_ml_baselines.sh` | LightGBM and XGBoost, tune then train, 5 seeds |
| `04_train_gcpal.sh` | GCPAL, the configuration that produced the released checkpoints |
| `05_infer_gcpal_temporal.sh` | re-infer GCPAL on the temporal split |
| `07_build_ht_coreset.sh` | HT-Coreset construction |
| `07b_build_naive_coreset.sh` | Naive Coreset, only needed for the Table 10 column |
| `08_run_ablation.sh` | the seven sampling baselines |
| `09_compare_naive_vs_ht.sh` | Naive versus HT per model |
| `11_serialize_coreset.sh` | serialise the coreset into typed-graph prompts |
| `13_run_llm_eval.sh` | one LLM over the coreset, all promptings and seeds |
| `14_run_intervention.sh` | the ICL-V prompt intervention |
| `15_rerun_thinking.sh` | re-run cases whose reasoning trace was truncated |
| `17_run_doubt_triage.sh` | Doubt Triage over every cell |
| `18_score_deferral_grid.sh` | the score-only deferral baseline of Appendix G.3 |
| `20_elliptic_pipeline.sh` | Elliptic download, ML, coreset, evaluation |
| `21_elliptic_llm_eval.sh` | Elliptic LLM evaluation (never run for the paper) |
| `check_traces.py` | trace completeness check |

`host_vllm/` holds one script per evaluated model, plus the Nemotron environment
setup and a cache cleaner for swapping models on a node.

## Removed from the release

| script | why |
|---|---|
| `host_vllm1.sh` | generic host script, superseded and carried a W&B key line |
| `host_vllm2.sh` | generic host script, superseded and carried a W&B key line |
| `host_vllm_glm47.sh` | GLM-4.7 is not in any paper table |
| `host_vllm_glm47_flash.sh` | GLM-4.7-Flash is not in any paper table |
| `host_vllm_glm_z1.sh` | GLM-Z1 is not in any paper table |
| `host_vllm_kimi_k2.sh` | Kimi-K2 was planned but never evaluated |
| `host_vllm_llama4_scout.sh` | Llama-4-Scout is not in any paper table |
| `eval_gcpal.sh` | superseded GCPAL config |
| `eval_gcpal_v4.sh` | superseded GCPAL config |
| `eval_gcpal_v5.sh` | superseded GCPAL config |
| `eval_aml_cpu.sh` | CPU variant of the ML training job |
| `eval_baselines.sh` | superseded by the v2 ablation runner |
| `02_hybrid_infer_gptoss120b.sh` | one-off single-model rerun |
| `check_nemotron_compat.sh` | environment diagnostic, not part of the pipeline |
| `run_experiment.sh` | generic launcher predating the numbered stages |

## Repointed at the release, 2026-08-13

Every one of these fifteen scripts invoked a filename from the pre-release tree, and none of
those filenames ship: `run_all_baselines.py`, `gcpal_baseline.py`, `downsample_eval_v2.py` and
eleven more. They had been copied across without being repointed, so all fifteen died at the
first `python` line. Each now calls the numbered entry point in `scripts/`, and each `cd` goes to
`${AMLC_REPO}` rather than a `${PROJECT_ROOT}/upstream/` that exists on no machine here.

Flags moved with the names, since the release runner renamed several: `--variant` is `--dataset`,
`--methods` is `--members` for the supervised stages and `--promptings` for the LLM ones, and the
prompting values are the paper's `ICL-FS`, `ICL-ZS` and `ICL-V`. Two stale commented-out examples
in `03_train_ml_baselines.sh` were deleted rather than repointed; they named conditions that
appear nowhere in the paper and a `--mode non-llm` the runner never had. That same wrong mode was
live in the real invocation, and is now `--mode supervised`.

`14_run_intervention.sh` runs the intervention prompt, so it calls the evaluation runner with
`--promptings ICL-V`, not the comparison stage that reads its output.

All twenty-seven live invocations were then checked against the parsers they call: every path
resolves, every flag is accepted, and every value passed to a choice-constrained flag is in its
choices.
