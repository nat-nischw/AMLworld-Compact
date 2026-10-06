# Cluster jobs

Numbered to match `scripts/`. Every credential and node name is an environment
variable; see `.env.example`. `#SBATCH` directives cannot expand shell variables,
so pass `--nodelist` and the log paths on the `sbatch` command line.

| script | stage |
|---|---|
| `03_train_ml_baselines.sh` | LightGBM and XGBoost, tune then train, 5 seeds |
| `04_train_gcpal.sh` | GCPAL, the configuration that produced the released checkpoints |
| `05_infer_gcpal_temporal.sh` | re-infer GCPAL on the file-order split |
| `07_build_ht_coreset.sh` | HT-Coreset construction |
| `07b_build_naive_coreset.sh` | Naive Coreset for the appendix comparison |
| `08_run_ablation.sh` | the seven sampling baselines |
| `09_compare_naive_vs_ht.sh` | Naive versus HT per model |
| `11_serialize_coreset.sh` | serialise the coreset into typed-graph prompts |
| `13_run_llm_eval.sh` | one LLM over the coreset, all promptings and seeds |
| `14_run_intervention.sh` | the ICL-V prompt intervention |
| `15_rerun_thinking.sh` | re-run cases whose reasoning trace was truncated |
| `17_run_doubt_triage.sh` | Doubt Triage over every cell |
| `18_score_deferral_grid.sh` | the score-only deferral baseline of Appendix G.2 |
| `20_elliptic_pipeline.sh` | Elliptic download, ML, coreset, evaluation |
| `21_elliptic_llm_eval.sh` | Elliptic LLM evaluation (never run for the paper) |
| `check_traces.py` | trace completeness check |

`host_vllm/` holds one script per evaluated model, plus the Nemotron environment
setup and a cache cleaner for swapping models on a node.

Each script calls the numbered entry point in `scripts/` and `cd`s to
`${AMLC_REPO}`. Flag names follow the entry points: `--dataset` (not
`--variant`), `--members` for the supervised stages, `--promptings` for the LLM
ones, with the paper's `ICL-FS`, `ICL-ZS` and `ICL-V` as values.
`14_run_intervention.sh` runs the intervention prompt, so it calls the
evaluation runner with `--promptings ICL-V`, not the comparison stage that
reads its output.
