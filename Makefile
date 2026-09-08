# AMLCompact. `make help` lists everything.
#
# Stages are numbered in pipeline order and match scripts/ and scripts/slurm/.
# Archive reconstruction uses AMLC_ARCHIVE pointing at the original run
# directory's outputs/. Fresh LLM scoring can use RUNS with the released coreset.

PY      ?= python
SCRIPTS := scripts
DATASETS ?= HI-Small LI-Small

.DEFAULT_GOAL := help
.PHONY: help install install-dev selftest check-citation verify test lint env \
        verify-probe download gfp tune train-ml train-gcpal infer-gcpal ensemble \
        coreset coreset-naive ablation compare serialize icl \
        llm-eval triage triage-stats deferral rubric judges iaa human-rating \
        analysis llm-ht figures dataset clean

# ── setup ──────────────────────────────────────────────────────────────
install:  ## Install the package and the core dependencies
	pip install -e .

install-dev:  ## Install everything the test suite and the figures need
	pip install -e ".[dev]"

env:  ## Create the conda environments the cluster jobs expect
	conda env create -f envs/amlc-bench.yml
	conda env create -f envs/amlc-vllm.yml
	conda env create -f envs/amlc-vllm-nemo.yml

# ── checks that need nothing but this checkout ─────────────────────────
selftest: ## Score the shipped ensemble and compare with the paper. Needs the dataset
	$(PY) -m amlc.selftest

check-citation: ## Check paper title, URL, and BibTeX across release documents
	$(PY) $(SCRIPTS)/check_citation.py

verify: ## Prove the prompt templates reproduce the executed prompts
	$(PY) $(SCRIPTS)/verify_prompt_templates.py

verify-probe: ## Same for the frontier-probe templates. Needs AMLC_PILOT_DIR from the archive
	@test -n "$(AMLC_PILOT_DIR)" || { echo "Set AMLC_PILOT_DIR to the archived pilot prompt directory. The probe prompts embed AMLworld transactions, so they are not redistributed here."; exit 2; }
	$(PY) $(SCRIPTS)/verify_probe_templates.py

lint: ## Run the lint selection CI enforces
	ruff check --select E4,E9,F,RUF100 .

test: ## Run the smoke test a fresh clone must pass
	$(PY) -m pytest tests/ -v

# ── stages 00 to 06: data and the supervised baselines ─────────────────
download: ## 00  Fetch AMLworld from Kaggle
	$(PY) $(SCRIPTS)/00_download_amlworld.py

gfp: ## 01  Build the Graph-Feature-Preprocessor tensors (2-3 h per split)
	$(PY) $(SCRIPTS)/01_build_gfp.py --datasets $(DATASETS)

tune: ## 02  Optuna search. Optional, data/tuned_params/ already has the results
	@for d in $(DATASETS); do $(PY) $(SCRIPTS)/02_tune_hyperparams.py --dataset $$d || exit 1; done

train-ml: ## 03  LightGBM+GFP and XGBoost+GFP over 5 seeds
	$(PY) $(SCRIPTS)/03_train_ml_baselines.py --datasets $(DATASETS)

train-gcpal: ## 04  The GCPAL graph baseline
	@for d in $(DATASETS); do $(PY) $(SCRIPTS)/04_train_gcpal.py --dataset $$d || exit 1; done

infer-gcpal: ## 05  Re-infer GCPAL on the temporal split
	$(PY) $(SCRIPTS)/05_infer_gcpal_temporal.py --datasets $(DATASETS)

ensemble: ## 06  Soft-vote the three members
	$(PY) $(SCRIPTS)/06_score_ensemble.py --datasets $(DATASETS)

# ── stages 07 to 12: the coreset and its prompts ───────────────────────
coreset: ## 07  Construct the HT-Coreset
	$(PY) $(SCRIPTS)/07_build_ht_coreset.py --datasets $(DATASETS)

coreset-naive: ## 07b Construct the Naive Coreset (Table 10 only, slow)
	$(PY) $(SCRIPTS)/07b_build_naive_coreset.py --datasets $(DATASETS)

ablation: ## 08  The seven sampling baselines
	$(PY) $(SCRIPTS)/08_run_ablation.py --datasets $(DATASETS)

compare: ## 09  Naive versus HT-Coreset, per model
	$(PY) $(SCRIPTS)/09_compare_naive_vs_ht.py --datasets $(DATASETS)

serialize: ## 11  Serialise the coreset into typed-graph prompts
	$(PY) $(SCRIPTS)/11_serialize_coreset.py --datasets $(DATASETS)

icl: ## 12  Build the in-context demonstration pool
	$(PY) $(SCRIPTS)/12_build_icl_examples.py --datasets $(DATASETS)

# ── stages 13 to 18: the LLM evaluation and Doubt Triage ───────────────
llm-eval: ## 13  Evaluate one LLM. Host a vLLM server first; see scripts/slurm/host_vllm/
	@test -n "$(MODEL)" -a -n "$(VLLM)" || { echo "Needs both: make llm-eval MODEL=GPT-OSS-120B VLLM=http://host:18809/v1"; exit 2; }
	$(PY) $(SCRIPTS)/13_run_llm_eval.py --model $(MODEL) --vllm-url $(VLLM) --datasets $(DATASETS)

triage: ## 17  Doubt Triage over every cell
	$(PY) $(SCRIPTS)/17_run_doubt_triage.py

triage-stats: ## 17b Paired DT-versus-ML statistics from the stage-17 table
	$(PY) $(SCRIPTS)/17b_dt_stats.py

deferral: ## 18  The score-only deferral baseline of Appendix G.3
	$(PY) $(SCRIPTS)/18_score_deferral_grid.py

# ── stages 16 and 19: the four-step audit rubric ───────────────────────
rubric: ## 16  Draw the 1,000-trace sample and run the regex annotator
	$(PY) $(SCRIPTS)/16_sample_traces.py

judges: ## 19a Score the rubric with a frontier-API judge (needs an API key)
	@test -n "$(JUDGE)" || { echo "Needs a provider: make judges JUDGE=deepseek"; exit 2; }
	$(PY) $(SCRIPTS)/19a_run_judges.py $(JUDGE)

iaa: ## 19c Fleiss and pairwise kappa across the four judges
	$(PY) $(SCRIPTS)/19b_analyze_rubric.py
	$(PY) $(SCRIPTS)/19c_compute_iaa.py

human-rating: ## 19d The three-rater human validation tables
	$(PY) $(SCRIPTS)/19d_human_rating.py

# ── stages 20 to 22: analysis and artefacts ────────────────────────────
analysis: ## 20  Per-typology and error-transition analysis
	$(PY) $(SCRIPTS)/20_error_analysis.py

llm-ht: ## 23  Score HT and compact detection metrics; RUNS=path uses fresh runner output
	$(PY) $(SCRIPTS)/23_score_llm_ht.py $(if $(RUNS),--runs-dir "$(RUNS)") $(if $(MODEL),--models $(MODEL)) $(if $(PROMPTINGS),--promptings $(PROMPTINGS)) $(if $(SEEDS),--seeds $(SEEDS)) $(foreach d,$(DATASETS),--dataset $(d)) $(if $(OUT),--out "$(OUT)")

figures: analysis ## Generate the typology and error-transition plots from stage 20

dataset: ## Rebuild the published dataset from the archive (maintainers)
	$(PY) $(SCRIPTS)/build_hf_dataset.py --archive $$AMLC_ARCHIVE --out build/dataset

clean: ## Remove caches and build output
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ .pytest_cache/ .ruff_cache/ *.egg-info

help: ## Show this message
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'
