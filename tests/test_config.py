"""The configuration file says what the paper says.

`config.yaml` holds two kinds of value and only one of them is yours to change.
The addresses, the dataset repo and the vLLM endpoint, are meant to be edited.
The rest are facts about the published run: split sizes, coreset sizes,
thresholds, seeds, the sampling parameters each model was served with. Editing
one of those does not reconfigure anything, it makes the package describe a run
that did not happen, and nothing else in the code would notice.

So they are pinned here, against the values printed in the paper. A test that
merely re-read the YAML and compared it to itself would pass on any edit and is
worth nothing; every expected value below is written out by hand.
"""

import pytest

from amlc import config


def test_the_two_splits_are_the_ones_the_paper_evaluates():
    assert config.DATASETS == ("HI-Small", "LI-Small")


def test_full_split_sizes_match_the_paper():
    # Section 3.1 and Table 9.
    assert config.N_TEST_FULL == {"HI-Small": 1_015_669, "LI-Small": 1_384_810}


def test_coreset_sizes_match_the_paper():
    # Table 9, and the row counts of the published parquet files.
    assert config.N_CORESET == {"HI-Small": 3_753, "LI-Small": 2_268}


def test_the_coreset_is_a_1_to_2_design():
    # Every illicit edge retained, twice as many benign. 3,753 = 1,251 * 3.
    for ds in config.DATASETS:
        illicit = config.N_CORESET[ds] / (1 + config.BENIGN_MULTIPLIER)
        assert illicit == int(illicit), ds
        assert config.N_CORESET[ds] == int(illicit) * 3, ds


def test_operating_thresholds_match_the_paper():
    # Section 5.1 and Algorithm 2.
    assert config.ML_THRESHOLDS == {"HI-Small": 0.80, "LI-Small": 0.48}


def test_five_seeds_in_the_documented_order():
    assert config.SEEDS == (42, 123, 456, 789, 1011)
    assert config.RUBRIC_SEED == config.SEEDS[0]


def test_seven_models_and_every_one_has_sampling_parameters():
    assert len(config.LLM_MODELS) == 7
    assert set(config.LLM_SAMPLING) == set(config.LLM_MODELS)


@pytest.mark.parametrize("model,expected", [
    # Table 7. Vendor-recommended thinking-mode settings, not tuned by us.
    ("GPT-OSS-20B",           {"temperature": 1.0, "top_p": 1.00}),
    ("GPT-OSS-120B",          {"temperature": 1.0, "top_p": 1.00}),
    ("Nemotron-3-Nano-30B",   {"temperature": 1.0, "top_p": 0.95}),
    ("Nemotron-3-Super-120B", {"temperature": 1.0, "top_p": 0.95}),
    ("Qwen3.5-27B",           {"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
    ("Qwen3.5-35B-A3B",       {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
    ("Qwen3.5-397B-A17B",     {"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
])
def test_sampling_parameters_match_table_7(model, expected):
    assert config.LLM_SAMPLING[model] == expected


def test_output_budget_is_the_one_every_run_used():
    assert config.LLM_MAX_TOKENS == 32_768


def test_construction_parameters_match_section_3():
    assert config.K_HOP == 2
    assert config.MAX_NEIGHBOURS_PER_HOP == 50
    assert config.HARD_NEG_RATIO == 0.3
    assert config.HARD_NEG_THRESHOLD_FRAC == 0.5


def test_the_rubric_is_four_steps_over_a_thousand_traces():
    assert config.RUBRIC_STEPS == ("parse", "recall", "match", "conclude")
    assert config.RUBRIC_N_TRACES == 1_000


def test_three_judges_plus_the_regex_heuristic():
    assert config.JUDGE_PROVIDERS == ("deepseek", "opus", "gemini")
    assert set(config.RELEASED_JUDGE_MODEL) == set(config.JUDGE_PROVIDERS)
    assert set(config.JUDGE_COLUMN_PREFIX) == set(config.JUDGE_PROVIDERS)
    assert set(config.JUDGE_API_KEY_ENV) == set(config.JUDGE_PROVIDERS)


def test_the_consumers_read_the_config_rather_than_their_own_copy():
    """The point of the file is that there is one copy of each value."""
    from amlc import hub
    from amlc.audit import judges
    from amlc.llm import clients

    assert hub.DATASET_REPO == config.DATASET_REPO
    assert tuple(hub._ARRAYS) == config.CORESET_ARRAYS
    assert judges.PROVIDERS == config.JUDGE_PROVIDERS
    assert judges.RELEASED_JUDGE_MODEL == config.RELEASED_JUDGE_MODEL
    assert clients._SAMPLING == config.LLM_SAMPLING
    assert set(clients.MODELS) == set(config.LLM_MODELS)


def test_an_override_reaches_the_loader(monkeypatch, tmp_path):
    """AMLC_DATASET is the documented way to point at a fork."""
    import importlib

    monkeypatch.setenv("AMLC_DATASET", "someone-else/their-fork")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DATASET_REPO == "someone-else/their-fork"
    finally:
        monkeypatch.delenv("AMLC_DATASET")
        importlib.reload(config)
