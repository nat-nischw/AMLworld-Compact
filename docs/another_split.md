# Running this on another split, or your own synthetic data

The paper reports HI-Small and LI-Small because an LLM study over more splits was out of
budget, not because anything here is specific to them. Nothing in the coreset protocol depends
on which AMLworld variant it is given.

Every stage takes `--dataset`, and the data layer wants the two files AMLworld ships per
variant, `<name>_Trans.csv` and `<name>_Patterns.txt`. Point it at HI-Medium, LI-Large, or a
fresh run of IBM's generator under any name you like, and the sequence is the same:

```bash
make gfp DATASETS=HI-Medium        # 01  graph features
make train-ml DATASETS=HI-Medium   # 03  supervised baselines
make ensemble DATASETS=HI-Medium   # 06  soft vote, and the operating threshold
make coreset DATASETS=HI-Medium    # 07  the HT-Coreset itself
make serialize DATASETS=HI-Medium  # 11  the coreset as typed subgraph prompts
```

Stage 11 is the text conversion: it walks the two-hop neighbourhood of each retained edge and
renders the typed graph the LLM sees. It reads the coreset indices and the transaction graph
and nothing else, so it needs no per-split tuning.

Four constants have to be supplied, and they all live in `amlc/config.py`:

| Constant | Where it comes from |
|---|---|
| `DATASETS` | add the name |
| `N_TEST_FULL` | rows in the full temporal test split. The HT weights must sum to it, and `build_hf_dataset.py` refuses to publish when they do not |
| `ML_THRESHOLDS` | the operating point stage 06 selects. Not a guess |
| `N_CORESET` | what stage 07 produced, recorded after the fact |

Two things do not carry over. `data/tuned_params/` holds Optuna results for the two Small
splits only, so either run `make tune` or accept the shipped parameters as a starting point.
And `scripts/00_download_amlworld.py` knows the Small filenames and their SHA256s; a different
variant needs its entries added, or the download skipped and the CSVs placed by hand.

The zero-bias property is a claim about the construction rather than about these two splits, so
a new split is also a test of it. `make ablation` runs the same seven alternative samplers
against it and reports whether the weighted metrics still reproduce the full-set ones.
