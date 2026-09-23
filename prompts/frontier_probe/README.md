# Frontier-API prompt probe

Six prompt designs, each a system template and a user template; the user
template takes `graph_text`. The scored runs and their results live in
[`results/frontier_probe/`](../../results/frontier_probe/).

| template | what it changes |
|---|---|
| `zs_base` | zero-shot baseline, flagged-transaction framing |
| `zs_graph_classification` | zero-shot, reframed to classify the subgraph rather than the flagged transaction |
| `fs_base` | 8 suspicious + 4 non-suspicious demonstrations, with a computed structural-features header |
| `zs_abstain` | zero-shot plus a soft-abstention preamble |
| `fs_cot_elim` | few-shot plus explicit chain-of-thought elimination before concluding |
| `fs_typfirst` | few-shot, typology answered first and the binary verdict derived from it |

`scripts/verify_probe_templates.py` renders all six against the nine pilot
cases and compares byte-for-byte with the prompts that were sent. All twelve
templates reproduce exactly.

Two pairs share a system prompt, with the delta carried entirely in the user
template: `zs_base` with `zs_graph_classification`, and `fs_base` with
`fs_cot_elim`.

## Serialisation

These prompts use a condensed serialisation rather than the typed-graph
serialiser of the main evaluation: self-transfers removed, repeated identical
transfers collapsed into a count and mean amount, and account roles labelled
SOURCE / SINK / HUB / PASS.

## Naming

The archived directories call the zero-shot variants `nofs`. This release uses
`ZS` throughout, matching the paper's ICL-ZS.
