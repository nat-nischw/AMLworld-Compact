# Frontier-API prompt probe

Six prompt designs run against three frontier APIs on a nine-case pilot slice, reported in
the discussion section. Each variant is a system template and a user template; the user
template takes `graph_text`.

| template | what it changes |
|---|---|
| `zs_base` | zero-shot baseline, flagged-transaction framing |
| `zs_graph_classification` | zero-shot, reframed to classify the subgraph rather than the flagged transaction |
| `fs_base` | 8 suspicious + 4 non-suspicious demonstrations, with a computed structural-features header |
| `zs_abstain` | zero-shot plus a soft-abstention preamble |
| `fs_cot_elim` | few-shot plus explicit chain-of-thought elimination before concluding |
| `fs_typfirst` | few-shot, typology answered first and the binary verdict derived from it |

`scripts/verify_probe_templates.py` renders all six against the nine pilot cases and compares
byte-for-byte with the prompts that were sent. All twelve templates reproduce exactly.

Naming: the archived directories call the zero-shot variants `nofs`. This release uses `ZS`
throughout, matching the paper's ICL-ZS.

## Where the appendix and the executed runs disagree

Found while extracting these. Each is a camera-ready item, and the release makes all of them
checkable.

1. **`FS-Base` as the appendix describes it was never run.** The appendix documents it as a
   distinct variant adding a structural-features header, which is archive variant `V3`. `V3`
   has no results for any model. The variant that ran under the few-shot baseline slot is
   `V2`, and it *already contains* that structural-features header, so the header is not a
   delta over the baseline: it is part of it.
2. **A sixth executed variant is undocumented.** `V0`, the subgraph-level reframing, ran on
   all three models and appears in no appendix box.
3. **The abstain variant that ran is the softened one.** The appendix box describes `V4`; the
   results come from `V4p1`.
4. **The serialisation claim is wrong.** The appendix states that all six share the
   typed-graph serialiser of Pirmorad et al. They do not. These prompts use a condensed
   serialisation: self-transfers removed, repeated identical transfers collapsed into a count
   and mean amount, and account roles labelled SOURCE / SINK / HUB / PASS. There is not one
   typed-graph marker in any of the pilot prompts.

Two pairs also share a system prompt, with the delta carried entirely in the user template:
`zs_base` with `zs_graph_classification`, and `fs_base` with `fs_cot_elim`.
