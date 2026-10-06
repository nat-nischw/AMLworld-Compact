# Frontier-API prompt probe

Six prompt designs, each a system template and a user template; the user
template takes `graph_text`. The scored runs and their results live in
[`results/frontier_probe/`](../../results/frontier_probe/).

| Paper variant | Template stem | Demonstrations | Instructions |
|---|---|---|---|
| ZS-Graph | `zs_graph_classification` | None | Classify the subgraph |
| ZS-Base | `zs_base` | None | Classify the flagged transaction in its surrounding subgraph |
| FS-Base | `fs_base` | 8 suspicious + 4 non-suspicious | Use worked examples and a computed structural-features header |
| ZS-Abstain | `zs_abstain` | None | Give an initial verdict and confidence; LOW forces benign, otherwise require at least 3 of 4 policy conditions for an illicit verdict |
| FS-CoT-Elim | `fs_cot_elim` | 8 suspicious + 4 non-suspicious | Mark each typology REJECT or CONSIDER, then choose among survivors |
| FS-TypFirst | `fs_typfirst` | 8 suspicious + 4 non-suspicious | Select a typology and derive the binary verdict from it |

Each stem names a `_system.j2` and a `_user.j2` file. ZS-Graph and ZS-Base
are both zero-shot; the main evaluation's `ICL-FS` is a separate protocol.
ZS-Abstain's four conditions concern specific structural features, evidence
beyond a hub, HIGH or MEDIUM confidence in one typology, and ruling out
plausible legitimate-business explanations.

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
`ZS` throughout. The scored archive maps `V0` to ZS-Graph, `V1` to ZS-Base,
`V2` to FS-Base, `V4p1` to ZS-Abstain, `V7` to FS-CoT-Elim, and `V8` to
FS-TypFirst. These names describe the probe prompts rather than the main
evaluation's ICL-FS/ICL-ZS inputs.
