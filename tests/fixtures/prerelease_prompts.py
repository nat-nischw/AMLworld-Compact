"""Pre-release prompt builders, frozen as a test fixture.

These are the Python string templates that produced every prompt sent to every
model in the paper. They are kept here verbatim so
``scripts/verify_prompt_templates.py`` can prove, from a clone alone, that the
Jinja templates in ``prompts/`` render byte-for-byte identical output. The check
used to read this file from a staging directory beside the repository, which
meant the byte-for-byte claim could only be verified on the machine that built
the release.

Nothing imports this at runtime. Do not edit: its whole value is being the
unchanged original.
"""

"""
Prompting Strategies for AML Detection

Methods:
- LLM+ICL-AML (Pirmorad et al., 2507.14785) — exact paper format with typed
  nodes/edges (Account/Bank, transfers_to/belongs_to).
- LLM+ICL-AML-NoFS — zero-shot ablation of the same preamble/task template
  (no few-shot demonstrations).
- LLM+ICL-AML-V — ICL-AML with an explicit Verification step inserted before
  the answer format. Targets the vocabulary-without-discrimination failure
  (§6.3): asks the model to (i) reject payroll/salary patterns, (ii) require
  >=2 structural markers before flagging, (iii) default to Not Suspicious
  when no specific typology marker is identifiable. Pre-registered
  intervention for the prompt-intervention experiment (§6.5).
"""

from typing import Optional

# =============================================================================
# LLM+ICL-AML (Pirmorad et al. 2507.14785 - Exact Paper Format)
# =============================================================================

ICL_AML_PREAMBLE = """You are an expert financial crime investigator reviewing patterns of financial activities and behaviors of involved accounts to identify potential cases of money laundering. The data is represented as a graph, where:
- Nodes are of type Account or Bank.
- Edges represent relationships of type transfers_to or belongs_to, and include metadata such as amount, currency, payment method, and timestamp.

For training purposes, you will be shown examples of subgraph typologies that are known to be either suspicious (indicative of laundering tactics) or non-suspicious (routine financial activity). These typologies illustrate common structural patterns in financial networks.
"""

ICL_AML_TASK = """
Task: Given a transaction (edge) with Transaction ID <ID>, along with its surrounding subgraph, determine whether the transaction is suspicious, or not suspicious. Your reasoning should be based on whether the surrounding subgraph resembles any of the suspicious typologies provided in the examples.
Test Example:
{graph_text}
Answer Format:
- Conclusion: Suspicious or Not Suspicious
- Explanation: (2-3 sentences reasoning)
- Observed Pattern: (e.g., gather-scatter)
"""


# ICL-AML-V (Verified): adds explicit verification step before answer format.
# Pre-registered intervention targeting vocabulary-without-discrimination (§6.3, F6).
ICL_AML_V_TASK = """
Task: Given a transaction (edge) with Transaction ID <ID>, along with its surrounding subgraph, determine whether the transaction is suspicious, or not suspicious. Your reasoning should be based on whether the surrounding subgraph resembles any of the suspicious typologies provided in the examples.
Test Example:
{graph_text}
Verification step (perform before concluding):
1. If the subgraph shows a hub account with recurring fixed-amount transfers to many recipients with regular timestamps (e.g., monthly/biweekly), this pattern is consistent with payroll/salary disbursement and is NOT suspicious. Do not select fan-out, gather-scatter, or stack solely on degree and volume features.
2. If the only observed signal is high transaction volume, identify at least one additional structural marker - cycles, identical amounts crossing reporting thresholds, or rapid layering across multiple counterparties - before classifying as suspicious.
3. If you cannot identify a specific structural marker matching one of the 8 typologies, classify the transaction as Not Suspicious.
Answer Format:
- Conclusion: Suspicious or Not Suspicious
- Explanation: (2-3 sentences reasoning, must reference at least one structural marker if Suspicious)
- Observed Pattern: (e.g., gather-scatter)
"""


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def build_prompt_icl_aml(graph_text: str, icl_examples: Optional[dict] = None) -> str:
    """Build ICL-AML prompt with real examples from training set.

    Args:
        graph_text: Serialized test case subgraph
        icl_examples: Dict with 'suspicious_examples' and 'non_suspicious_examples'
                      loaded from ICL examples cache (generate_icl_examples.py).
                      If None, raises an error (examples are required).
    """
    if icl_examples is None:
        raise ValueError(
            "ICL-AML requires real ICL examples. "
            "Run: python generate_icl_examples.py --datasets <DATASET> --seed 42"
        )

    parts = [ICL_AML_PREAMBLE]

    # Suspicious examples
    parts.append("Few-shot Examples:\n")
    for ex in icl_examples["suspicious_examples"]:
        parts.append(ex["serialized_text"])
        parts.append(f"\nExplanation: {ex['explanation']}\n\n")

    # Non-suspicious examples
    parts.append("non-suspicious Examples:\n")
    for ex in icl_examples["non_suspicious_examples"]:
        parts.append(ex["serialized_text"])
        parts.append(f"\nExplanation: {ex['explanation']}\n\n")

    # Task section
    parts.append(ICL_AML_TASK.format(graph_text=graph_text))

    return "".join(parts)


def build_prompt_icl_aml_nofs(graph_text: str) -> str:
    """Build ICL-AML prompt WITHOUT few-shot examples (ablation).

    Same paper_format preamble and task template as ICL-AML,
    but no 8 suspicious + 4 non-suspicious examples in context.
    """
    parts = [ICL_AML_PREAMBLE]
    parts.append(ICL_AML_TASK.format(graph_text=graph_text))
    return "".join(parts)


def build_prompt_icl_aml_v(graph_text: str, icl_examples: Optional[dict] = None) -> str:
    """Build ICL-AML-V (Verified) prompt with same few-shot examples but
    an explicit verification step inserted before the answer format.

    Pre-registered prompt-intervention variant (§6.5). All other content
    (preamble, ICL examples, graph text) is identical to LLM+ICL-AML so
    the only contrast is the verification rubric.
    """
    if icl_examples is None:
        raise ValueError(
            "ICL-AML-V requires real ICL examples. "
            "Run: python generate_icl_examples.py --datasets <DATASET> --seed 42"
        )

    parts = [ICL_AML_PREAMBLE]

    parts.append("Few-shot Examples:\n")
    for ex in icl_examples["suspicious_examples"]:
        parts.append(ex["serialized_text"])
        parts.append(f"\nExplanation: {ex['explanation']}\n\n")

    parts.append("non-suspicious Examples:\n")
    for ex in icl_examples["non_suspicious_examples"]:
        parts.append(ex["serialized_text"])
        parts.append(f"\nExplanation: {ex['explanation']}\n\n")

    parts.append(ICL_AML_V_TASK.format(graph_text=graph_text))

    return "".join(parts)


# =============================================================================
# PROMPT GETTER
# =============================================================================

def get_prompt_builder(method: str):
    """Get prompt builder function for a method"""
    builders = {
        "LLM+ICL-AML": build_prompt_icl_aml,
        "LLM+ICL-AML-NoFS": build_prompt_icl_aml_nofs,
        "LLM+ICL-AML-V": build_prompt_icl_aml_v,
    }
    return builders.get(method)
