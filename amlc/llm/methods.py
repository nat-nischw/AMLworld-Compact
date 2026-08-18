"""The three in-context conditions, and how a verdict is read out of the text.

Each condition sends one prompt per case and parses one answer. They differ
only in the template:

    ICL-FS   preamble + 8 suspicious and 4 non-suspicious demonstrations + task
    ICL-ZS   preamble + task, the same instructions with no demonstrations
    ICL-V    ICL-FS plus a three-rule verification step before the answer format

The prompt text itself is not here. It lives in ``prompts/*.j2`` and is rendered
by :func:`amlc.llm.prompts.render`, which is byte-compared against the
pre-release string builders by ``scripts/verify_prompt_templates.py``. The
pre-release carried the same strings in a Python module beside the methods,
where a stray edit could not be caught.

The models answer in prose, in the three-line format the task block asks for::

    - Conclusion: Suspicious
    - Explanation: ...
    - Observed Pattern: gather-scatter

:func:`parse_verdict` turns that into a detection decision and a typology, and
it is what the paper means by the parser-extracted verdict: the Conclude step of
the audit rubric scores this field, not a hedge inside the reasoning.

Graph text is not re-serialised here. Every case is looked up in the texts
mapping the runner supplies, which comes from the released evaluation table or
from ``cases_typed_graph.jsonl``. The pre-release fell back to re-rendering the
case as an edge list when a typed graph was missing, which is the format
mismatch fixed in March: the preamble describes typed nodes and edges, so an
edge-list body produced prompts unlike anything reported. A missing text is now
an error.
"""

from __future__ import annotations

import re
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .. import config
from .clients import BaseClient
from .prompts import render

#: Unicode dash variants that show up in model output. GPT-OSS in particular
#: emits U+2011 inside 'fan-out' and 'gather-scatter', which silently broke the
#: ASCII substring match and mis-extracted the typology as 'stack' through the
#: old greedy 'layer' rule.
_UNICODE_HYPHENS = ("\u2010\u2011\u2012\u2013\u2014\u2015"
                    "\u2212\u2043\u00ad")
_ASCII_HYPHEN = {ord(c): "-" for c in _UNICODE_HYPHENS}

#: Surface form -> typology name. Order matters: the longer, more specific keys
#: come first so 'fan-out' wins over 'fan', and 'scatter-gather' over 'scatter'.
#:
#: 'layer' is deliberately absent. It used to map to 'stack' and greedily
#: matched 'distribution layer' and 'service layer', which are not typologies.
#: 'layering' stays, because that is the AML term the stack typology names.
PATTERN_MAP = (
    ("scatter-gather", "scatter-gather"),
    ("scattergather", "scatter-gather"),
    ("gather-scatter", "gather-scatter"),
    ("gatherscatter", "gather-scatter"),
    ("fan-out", "fan-out"), ("fanout", "fan-out"), ("fan out", "fan-out"),
    ("fan-in", "fan-in"), ("fanin", "fan-in"), ("fan in", "fan-in"),
    ("bipartite", "bipartite"), ("mule", "bipartite"),
    ("cycle", "cycle"), ("circular", "cycle"), ("round-trip", "cycle"),
    ("structuring", "random"), ("smurfing", "random"), ("random", "random"),
    ("layering", "stack"), ("stack", "stack"),
)


@dataclass
class PredictionResult:
    """One model verdict on one case, as written to the per-seed JSON."""

    case_id: str
    method: str
    illicit: bool
    typology: Optional[str]
    evidence_edges: List[str]
    confidence: float
    rationale: str
    llm_calls: int = 1
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    # Always None. The pre-release typology verifier is not part of this
    # release; the field stays so a regenerated prediction file has the same
    # shape as the archived ones.
    verification_passed: Optional[bool] = None
    raw_outputs: List[Dict] = field(default_factory=list)
    raw_response: str = ""        # the answer text, thinking already stripped
    center_edge_id: str = ""      # joins back to the coreset row
    reasoning_content: str = ""   # the chain of thought, where the model emits one
    llm_raw_response: Optional[Dict] = None


def ascii_normalise(s: str) -> str:
    """Fold Unicode dashes onto ASCII hyphens before any matching."""
    if not s:
        return s
    return s.translate(_ASCII_HYPHEN)


def parse_verdict(content: str) -> Dict:
    """Read the answer block: conclusion, explanation, observed pattern.

    Anything the model does not state falls back to a benign verdict with no
    typology, which is how an unparseable answer is counted.
    """
    result = {
        "illicit": False,
        "typology_program": None,
        "evidence_edges": [],
        "confidence": 0.5,
        "rationale": "",
    }

    content_n = ascii_normalise(content)

    conclusion_match = re.search(r"[Cc]onclusion[:\s]*([^\n]+)", content_n,
                                 re.IGNORECASE)
    if conclusion_match:
        conclusion = conclusion_match.group(1).strip().lower()
        result["illicit"] = ("suspicious" in conclusion
                             and "not suspicious" not in conclusion)

    explanation_match = re.search(
        r"[Ee]xplanation[:\s]*([^\n]+(?:\n(?![Oo]bserved)[^\n]+)*)",
        content_n, re.IGNORECASE)
    if explanation_match:
        result["rationale"] = explanation_match.group(1).strip()

    pattern_match = re.search(r"[Oo]bserved\s*[Pp]attern[:\s]*([^\n]+)",
                              content_n, re.IGNORECASE)
    if pattern_match:
        pattern = pattern_match.group(1).strip().lower()
        for key, value in PATTERN_MAP:
            if key in pattern:
                result["typology_program"] = value
                break

    # Confidence is a coarse stand-in, not a calibrated probability: the answer
    # format asks for no number, so it records how specific the answer was.
    if result["illicit"] and result["typology_program"]:
        result["confidence"] = 0.8
    elif result["illicit"]:
        result["confidence"] = 0.6
    else:
        result["confidence"] = 0.7

    return result


class PromptingMethod(ABC):
    """One in-context condition.

    Subclasses set :attr:`prompting` and nothing else: the template decides what
    the model sees, so the loop around it is shared.
    """

    #: Paper name of the condition, and the template key in prompts.render.
    prompting: str = ""

    def __init__(self, client: BaseClient, texts: Mapping[str, str],
                 icl_examples: Optional[dict] = None):
        if client is None:
            raise ValueError("a client is required; build one with get_client()")
        if self.prompting not in config.PROMPTINGS:
            raise ValueError(f"prompting must be one of {config.PROMPTINGS}")
        self.client = client
        self.texts = texts
        self.icl_examples = icl_examples
        self.name = self.prompting

    def prompt_for(self, case: Any) -> str:
        """The full prompt for one case."""
        try:
            graph_text = self.texts[case.case_id]
        except KeyError:
            raise KeyError(
                f"no serialised context graph for {case.case_id}. Prompts come "
                "from the released evaluation table or from "
                "cases_typed_graph.jsonl; re-serialising the case here would "
                "not reproduce the executed prompt."
            ) from None
        return render(self.prompting, graph_text, self.icl_examples)

    def predict(self, case: Any) -> PredictionResult:
        response = self.client.call(self.prompt_for(case))
        output = parse_verdict(response.content)
        raw = response.raw_response if isinstance(response.raw_response, dict) else {}

        return PredictionResult(
            case_id=case.case_id,
            method=self.name,
            illicit=output["illicit"],
            typology=output["typology_program"],
            evidence_edges=output["evidence_edges"],
            confidence=output["confidence"],
            rationale=output["rationale"],
            llm_calls=1,
            total_tokens=response.usage.get("total_tokens", 0),
            input_tokens=response.usage.get("prompt_tokens", 0),
            output_tokens=response.usage.get("completion_tokens", 0),
            latency_ms=response.latency_ms,
            raw_outputs=[output],
            raw_response=response.content,
            center_edge_id=getattr(case, "center_edge_id", ""),
            reasoning_content=raw.get("reasoning") or "",
            llm_raw_response=response.raw_response,
        )


class ICLFewShot(PromptingMethod):
    """ICL-FS: the protocol of Pirmorad et al. (2025), 8 + 4 demonstrations."""

    prompting = "ICL-FS"


class ICLZeroShot(PromptingMethod):
    """ICL-ZS: the same preamble and task with the demonstrations removed.

    The ablation that asks whether the ~22K tokens of demonstrations buy
    anything.
    """

    prompting = "ICL-ZS"


class ICLVerified(PromptingMethod):
    """ICL-V: ICL-FS plus a verification step, the paper's prompt intervention.

    Three rules inserted before the answer format, aimed at the
    vocabulary-without-discrimination failure: reject payroll and salary
    patterns, require at least two structural markers, and default to Not
    Suspicious when no specific marker is present.
    """

    prompting = "ICL-V"


METHODS = {cls.prompting: cls for cls in (ICLFewShot, ICLZeroShot, ICLVerified)}


def get_method(prompting: str, client: BaseClient, texts: Mapping[str, str],
               icl_examples: Optional[dict] = None) -> PromptingMethod:
    """Build one condition by paper name.

    ``icl_examples`` is required for ICL-FS and ICL-V and ignored for ICL-ZS;
    :func:`amlc.llm.prompts.render` enforces that.
    """
    if prompting not in METHODS:
        raise ValueError(f"prompting must be one of {list(METHODS)}, "
                         f"got {prompting!r}")
    return METHODS[prompting](client, texts, icl_examples=icl_examples)
