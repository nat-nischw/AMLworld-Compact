#!/usr/bin/env python3
"""Assert the shipped Jinja2 templates reproduce the executed prompts exactly.

Renders each template and compares byte-for-byte against the pre-release Python
string builders that produced every result in the paper. A released prompt that
differs from the executed one, even by whitespace, is a silent reproducibility
break, so this is a hard equality test rather than a similarity check.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from amlc.llm.prompts import PROMPTINGS, load_icl_examples, render


def reference(prompting: str, graph_text: str, ex: dict) -> str:
    """The pre-release builders, read from the fixture frozen in this repo.

    This used to reach outside the repository, to a staging directory beside it,
    which made the byte-for-byte claim unverifiable by anyone who cloned.
    """
    import re
    legacy = (REPO / "tests" / "fixtures" / "prerelease_prompts.py").read_text()
    grab = lambda n: re.search(rf'{n} = """(.*?)"""', legacy, re.S).group(1)
    preamble, task, task_v = (grab("ICL_AML_PREAMBLE"), grab("ICL_AML_TASK"),
                              grab("ICL_AML_V_TASK"))
    if prompting == "ICL-ZS":
        return preamble + task.format(graph_text=graph_text)
    parts = [preamble, "Few-shot Examples:\n"]
    for e in ex["suspicious_examples"]:
        parts += [e["serialized_text"], f"\nExplanation: {e['explanation']}\n\n"]
    parts.append("non-suspicious Examples:\n")
    for e in ex["non_suspicious_examples"]:
        parts += [e["serialized_text"], f"\nExplanation: {e['explanation']}\n\n"]
    parts.append((task_v if prompting == "ICL-V" else task).format(graph_text=graph_text))
    return "".join(parts)


def main() -> None:
    graph_text = "=== Transaction Subgraph (Case: amlc_00000) ===\n\n**Nodes:**\n- acct_X\n"
    failed = 0
    for dataset in ("HI-Small", "LI-Small"):
        ex = load_icl_examples(dataset)
        for prompting in PROMPTINGS:
            got = render(prompting, graph_text, ex)
            want = reference(prompting, graph_text, ex)
            if got == want:
                print(f"  OK   {dataset:9s} {prompting:7s} {len(got):>8,} chars")
            else:
                failed += 1
                n = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                         min(len(got), len(want)))
                print(f"  FAIL {dataset:9s} {prompting:7s} diverges at char {n}")
                print(f"       rendered  {got[max(0,n-60):n+60]!r}")
                print(f"       reference {want[max(0,n-60):n+60]!r}")
    if failed:
        sys.exit(f"{failed} template(s) do not reproduce the executed prompt")
    print("all templates reproduce the executed prompts byte-for-byte")


if __name__ == "__main__":
    main()
