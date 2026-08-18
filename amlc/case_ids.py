"""Canonical case identifiers, and the rule about what may be rewritten.

Every case in the released evaluation set has a paper-named identifier::

    amlc_00000 .. amlc_03752   (HI-Small, 3,753 cases)
    amlc_00000 .. amlc_02267   (LI-Small, 2,268 cases)

The numeric part is the row position in
``data/coreset/<dataset>/ht_subset_indices.npy``, so ``amlc_00417`` is row 417
of the released HT-Coreset draw. Identifiers are scoped per dataset; the
``dataset`` column disambiguates.

Pre-release, the same cases were identified as ``v2_00000`` .. and that string
was frozen into artefacts generated at the time. Those artefacts carry a
``legacy_case_id`` column so published results stay matchable.

The rename is applied everywhere, including inside text
--------------------------------------------------------

Columns, filenames, prompts and recorded model output all carry ``amlc_``. The
release therefore has a single vocabulary and no ``legacy_case_id`` column.

Rewriting recorded model output deserves an explicit justification, because the
default answer is never to touch it. In a 400-record sample of GPT-OSS-120B
ICL-FS on HI-Small, 239 traces quote the identifier inside the model's own chain
of thought, as in "the given transaction subgraph (case v2_01210)". That quote
is the model echoing a label **we** wrote into the prompt; it is not a claim the
model made about the world. Renaming it on both sides at once is a relabelling,
not an edit to what the model reasoned.

Three conditions make that safe, and :func:`retag` enforces all three:

1. **Exact token only.** The substitution matches ``v2_`` followed by exactly
   five digits on a word boundary, and nothing else. It is never a free-text
   search and replace.
2. **Both sides together.** The prompt and the output that quotes it are
   rewritten in the same pass, so no artefact is left internally inconsistent.
3. **Reversible and disclosed.** The digits are untouched, so ``amlc_00417``
   maps back to ``v2_00417`` by prefix. The archived run directory keeps the
   originals unmodified, and the dataset card states the substitution.

The one cost, stated plainly: a released prompt is no longer byte-identical to
the executed one. It differs in this identifier token and in nothing else, which
:func:`retag` verifies by checking that the two strings agree outside the
matched spans.
"""

from __future__ import annotations

import re

#: Prefix of the canonical, paper-named identifier.
CASE_ID_PREFIX = "amlc_"

#: Prefix frozen into pre-release artefacts and into recorded text.
LEGACY_CASE_ID_PREFIX = "v2_"

#: Zero-padding width, unchanged across the rename.
CASE_ID_WIDTH = 5

_ANY_CASE_ID = re.compile(r"^(?:amlc_|v2_)(\d+)$")

#: A whole identifier token, anywhere in a body of text, in each vocabulary.
_TOKEN = re.compile(rf"(?<![0-9A-Za-z]){re.escape(LEGACY_CASE_ID_PREFIX)}"
                    rf"(\d{{{CASE_ID_WIDTH}}})(?![0-9A-Za-z])")
_TOKEN_NEW = re.compile(rf"(?<![0-9A-Za-z]){re.escape(CASE_ID_PREFIX)}"
                        rf"(\d{{{CASE_ID_WIDTH}}})(?![0-9A-Za-z])")
#: Either vocabulary, for the invariant checks in :func:`retag`.
_ANY_TOKEN = re.compile(rf"(?<![0-9A-Za-z])(?:{re.escape(CASE_ID_PREFIX)}"
                        rf"|{re.escape(LEGACY_CASE_ID_PREFIX)})"
                        rf"(\d{{{CASE_ID_WIDTH}}})(?![0-9A-Za-z])")


def case_id(position: int) -> str:
    """Canonical identifier for a row position in the released draw."""
    return f"{CASE_ID_PREFIX}{position:0{CASE_ID_WIDTH}d}"


def position(identifier: str) -> int:
    """Row position encoded in an identifier, accepting either prefix."""
    m = _ANY_CASE_ID.match(str(identifier).strip())
    if not m:
        raise ValueError(f"not a case identifier: {identifier!r}")
    return int(m.group(1))


def to_canonical(identifier: str) -> str:
    """Rewrite any accepted identifier into the canonical form."""
    return case_id(position(identifier))


def is_case_id(value: object) -> bool:
    """True when the value parses as a case identifier under either prefix."""
    return bool(_ANY_CASE_ID.match(str(value).strip()))


def retag(text: str) -> tuple[str, int]:
    """Rewrite every ``v2_NNNNN`` token to ``amlc_NNNNN``.

    Returns ``(new_text, n_substitutions)``. Only whole identifier tokens match,
    so surrounding prose, account ids and edge ids are untouched.

    Three invariants are asserted, which together mean the substitution changed
    identifiers and nothing else:

    1. deleting every identifier token, in either vocabulary, from the input
       and from the output leaves byte-identical remainders,
    2. the ordered sequence of five-digit groups is unchanged,
    3. no pre-release token survives in the output.

    Both vocabularies have to be stripped in check 1, because a partly migrated
    artefact legitimately holds ``amlc_`` and ``v2_`` tokens side by side.
    """
    out, n = _TOKEN.subn(lambda m: f"{CASE_ID_PREFIX}{m.group(1)}", text)
    if n == 0:
        return text, 0

    if _ANY_TOKEN.sub("", text) != _ANY_TOKEN.sub("", out):
        raise AssertionError("retag changed text outside the identifier tokens")
    if _ANY_TOKEN.findall(text) != _ANY_TOKEN.findall(out):
        raise AssertionError("retag did not preserve the identifier digits")
    if _TOKEN.search(out):
        raise AssertionError("retag left a pre-release identifier in the output")
    return out, n


