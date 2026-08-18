#!/usr/bin/env python3
"""Assert every copy of the title and venue string agrees with docs/citation.md.

The title and the venue phrase live in five files and the venue state changes
on a known date. This is the check that stops one copy being updated and four
being forgotten, which is the failure the release has the most exposure to in
the week around notification.

    python scripts/check_citation.py
    make check-citation

It compares text, not intent: the source is docs/citation.md, and everything
else has to match it. Change the source first, then run this to find what still
disagrees.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "docs" / "citation.md"
#: The card lives in the sibling staging directory, and on the Hub. Checked when
#: it is present; skipped with a note when this is a bare clone of the code.
CARD = REPO.parent / "amlcompact-dataset" / "README.md"

#: The heading over the intended-use block on each surface. The *text* is
#: deliberately duplicated and must not drift; the heading over it is allowed to
#: differ. Both currently say "Uses": no README among the seven evaluation
#: repositories surveyed has an intended-use heading at all, so the Hugging Face
#: card template is the only authority, and it prescribes Uses / Direct Use /
#: Out-of-Scope Use. The mapping stays a mapping so one surface can be renamed
#: without the other, and the check compares bodies rather than headings.
USE_HEADING = {"repo": "## Uses", "card": "## Uses"}


def source_fields() -> dict[str, str]:
    """Read the title, venue state and phrase out of the source table."""
    text = SOURCE.read_text()
    out = {}
    for key in ("title", "venue-state", "venue-phrase", "url"):
        m = re.search(rf"^\|\s*`{re.escape(key)}`\s*\|\s*(.+?)\s*\|$", text, re.M)
        if not m:
            raise SystemExit(f"{SOURCE}: no row for `{key}` in the current-state table")
        # Values are written in the table with backticks around the short ones.
        # Strip them: leaving them on made the venue-state comparison below
        # compare "`committed`" against "committed", so the bare-venue-claim
        # check short-circuited to a pass on a README that made the claim.
        out[key] = m.group(1).strip().strip("`")
    m = re.search(r"```bibtex\n(@misc.*?|@inproceedings.*?)\n```", text, re.S)
    if not m:
        raise SystemExit(f"{SOURCE}: no BibTeX block for the paper")
    out["bibtex"] = m.group(1).strip()
    return out


def norm(s: str) -> str:
    return " ".join(s.split())


def main() -> int:
    f = source_fields()
    title, phrase, bib = f["title"], f["venue-phrase"], norm(f["bibtex"])
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok and detail else ""))
        if not ok:
            fails.append(name)

    readme = (REPO / "README.md").read_text()
    check("README carries the venue phrase", phrase in norm(readme),
          f"expected: {phrase}")
    check("README carries the BibTeX", bib in norm(readme))
    check("README makes no bare venue claim",
          f.get("venue-state") != "committed" or not re.search(r"\[EMNLP \d{4}\]|\(EMNLP \d{4}\)", readme),
          "state is `committed`, so a bare (EMNLP 2026) or [EMNLP 2026] over-claims")

    cff = (REPO / "CITATION.cff").read_text()
    check("CITATION.cff title", title in cff)
    check("CITATION.cff url", f["url"] in cff)
    check("CITATION.cff has no venue while uncommitted",
          f.get("venue-state") in ("published",) or "conference:" not in cff,
          "conference: belongs only in the published state")

    pyproject = (REPO / "pyproject.toml").read_text()
    check("pyproject version string is not a stale venue",
          not re.search(r"EMNLP \d{4}", pyproject))

    if CARD.exists():
        card = CARD.read_text()
        check("dataset card carries the venue phrase", phrase in norm(card),
              f"expected: {phrase}")
        check("dataset card BibTeX is byte-identical", bib in norm(card))
    else:
        print(f"  SKIP  dataset card, not present at {CARD}")

    # The use statement is deliberately duplicated: each surface has to stand
    # alone for its own reader. Deliberate duplication still drifts, so it is
    # checked rather than trusted.
    if CARD.exists():
        def block(text: str, head: str) -> str:
            i = text.find(head)
            if i < 0:
                return ""
            rest = text[i + len(head):]
            j = rest.find("\n## ")
            return norm(rest if j < 0 else rest[:j])
        a = block((REPO / "README.md").read_text(), USE_HEADING["repo"])
        b = block(CARD.read_text(), USE_HEADING["card"])
        check("the use statement is identical on both surfaces",
              bool(a) and a == b,
              "present on both but not identical" if a and b else "missing from one side")

    print()
    if fails:
        print(f"{len(fails)} disagree with {SOURCE.relative_to(REPO)}.")
        print("Edit that file first, then bring the others to match it.")
        return 1
    print(f"every copy agrees with {SOURCE.relative_to(REPO)}"
          f" (state: {f['venue-state']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
