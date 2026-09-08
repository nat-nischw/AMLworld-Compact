#!/usr/bin/env python3
"""Check README citations against CITATION.cff and compare shared Uses text.

    python scripts/check_citation.py
    make check-citation

The sibling dataset card is checked when present. Additional citations, such as
AMLworld, are allowed. Publication dates are never inferred.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "CITATION.cff"
CARD = REPO.parent / "amlcompact-dataset" / "README.md"


def norm(text: str) -> str:
    return " ".join(text.split())


def source_fields() -> dict:
    """Read the canonical fields, retaining the CFF author order."""
    try:
        cff = yaml.safe_load(SOURCE.read_text())
        fields = {key: norm(cff[key]) for key in ("title", "url")}
        fields["authors"] = [
            (norm(f"{a['family-names']}, {a['given-names']}"),
             norm(f"{a['given-names']} {a['family-names']}"))
            for a in cff["authors"]]
        if not fields["authors"] or not all(fields[key] for key in ("title", "url")):
            raise ValueError("title, URL, and authors must be nonempty")
        if "year" in cff:
            fields["year"] = str(cff["year"])
        return fields
    except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise SystemExit(f"{SOURCE}: cannot read citation fields: {exc}") from exc


def bibtex_entries(text: str) -> list[dict[str, str]]:
    """Read citation fields from fenced BibTeX, allowing nested title braces."""
    entries = []
    for block in re.findall(r"```bibtex\s*\n(.*?)```", text, re.S | re.I):
        for entry in re.split(r"@\w+\s*[{(]", block)[1:]:
            fields = {}
            for match in re.finditer(r"\b(title|url|author|year)\s*=\s*", entry, re.I):
                value = entry[match.end():]
                if value.startswith("{"):
                    depth = 0
                    for end, char in enumerate(value):
                        if end and value[end - 1] == "\\":
                            continue
                        depth += (char == "{") - (char == "}")
                        if depth == 0:
                            value = value[1:end]
                            break
                    else:
                        continue
                    value = re.sub(r"(?<!\\)[{}]", "", value)
                elif value.startswith('"'):
                    quoted = re.match(r'"((?:\\.|[^"\\])*)"', value)
                    if not quoted:
                        continue
                    value = quoted.group(1)
                else:
                    value = re.split(r"[,}\n]", value, maxsplit=1)[0]
                fields[match.group(1).lower()] = norm(value)
            entries.append(fields)
    return entries


def uses_block(text: str) -> str:
    match = re.search(r"^## Uses\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return norm(match.group(1)) if match else ""


def main() -> int:
    source = source_fields()
    failures = []

    def check(name: str, ok: bool) -> None:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
        if not ok:
            failures.append(name)

    documents = {"README": (REPO / "README.md").read_text()}
    if CARD.exists():
        documents["dataset card"] = CARD.read_text()
    else:
        print(f"  SKIP  dataset card, not present at {CARD}")

    for name, text in documents.items():
        candidates = [entry for entry in bibtex_entries(text)
                      if entry.get("title") == source["title"]
                      or entry.get("url") == source["url"]]
        check(f"{name}: one paper BibTeX entry", len(candidates) == 1)
        if len(candidates) != 1:
            continue
        entry = candidates[0]
        for field in ("title", "url", "year"):
            if field in source:
                check(f"{name}: {field}", entry.get(field) == source[field])
        authors = re.split(r"\s+and\s+", entry.get("author", ""))
        expected = source["authors"]
        check(f"{name}: authors in CFF order",
              len(authors) == len(expected)
              and all(author in forms for author, forms in zip(authors, expected)))

    if "dataset card" in documents:
        a, b = (uses_block(documents[name]) for name in ("README", "dataset card"))
        check("shared Uses statement", bool(a) and a == b)

    if failures:
        print(f"\n{len(failures)} consistency check(s) failed; compare the fields above "
              "with CITATION.cff and the shared Uses text.")
        return 1
    print("\nCitation fields match CITATION.cff; shared Uses text matches when present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
