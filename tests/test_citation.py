"""Check anonymous disclosure and identified citation consistency."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_citation.py"
SPEC = importlib.util.spec_from_file_location("check_citation", SCRIPT)
citation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(citation)

TITLE = "A transaction evaluation set"
USES = "\n## Uses\nSynthetic research only.\n"
WITHHELD = "The citation for this release is withheld during review.\n"
ANONYMOUS_CFF = f"""title: {TITLE}
authors:
  - name: Authors withheld during review
year: 2026
"""
PUBLIC_CFF = f"""title: {TITLE}
url: https://example.org/paper
authors:
  - family-names: Researcher
    given-names: A.
  - name: Example Laboratory
year: 2026
"""


def setup_release(monkeypatch, tmp_path, cff, readme, card=None):
    repo = tmp_path / "code-review"
    repo.mkdir()
    source = repo / "CITATION.cff"
    source.write_text(cff)
    (repo / "README.md").write_text(readme)
    card_path = tmp_path / "dataset-review" / "README.md"
    if card is not None:
        card_path.parent.mkdir()
        card_path.write_text(card)
    monkeypatch.setattr(citation, "REPO", repo)
    monkeypatch.setattr(citation, "SOURCE", source)
    monkeypatch.setattr(citation, "CARD", card_path)


def paper_entry(author="Researcher, A. and Example Laboratory",
                url="https://example.org/paper"):
    return f"""```bibtex
@misc{{evaluation,
  title = {{{TITLE}}},
  author = {{{author}}},
  url = {{{url}}},
  year = {{2026}}
}}
```
"""


def test_anonymous_release_accepts_withholding_and_unrelated_source_citation(
        monkeypatch, tmp_path):
    unrelated = "```bibtex\n@misc{source, title={Source dataset}, year={2023}}\n```\n"
    setup_release(monkeypatch, tmp_path, ANONYMOUS_CFF,
                  WITHHELD + unrelated + USES, WITHHELD + USES)
    assert citation.main() == 0


@pytest.mark.parametrize("readme", [
    "No citation section here.\n" + USES,
    WITHHELD + paper_entry() + USES,
    WITHHELD + paper_entry("Authors withheld during review") + USES,
])
def test_anonymous_release_rejects_missing_disclosure_or_identified_self_citation(
        monkeypatch, tmp_path, readme):
    setup_release(monkeypatch, tmp_path, ANONYMOUS_CFF, readme)
    assert citation.main() == 1


def test_anonymous_cff_rejects_a_public_url(monkeypatch, tmp_path):
    setup_release(monkeypatch, tmp_path,
                  ANONYMOUS_CFF + "url: https://example.org/paper\n", WITHHELD)
    with pytest.raises(SystemExit, match="anonymous citation must withhold its URL"):
        citation.source_fields()


def test_public_release_accepts_ordered_person_and_organization_authors(
        monkeypatch, tmp_path):
    text = paper_entry() + USES
    setup_release(monkeypatch, tmp_path, PUBLIC_CFF, text, text)
    assert citation.main() == 0


@pytest.mark.parametrize("readme", [
    paper_entry("Example Laboratory and Researcher, A.") + USES,
    paper_entry(url="https://example.org/wrong-paper") + USES,
    paper_entry().replace("2026", "2025") + USES,
])
def test_public_release_rejects_changed_citation_fields(monkeypatch, tmp_path, readme):
    setup_release(monkeypatch, tmp_path, PUBLIC_CFF, readme)
    assert citation.main() == 1


def test_dataset_card_uses_statement_must_match(monkeypatch, tmp_path):
    setup_release(monkeypatch, tmp_path, ANONYMOUS_CFF, WITHHELD + USES,
                  WITHHELD + "\n## Uses\nA different scope.\n")
    assert citation.main() == 1
