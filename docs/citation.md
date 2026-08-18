# How to cite, and the one place the venue string lives

The paper's title, its venue state and its BibTeX appear in five files. They are
about to change: notification is 2026-08-20, and the committed revision list may
also qualify the title. Five hand-maintained copies of a string that is about to
change is the definition of drift, so this file is the source and
`make check-citation` asserts the others agree with it.

## Current state

| Field | Value |
|---|---|
| `title` | AMLCompact: Bias-Free Downsampling Makes AMLworld Tractable for LLM Evaluation |
| `venue-state` | `committed` |
| `venue-phrase` | an ARR submission committed to EMNLP 2026 (Resources and Evaluation) |
| `url` | https://openreview.net/forum?id=VouFZFf8Ph |

## BibTeX

```bibtex
@misc{nitarach2026amlcompact,
  title  = {AMLCompact: Bias-Free Downsampling Makes AMLworld Tractable for LLM Evaluation},
  author = {Nitarach, Natapong and Ngampornsukswadi, Phume and
            Taveekitworachai, Pittawat and Nonesung, Surapon and
            Sirichotedumrong, Warit and Halverson, Duncan and
            Pipatanakul, Kunat},
  year   = {2026},
  url    = {https://openreview.net/forum?id=VouFZFf8Ph}
}
```

Cite AMLworld too. Nothing here exists without it, and its licence asks for the
attribution to be preserved.

```bibtex
@inproceedings{altman2023realistic,
  title     = {Realistic Synthetic Financial Transactions for Anti-Money Laundering Models},
  author    = {Altman, Erik and Blanu{\v{s}}a, Jovan and von Niederh{\"a}usern, Luc and
               Egressy, B{\'e}ni and Anghel, Andreea and Atasu, Kubilay},
  booktitle = {Advances in Neural Information Processing Systems 36,
               Datasets and Benchmarks Track},
  year      = {2023}
}
```

## The three states, and what changes at each

The point of writing them down is that none of them is a judgement call at the
moment it applies.

**`committed`, now until notification.** Say "an ARR submission committed to
EMNLP 2026 (Resources and Evaluation)". Do not write `[EMNLP 2026]` as a bare
stamp anywhere, do not write `@inproceedings`, and keep the OpenReview URL. The
paper is committed, not accepted, and a bare venue stamp claims otherwise.

**`accepted`, from notification until the Anthology record exists.** Add
`[EMNLP 2026]` to the GitHub repository description. BibTeX stays `@misc` and
gains `note = {To appear}`. Do not fabricate `pages` or an `aclanthology.org`
URL; neither exists yet.

**`published`, once the Anthology record exists.** Switch to `@inproceedings`
with the real `url` and `pages`, add `conference:` to `CITATION.cff`, and
replace the OpenReview `url:` there.

If the paper is not accepted, the state goes to `preprint`: drop the venue
phrase entirely, keep the `@misc`, and point at whatever record replaces the
OpenReview one.

## Where the copies are

`make check-citation` reads this file and asserts each of these matches.

| File | What it carries |
|---|---|
| `README.md` | the venue phrase in the binding sentence, and the BibTeX block |
| `CITATION.cff` | title, authors, year, `url` |
| `pyproject.toml` | the title in `description`, in paraphrase |
| the dataset card | the venue phrase and a byte-identical BibTeX block |
