"""Render the in-context prompts from the shipped Jinja2 templates.

The templates in ``prompts/`` are the released form of the prompts. They are
verified byte-identical to the pre-release Python string builders that produced
every result in the paper; ``scripts/verify_prompt_templates.py`` is the test.

    ICL-FS   preamble + 8 suspicious and 4 non-suspicious demonstrations + task
    ICL-ZS   preamble + task, no demonstrations
    ICL-V    ICL-FS with a verification step inserted before the answer format

The demonstration pool ships at ``data/icl_examples/<dataset>/``. It is drawn
from the training split only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import PROMPTINGS
_TEMPLATE = {"ICL-FS": "icl_fs.j2", "ICL-ZS": "icl_zs.j2", "ICL-V": "icl_v.j2"}


def template_dir(root: Optional[Path] = None) -> Path:
    return Path(root) if root else Path(__file__).resolve().parents[2] / "prompts"


def _env(root: Optional[Path] = None) -> Environment:
    # keep_trailing_newline matters: the task blocks end with a newline and the
    # models were shown it.
    return Environment(
        loader=FileSystemLoader(str(template_dir(root))),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )


def load_icl_examples(dataset: str, data_dir: Optional[Path] = None) -> dict:
    """Load the demonstration pool for a dataset."""
    base = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data"
    with (base / "icl_examples" / dataset / "icl_examples.json").open() as f:
        return json.load(f)


def render(prompting: str, graph_text: str,
           icl_examples: Optional[dict] = None,
           template_root: Optional[Path] = None) -> str:
    """Render one prompt.

    ``icl_examples`` is required for ICL-FS and ICL-V and ignored for ICL-ZS.
    """
    if prompting not in _TEMPLATE:
        raise ValueError(f"prompting must be one of {PROMPTINGS}, got {prompting!r}")
    if prompting != "ICL-ZS" and not icl_examples:
        raise ValueError(
            f"{prompting} needs the demonstration pool; load it with "
            "load_icl_examples(dataset)"
        )
    ctx = {"graph_text": graph_text}
    if prompting != "ICL-ZS":
        ctx["suspicious_examples"] = icl_examples["suspicious_examples"]
        ctx["non_suspicious_examples"] = icl_examples["non_suspicious_examples"]
    return _env(template_root).get_template(_TEMPLATE[prompting]).render(**ctx)
