#!/usr/bin/env python3
"""Assert the frontier-probe templates reproduce the executed prompts exactly.

Renders each of the six variants against the nine pilot cases and compares
byte-for-byte with the prompt files that were sent to the APIs. Needs the
archived pilot directory; skips with a clear message when it is absent.
"""

import glob
import os
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "prompts" / "frontier_probe"

#: released name -> directory name in the archived pilot tree
VARIANTS = {
    "zs_graph_classification": "V0_nofs_graph_classification",
    "zs_base": "V1_nofs_baseline",
    "fs_base": "V2_fs_baseline",
    "zs_abstain": "V4p1_nofs_abstain_softened",
    "fs_cot_elim": "V7_fs_cot_elimination",
    "fs_typfirst": "V8_fs_typology_first",
}


def main() -> None:
    src = os.environ.get("AMLC_PILOT_DIR")
    if not src or not Path(src).is_dir():
        print("AMLC_PILOT_DIR is not set to the archived pilot prompt "
              "directory; skipping. The templates are the released artefact and "
              "this check only re-proves them against the originals.")
        return

    env = Environment(loader=FileSystemLoader(str(PROBE)),
                      undefined=StrictUndefined, keep_trailing_newline=True)
    failed = 0
    for name, legacy in VARIANTS.items():
        tmpl = (PROBE / f"{name}_user.j2").read_text()
        pre, suf = tmpl.split("{{ graph_text }}")
        exact = 0
        cases = sorted(glob.glob(f"{src}/{legacy}/user_*.txt"))
        for p in cases:
            want = Path(p).read_text()
            graph = want[len(pre):len(want) - len(suf)]
            exact += env.get_template(f"{name}_user.j2").render(graph_text=graph) == want
        sys_ok = (env.get_template(f"{name}_system.j2").render()
                  == Path(f"{src}/{legacy}/system_prompt.txt").read_text())
        status = "OK  " if exact == len(cases) and sys_ok else "FAIL"
        print(f"  {status} {name:24s} user {exact}/{len(cases)}  "
              f"system {'exact' if sys_ok else 'MISMATCH'}")
        failed += exact != len(cases) or not sys_ok

    if failed:
        sys.exit(f"{failed} probe template(s) do not reproduce the executed prompt")
    print("all frontier-probe templates reproduce the executed prompts byte-for-byte")


if __name__ == "__main__":
    main()
