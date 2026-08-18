"""Fetch the four AMLworld files this pipeline needs, and check them.

This repository does not redistribute AMLworld. The dataset is IBM's, released
under a Community Data License Agreement on Kaggle, and the release ships only
derived artefacts: coreset indices, Horvitz-Thompson weights, labels and
typologies. Anyone rebuilding the pipeline from raw transactions downloads the
originals under their own Kaggle account, which is what this module does.

The Kaggle dataset holds four AMLworld variants and runs to tens of gigabytes.
Only two of them appear in the paper, so the files are fetched one at a time
rather than as a bulk download:

    <data>/HI-Small_Trans.csv       the transaction table
    <data>/HI-Small_Patterns.txt    the laundering-attempt typology labels
    <data>/LI-Small_Trans.csv
    <data>/LI-Small_Patterns.txt

Both loaders in this package read exactly those paths, through
:func:`trans_csv` and :func:`patterns_txt`, so nothing else has to know how the
files are named.

Integrity is checked against ``data/checksums/amlworld_sha256.txt``. When that
file is absent, the digests are computed and written, and printed so the author
can commit them; from then on every download is verified against them. A
mismatch raises. Kaggle datasets are versioned and can be replaced in place, so
a silent content change is the failure this guards against: it would move every
number in the paper without moving a line of code.

Usage::

    python -m amlc.data.download_amlworld
    python -m amlc.data.download_amlworld --source kaggle --force
    python -m amlc.data.download_amlworld --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from .. import paths
from .. import config
from ..config import DATASETS

#: The Kaggle listing AMLworld is published on.
KAGGLE_DATASET = config.KAGGLE_DATASET

#: Filename suffixes AMLworld uses for each variant.
TRANS_SUFFIX = "_Trans.csv"
PATTERNS_SUFFIX = "_Patterns.txt"

#: Where the digests live, relative to the data directory.
CHECKSUM_DIRNAME = "checksums"
CHECKSUM_FILENAME = "amlworld_sha256.txt"

_READ_CHUNK = 1 << 20


def trans_csv(dataset: str, data_dir: Optional[Path] = None) -> Path:
    """Path to one variant's transaction CSV."""
    return (data_dir or paths.data()) / f"{dataset}{TRANS_SUFFIX}"


def patterns_txt(dataset: str, data_dir: Optional[Path] = None) -> Path:
    """Path to one variant's laundering-pattern file."""
    return (data_dir or paths.data()) / f"{dataset}{PATTERNS_SUFFIX}"


def required_files(
    datasets: Iterable[str] = DATASETS,
    data_dir: Optional[Path] = None,
) -> list[Path]:
    """Every AMLworld file the pipeline needs, in a stable order."""
    out: list[Path] = []
    for dataset in datasets:
        out.append(trans_csv(dataset, data_dir))
        out.append(patterns_txt(dataset, data_dir))
    return out


def checksum_file(data_dir: Optional[Path] = None) -> Path:
    """Path to the SHA256 manifest."""
    return (data_dir or paths.data()) / CHECKSUM_DIRNAME / CHECKSUM_FILENAME


def sha256(path: Path) -> str:
    """SHA256 of a file, read in chunks because these run to gigabytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_READ_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksums(data_dir: Optional[Path] = None) -> dict[str, str]:
    """Parse the manifest into ``{filename: digest}``; empty when absent.

    The format is the one ``sha256sum`` writes, so ``sha256sum -c`` also works
    on it from the data directory.
    """
    path = checksum_file(data_dir)
    if not path.exists():
        return {}
    digests: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        if digest and name:
            digests[name] = digest
    return digests


def write_checksums(
    digests: dict[str, str],
    data_dir: Optional[Path] = None,
) -> Path:
    """Write the manifest and return its path."""
    path = checksum_file(data_dir)
    paths.ensure(path.parent)
    body = "".join(f"{digests[name]}  {name}\n" for name in sorted(digests))
    path.write_text(body)
    return path


def verify(
    files: Iterable[Path],
    data_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Check the files against the manifest, or write it if there is none.

    Returns ``{filename: digest}`` for everything checked. Raises when a file
    is missing, or when its digest differs from the recorded one: at that point
    the download is not the AMLworld the paper's numbers came from, and going
    on would produce results that look valid and are not.
    """
    files = list(files)
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "AMLworld files not present: "
            + ", ".join(str(f) for f in missing)
            + ". Run python -m amlc.data.download_amlworld first."
        )

    computed = {f.name: sha256(f) for f in files}
    recorded = read_checksums(data_dir)

    if not recorded:
        path = write_checksums(computed, data_dir)
        print(f"[amlworld] no checksum manifest; wrote {path}")
        print("[amlworld] commit these digests so later downloads are checked:")
        for name in sorted(computed):
            print(f"  {computed[name]}  {name}")
        return computed

    bad = [
        (name, recorded[name], digest)
        for name, digest in computed.items()
        if name in recorded and recorded[name] != digest
    ]
    if bad:
        detail = "; ".join(
            f"{name}: expected {want}, got {got}" for name, want, got in bad)
        raise ValueError(
            f"AMLworld checksum mismatch ({detail}). The Kaggle listing is "
            "versioned and can be replaced in place, so this is a different "
            "dataset from the one behind the published numbers."
        )

    unrecorded = sorted(set(computed) - set(recorded))
    if unrecorded:
        merged = dict(recorded)
        merged.update({name: computed[name] for name in unrecorded})
        write_checksums(merged, data_dir)
        print("[amlworld] added digests for "
              f"{', '.join(unrecorded)} to {checksum_file(data_dir)}")
        for name in unrecorded:
            print(f"  {computed[name]}  {name}")

    print(f"[amlworld] {len(computed)} file(s) match the recorded SHA256")
    return computed


def _place(downloaded: Path, target: Path) -> None:
    """Move a fetched file into the data directory, unzipping if Kaggle did."""
    paths.ensure(target.parent)
    if downloaded.suffix == ".zip":
        with zipfile.ZipFile(downloaded) as zf:
            member = next(
                (m for m in zf.namelist() if Path(m).name == target.name), None)
            if member is None:
                raise FileNotFoundError(
                    f"{target.name} is not inside {downloaded.name}")
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return
    shutil.copyfile(downloaded, target)


def _fetch_kagglehub(name: str, target: Path) -> None:
    """Fetch one file with kagglehub, which caches under KAGGLEHUB_CACHE."""
    import kagglehub

    print(f"[amlworld] kagglehub {KAGGLE_DATASET} -> {name}")
    local = Path(kagglehub.dataset_download(KAGGLE_DATASET, path=name))
    _place(local, target)


def _fetch_kaggle_cli(name: str, target: Path) -> None:
    """Fetch one file with the kaggle CLI, which needs KAGGLE_USERNAME/KEY."""
    import subprocess

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "the kaggle CLI is not on PATH; pip install kaggle and set "
            "KAGGLE_USERNAME and KAGGLE_KEY, or use --source kagglehub"
        )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        print(f"[amlworld] kaggle datasets download -f {name}")
        subprocess.check_call([
            "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET, "-f", name, "-p", str(tmpdir), "--force",
        ])
        got = list(tmpdir.iterdir())
        if not got:
            raise FileNotFoundError(f"the kaggle CLI fetched nothing for {name}")
        _place(got[0], target)


def download(
    datasets: Iterable[str] = DATASETS,
    data_dir: Optional[Path] = None,
    source: str = "kagglehub",
    force: bool = False,
) -> dict[str, str]:
    """Fetch the AMLworld files for ``datasets`` and verify them.

    Files already present are left alone unless ``force``. Returns the
    ``{filename: digest}`` map that :func:`verify` produced.
    """
    fetch = {"kagglehub": _fetch_kagglehub, "kaggle": _fetch_kaggle_cli}
    if source not in fetch:
        raise ValueError(f"source must be one of {sorted(fetch)}, got {source!r}")

    targets = required_files(datasets, data_dir)
    for target in targets:
        if target.exists() and not force:
            size_gb = target.stat().st_size / 1e9
            print(f"[amlworld] present, skipping: {target.name} ({size_gb:.2f} GB)")
            continue
        fetch[source](target.name, target)
        print(f"[amlworld] wrote {target} "
              f"({target.stat().st_size / 1e9:.2f} GB)")

    return verify(targets, data_dir)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="override the destination; defaults to paths.data()")
    ap.add_argument("--source", choices=["kagglehub", "kaggle"],
                    default="kagglehub")
    ap.add_argument("--force", action="store_true",
                    help="refetch files that are already present")
    ap.add_argument("--verify-only", action="store_true",
                    help="checksum what is on disk and download nothing")
    args = ap.parse_args(argv)

    if args.verify_only:
        verify(required_files(args.datasets, args.data_dir), args.data_dir)
        return
    download(args.datasets, args.data_dir, args.source, args.force)


if __name__ == "__main__":
    main()
