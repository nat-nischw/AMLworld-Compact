#!/usr/bin/env python3
"""Download the Elliptic Bitcoin Dataset (Weber et al., 2019).

The canonical PyG mirror (`data.pyg.org/datasets/elliptic/*.zip`) currently
returns HTTP 403 from generic clients (the AWS S3 bucket has been locked
down), so we default to a public Hugging Face mirror instead.

Three sources are supported:

  --source hf      (default) Pull `yhoma/elliptic-bitcoin-dataset` from
                   the Hugging Face Hub. Public, no auth required, ~690 MB.

  --source pyg     Try the PyG S3 bucket. Will fail with 403 from most
                   networks today; kept for completeness.

  --source kaggle  Use the kaggle CLI (`pip install kaggle` and configure
                   `~/.kaggle/kaggle.json`) to pull
                   `ellipticco/elliptic-data-set`.

All three routes write the files unchanged to:

    data/elliptic/raw/elliptic_txs_features.csv   (~689 MB)
    data/elliptic/raw/elliptic_txs_classes.csv    (~3   MB)
    data/elliptic/raw/elliptic_txs_edgelist.csv   (~4   MB)
"""
from __future__ import annotations

import argparse
import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "data" / "elliptic" / "raw"

HF_REPO  = "yhoma/elliptic-bitcoin-dataset"
PYG_URL  = "https://data.pyg.org/datasets/EllipticBitcoinDataset.zip"

EXPECTED_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)


def already_present(out_dir: Path) -> bool:
    return all((out_dir / name).exists() for name in EXPECTED_FILES)


def download_hf(out_dir: Path) -> None:
    """Pull the three CSVs from a public Hugging Face dataset mirror."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("[download] huggingface_hub not installed; "
                 "`pip install huggingface_hub` (it ships with transformers).")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_FILES:
        print(f"[download] HF {HF_REPO} → {name}")
        local = hf_hub_download(
            repo_id=HF_REPO, filename=name, repo_type="dataset",
            local_dir=str(out_dir), local_dir_use_symlinks=False,
        )
        # hf_hub_download may put the file at out_dir/<name>; if it placed
        # it elsewhere (older API), copy it in.
        target = out_dir / name
        if Path(local).resolve() != target.resolve():
            shutil.copy2(local, target)
        print(f"  → {target}  ({target.stat().st_size/1e6:6.1f} MB)")


def download_pyg(out_dir: Path) -> None:
    """Pull the bundled zip from the PyG S3 mirror.

    Currently 403s for most networks; kept as a documented option in case
    the bucket is reopened.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] GET {PYG_URL}")
    req = urllib.request.Request(PYG_URL, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"[download] PyG mirror failed ({e.code}); fall back to "
                 "`--source hf` (default) or `--source kaggle`.")
    print(f"[download] received {len(blob)/1e6:.1f} MB; extracting…")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for member in zf.namelist():
            base = Path(member).name
            if base in EXPECTED_FILES:
                with zf.open(member) as src, (out_dir / base).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  → {out_dir / base}")
    if not already_present(out_dir):
        sys.exit("[download] PyG zip did not contain the expected CSVs.")


def download_kaggle(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        sys.exit("[download] kaggle CLI not on PATH; "
                 "`pip install kaggle` and configure ~/.kaggle/kaggle.json")
    print("[download] kaggle datasets download ellipticco/elliptic-data-set")
    subprocess.check_call(
        ["kaggle", "datasets", "download", "-d",
         "ellipticco/elliptic-data-set", "-p", str(out_dir), "--unzip"],
    )
    if not already_present(out_dir):
        sys.exit("[download] expected CSVs missing after kaggle pull")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "pyg", "kaggle"], default="hf")
    ap.add_argument("--out-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--force", action="store_true",
                    help="redownload even if files are already present")
    args = ap.parse_args()

    if not args.force and already_present(args.out_dir):
        print(f"[download] CSVs already present at {args.out_dir}; "
              "skip (use --force to redownload)")
        return

    if args.source == "hf":
        download_hf(args.out_dir)
    elif args.source == "pyg":
        download_pyg(args.out_dir)
    else:
        download_kaggle(args.out_dir)

    print(f"\n[download] done → {args.out_dir}")
    for name in EXPECTED_FILES:
        path = args.out_dir / name
        if path.exists():
            print(f"  {name}: {path.stat().st_size/1e6:7.1f} MB")
        else:
            print(f"  {name}: MISSING")


if __name__ == "__main__":
    main()
