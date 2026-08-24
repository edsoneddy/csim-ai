#!/usr/bin/env python3
"""Download and verify the ConPlag dataset (CC-BY-4.0, Slobodkin &
Sadovnikov 2023, https://zenodo.org/records/7332790). Eval-only, per
section 5 of the project brief -- never used for training.

Usage:
    python training/eval/external/fetch_conplag.py
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/7332790/files/conplag.zip?download=1"
MD5 = "76451f93984acb0369acc9d8951f82db"
DEST = Path(__file__).parent / "conplag"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    zip_path = DEST / "conplag.zip"

    if not zip_path.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)

    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if digest != MD5:
        raise SystemExit(f"md5 mismatch: got {digest}, expected {MD5}")
    print("md5 ok")

    out_dir = DEST / "extracted"
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    print(f"extracted to {out_dir}")


if __name__ == "__main__":
    main()
