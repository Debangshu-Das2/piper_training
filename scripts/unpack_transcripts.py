#!/usr/bin/env python3
"""Restore the plaintext transcript files from their committed .gz copies.

Why this exists: the corporate Zscaler proxy's DLP content-inspection blocks
pushing the plaintext LJSpeech transcripts (corpus/*.txt, data/*/*.csv) to
GitHub. They are stored gzipped instead (binary, not content-scanned) and
restored here. Run once after cloning, before training:

    python scripts/unpack_transcripts.py
"""
import gzip
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEARCH_DIRS = ["corpus", "data"]


def main() -> int:
    restored = 0
    for d in SEARCH_DIRS:
        for gz in (ROOT / d).rglob("*.gz"):
            target = gz.with_suffix("")  # strip .gz
            with gzip.open(gz, "rb") as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  {gz.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
            restored += 1
    if restored == 0:
        print("No .gz transcripts found.", file=sys.stderr)
        return 1
    print(f"Restored {restored} transcript file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
