"""Build corpus/sentences.txt from the downloaded LJSpeech filelist.

Extracts the transcript text from each line of corpus/_ljs_filelist.txt
(format: <path>.wav|<text>) and writes cleaned, length-filtered sentences,
one per line, to corpus/sentences.txt.

These are public-domain transcripts used by the VITS reference repo, suitable
as phonetically diverse TTS training prompts.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "corpus" / "_ljs_filelist.txt"
OUT = ROOT / "corpus" / "sentences.txt"

MIN_CHARS = 20
MAX_CHARS = 220


def clean(text: str) -> str:
    text = text.strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}. Download the filelist first.")
    seen = set()
    out_lines = []
    for line in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "|" not in line:
            continue
        text = clean(line.split("|", 1)[1])
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(text)
    OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} sentences to {OUT}")


if __name__ == "__main__":
    main()
