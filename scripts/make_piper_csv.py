"""Convert our LJSpeech-style metadata.csv (`id|text`) into the CSV format the
new Piper trainer expects (`id.wav|text`), written next to the audio as
piper_metadata.csv.

Usage:
  python scripts/make_piper_csv.py --voice david
  python scripts/make_piper_csv.py --voice all
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VOICES = ["david", "zira", "mark", "heera", "ravi"]


def convert(voice: str) -> None:
    vdir = DATA / voice
    src = vdir / "metadata.csv"
    if not src.exists():
        print(f"[skip] {voice}: no metadata.csv")
        return
    out = vdir / "piper_metadata.csv"
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        uid, text = line.split("|", 1)
        rows.append(f"{uid}.wav|{text}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[done] {voice}: wrote {len(rows)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="all")
    args = ap.parse_args()
    voices = VOICES if args.voice == "all" else [args.voice]
    for v in voices:
        convert(v)


if __name__ == "__main__":
    main()
