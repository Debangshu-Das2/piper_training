"""Package the generated voice datasets for upload to Google Colab.

Converts each voice's 16 kHz WAVs to FLAC (lossless, ~half the size) to keep the
upload small, writes a matching metadata.csv (`<id>.flac|text`), and zips
everything into colab_upload/piper_dataset.zip with this layout:

    <voice>/metadata.csv
    <voice>/<id>.flac
    ...

The Colab notebook unzips this and trains directly from it.

Usage:
  python scripts/package_for_colab.py --voices all
  python scripts/package_for_colab.py --voices david        # smaller pilot upload
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTDIR = ROOT / "colab_upload"
VOICES = ["david", "zira", "mark", "heera", "ravi"]


def package_voice(voice: str, stage: Path) -> int:
    vdir = DATA / voice
    meta = vdir / "metadata.csv"
    if not meta.exists():
        print(f"[skip] {voice}: no metadata.csv")
        return 0
    dst = stage / voice
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for line in meta.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        uid, text = line.split("|", 1)
        wav = vdir / "wav" / f"{uid}.wav"
        if not wav.exists():
            continue
        audio, sr = sf.read(str(wav), dtype="float32")
        sf.write(str(dst / f"{uid}.flac"), audio, sr, format="FLAC", subtype="PCM_16")
        rows.append(f"{uid}.flac|{text}")
        n += 1
    (dst / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  {voice}: packaged {n} clips")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="all")
    args = ap.parse_args()
    voices = VOICES if args.voices == "all" else [v.strip() for v in args.voices.split(",")]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stage = OUTDIR / "_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    total = 0
    for v in voices:
        total += package_voice(v, stage)

    name = f"piper_dataset_{voices[0]}.zip" if len(voices) == 1 else "piper_dataset.zip"
    zip_path = OUTDIR / name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:  # FLAC already compressed
        for f in stage.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(stage))
    shutil.rmtree(stage)
    mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[done] {total} clips across {len(voices)} voice(s) -> {zip_path} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
