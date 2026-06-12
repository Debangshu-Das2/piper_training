"""Generate LJSpeech-format datasets by cloning Windows voices via WinRT.

For each requested Windows voice, synthesize every sentence in
corpus/sentences.txt using Windows.Media.SpeechSynthesis (OneCore voices,
reachable through the `winsdk` WinRT projection), then:
  - decode the WAV the synthesizer returns,
  - downmix to mono and resample to the target rate (default 16 kHz),
  - peak-normalize and write 16-bit PCM,
  - emit data/<voice_key>/wav/<id>.wav and append `<id>|<text>` to metadata.csv.

No admin rights, no external binaries (ffmpeg) required: audio I/O uses
soundfile, resampling uses soxr.

Usage:
  python scripts/gen_dataset.py --voices all --count 500
  python scripts/gen_dataset.py --voices david --count 5      # smoke test
"""
import argparse
import asyncio
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from winsdk.windows.media.speechsynthesis import SpeechSynthesizer
from winsdk.windows.storage.streams import DataReader

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus" / "sentences.txt"
DATA = ROOT / "data"

# Map short keys -> WinRT voice display names present on this machine.
VOICE_MAP = {
    "david": "Microsoft David",
    "zira": "Microsoft Zira",
    "mark": "Microsoft Mark",
    "heera": "Microsoft Heera",
    "ravi": "Microsoft Ravi",
}


def list_winrt_voices():
    return {v.display_name: v for v in SpeechSynthesizer.all_voices}


async def synth_wav_bytes(synth: SpeechSynthesizer, text: str) -> bytes:
    """Synthesize text and return the raw WAV bytes from the WinRT stream."""
    stream = await synth.synthesize_text_to_stream_async(text)
    size = int(stream.size)
    reader = DataReader(stream.get_input_stream_at(0))
    await reader.load_async(size)
    buf = bytearray(size)
    reader.read_bytes(buf)
    reader.close()
    stream.close()
    return bytes(buf)


def to_pcm16(wav_bytes: bytes, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:  # downmix to mono
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.95 / peak)
    return audio.astype(np.float32)


async def gen_voice(key: str, sentences, target_sr: int):
    voices = list_winrt_voices()
    display = VOICE_MAP[key]
    if display not in voices:
        print(f"[skip] {key}: '{display}' not installed on this machine")
        return
    synth = SpeechSynthesizer()
    synth.voice = voices[display]

    vdir = DATA / key
    wdir = vdir / "wav"
    wdir.mkdir(parents=True, exist_ok=True)
    meta_path = vdir / "metadata.csv"

    rows = []
    n = len(sentences)
    for i, text in enumerate(sentences, start=1):
        uid = f"{key}_{i:04d}"
        try:
            wav_bytes = await synth_wav_bytes(synth, text)
            audio = to_pcm16(wav_bytes, target_sr)
            sf.write(wdir / f"{uid}.wav", audio, target_sr, subtype="PCM_16")
            rows.append(f"{uid}|{text}")
        except Exception as exc:  # keep going; report at end
            print(f"  [err] {uid}: {exc}")
        if i % 50 == 0 or i == n:
            print(f"  {key}: {i}/{n}")
    meta_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[done] {key}: {len(rows)} clips -> {vdir} (sr={target_sr})")


async def main_async(args):
    sentences = [s for s in CORPUS.read_text(encoding="utf-8").splitlines() if s.strip()]
    if args.count > 0:
        sentences = sentences[: args.count]
    if args.voices.lower() == "all":
        keys = list(VOICE_MAP.keys())
    else:
        keys = [k.strip().lower() for k in args.voices.split(",") if k.strip()]
    print(f"Generating {len(sentences)} sentences x {len(keys)} voice(s): {keys}")
    for key in keys:
        if key not in VOICE_MAP:
            print(f"[skip] unknown voice key: {key}")
            continue
        await gen_voice(key, sentences, args.sample_rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="all", help="'all' or comma list: david,zira,mark,heera,ravi")
    ap.add_argument("--count", type=int, default=0, help="limit number of sentences (0 = all)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
