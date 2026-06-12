# Windows-voice → edge TTS (Piper) for Android, with runtime voice selection

Clone the voices installed on this Windows PC into a tiny neural TTS that runs
on Android, and pick the voice at use time.

## Why Piper (not KittenTTS)

KittenTTS is **inference-only** — no public training code, and its 8 voices are
fixed embeddings baked into the ONNX, so it cannot clone your Windows voices.
[Piper](https://github.com/OHF-Voice/piper1-gpl) is the same class of model
(tiny, CPU-friendly, ONNX, runs on Android via sherpa-onnx) but has full
open-source fine-tuning — so it can actually clone a voice.

## Voices on this machine

Captured via WinRT `Windows.Media.SpeechSynthesis`:
David, Zira, Mark (en-US) and Heera, Ravi (en-IN) — 5 voices, 496 clips each.

## Pipeline

```
Windows voices ──(WinRT)──> data/<voice>/wav/*.wav (16 kHz) + metadata.csv
                                   │
                  scripts/package_for_colab.py  (FLAC + zip)
                                   ▼
                       colab_upload/piper_dataset.zip  (285 MB)
                                   │
              colab/piper_finetune.ipynb  (Colab GPU: fine-tune + export)
                                   ▼
                  <voice>.onnx + <voice>.onnx.json  (per voice)
                                   │
                       sherpa-onnx Android bundle + voice picker
```

## Environment notes (corporate machine, no admin)

This machine is locked down; the setup worked around all of it **without admin**:

- **No `.exe` downloads (Zscaler):** Python installed from the NuGet **zip**
  package into `.python/`; venv in `.venv/`.
- **Zscaler TLS interception broke pip SSL:** exported the Windows trust store to
  `certs/corp-ca-bundle.pem`; pip + Python SSL point at it (user env vars +
  `.venv/pip.ini`).
- **No ffmpeg (.exe blocked):** audio I/O via `soundfile`, resampling via `soxr`.
- **Piper training wheel omits the VITS trainer & needs a Cython kernel:** copied
  the `vits/` package from source and replaced `monotonic_align` with a **numba**
  implementation (no C compiler needed). The local trainer runs on CPU.
- **Hugging Face is fully blocked here**, and Piper's base checkpoints live only on
  HF. So fine-tuning runs on **Colab** (HF reachable, free GPU), then the ONNX
  models come back here for testing and Android packaging.

## How to run the fine-tune (Colab)

1. Open `colab/piper_finetune.ipynb` in Google Colab. Runtime → **GPU**.
2. Run cells top to bottom. When prompted, upload
   `colab_upload/piper_dataset.zip`.
   - Quick pilot first? Upload `colab_upload/piper_dataset_david.zip` and keep
     `VOICES = ['david']` in cell 5.
3. The notebook downloads the base checkpoint, fine-tunes, exports ONNX, and
   downloads `piper_onnx_models.zip`.
4. Put the unzipped `*.onnx` / `*.onnx.json` into `export/` here for testing and
   Android packaging.

## Layout

```
.python/        per-user Python (NuGet)        .venv/          virtualenv
certs/          corp CA bundle                 corpus/         sentences.txt
data/<voice>/   wav/ + metadata.csv            colab/          finetune notebook
colab_upload/   dataset zip(s) to upload       export/         trained ONNX (after Colab)
scripts/        gen_dataset, make_corpus, package_for_colab, make_*_csv, make_colab_notebook
```

## Scripts

| Script | Purpose |
|---|---|
| `scripts/make_corpus.py` | Build `corpus/sentences.txt` from the LJSpeech filelist |
| `scripts/gen_dataset.py` | Synthesize each Windows voice → 16 kHz WAV + metadata |
| `scripts/make_piper_csv.py` | `id\|text` → `id.wav\|text` (local training format) |
| `scripts/package_for_colab.py` | FLAC-compress + zip datasets for upload |
| `scripts/make_colab_notebook.py` | Generate the Colab fine-tune notebook |
