"""Generate colab/piper_finetune.ipynb — a self-contained Colab notebook that
fine-tunes Piper on the uploaded Windows-voice dataset and exports ONNX.

We generate the .ipynb as JSON (no nbformat dependency). The numba MAS patch is
embedded base64-encoded to avoid any quote-escaping problems inside cells.
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab" / "piper_finetune.ipynb"

# --- numba Monotonic Alignment Search (no triple quotes, # comments only) ---
NUMBA_SRC = """# Monotonic Alignment Search (numba), drop-in for the Cython core.
import numba
import numpy as np
import torch


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)
def _maximum_path_jit(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(b):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]
        v_prev = v_cur = 0.0
        index = t_x - 1
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)
        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
                index = index - 1


def maximum_path(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.zeros(neg_cent.shape, dtype=np.int32)
    t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
    _maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
"""

NUMBA_B64 = base64.b64encode(NUMBA_SRC.encode()).decode()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(md(
"""# Fine-tune Piper on cloned Windows voices → ONNX

This notebook fine-tunes [Piper](https://github.com/OHF-Voice/piper1-gpl) on the
voices captured from your Windows machine (David, Zira, Mark, Heera, Ravi) and
exports an Android-ready `.onnx` + `.onnx.json` per voice.

**Before running:** Runtime → Change runtime type → **GPU** (T4 is fine).

**You only need to upload one file:** `piper_dataset.zip` (produced locally by
`scripts/package_for_colab.py`). The notebook downloads the base checkpoint
itself from Hugging Face (reachable here, unlike your corporate network).

Run the cells top to bottom."""))

cells.append(code(
"""# 1. Check the GPU
!nvidia-smi -L || echo 'No GPU! Set Runtime > Change runtime type > GPU'
import torch
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"""))

cells.append(code(
"""# 2. Install Piper trainer + add the (wheel-omitted) VITS training package,
#    patching the Cython alignment kernel with a numba one (no compiler needed).
import base64, os, shutil, subprocess, sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'piper-tts[train]==1.4.2', 'huggingface_hub', 'onnxscript'], check=True)

subprocess.run(['git', 'clone', '-q', '--branch', 'v1.4.2', '--depth', '1',
                'https://github.com/OHF-Voice/piper1-gpl.git', '/content/piper1-gpl'], check=True)

import piper
dst = os.path.join(os.path.dirname(piper.__file__), 'train', 'vits')
src = '/content/piper1-gpl/src/piper/train/vits'
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)

# Replace Cython monotonic_align with the numba implementation.
mas_path = os.path.join(dst, 'monotonic_align', '__init__.py')
open(mas_path, 'w').write(base64.b64decode('%s').decode())
print('patched MAS ->', mas_path)

# torch>=2.12 defaults to the dynamo ONNX exporter, which fails on VITS.
# Force the legacy exporter in export_onnx.py.
exp = os.path.join(os.path.dirname(piper.__file__), 'train', 'export_onnx.py')
_t = open(exp).read()
if 'dynamo=False' not in _t:
    _t = _t.replace('opset_version=OPSET_VERSION,', 'dynamo=False,\\n        opset_version=OPSET_VERSION,')
    open(exp, 'w').write(_t)
    print('patched export_onnx -> dynamo=False')

# Sanity check: trainer imports and MAS runs.
from piper.train.vits.monotonic_align import maximum_path
import torch
p = maximum_path(torch.randn(2, 5, 7), torch.ones(2, 5, 7))
print('trainer OK, MAS path sums:', p.sum(dim=(1, 2)).tolist())""" % NUMBA_B64))

cells.append(code(
"""# 3. Upload piper_dataset.zip and unzip it.
#    (Alternative: mount Google Drive and unzip from there — see commented lines.)
import os, zipfile
from google.colab import files

up = files.upload()                       # choose piper_dataset.zip
zip_name = next(iter(up))
os.makedirs('/content/dataset', exist_ok=True)
with zipfile.ZipFile(zip_name) as zf:
    zf.extractall('/content/dataset')

# from google.colab import drive
# drive.mount('/content/drive')
# !unzip -q -o /content/drive/MyDrive/piper_dataset.zip -d /content/dataset

voices = sorted(d for d in os.listdir('/content/dataset')
                if os.path.isdir(f'/content/dataset/{d}'))
print('voices found:', voices)
for v in voices:
    n = len([f for f in os.listdir(f'/content/dataset/{v}') if f.endswith('.flac')])
    print(f'  {v}: {n} clips')"""))

cells.append(code(
"""# 4. Download a base checkpoint from Hugging Face (prefer a low/16k en_US voice;
#    fall back to lessac medium/22.05k). We read the sample rate from the path.
from huggingface_hub import hf_hub_download, list_repo_files

REPO = 'rhasspy/piper-checkpoints'
all_files = list_repo_files(REPO, repo_type='dataset')
ckpts = [f for f in all_files if f.endswith('.ckpt') and '/en/en_US/' in f]
low = [f for f in ckpts if '/low/' in f]
med = [f for f in ckpts if '/medium/' in f and 'lessac' in f] or [f for f in ckpts if '/medium/' in f]
choice = (low or med or ckpts)[0]
SAMPLE_RATE = 16000 if '/low/' in choice else 22050
print('base checkpoint:', choice, '| sample_rate:', SAMPLE_RATE)

BASE_CKPT = hf_hub_download(REPO, choice, repo_type='dataset')
print('downloaded ->', BASE_CKPT)

import torch
base_step = int(torch.load(BASE_CKPT, map_location='cpu').get('global_step', 0))
print('base global_step:', base_step)"""))

cells.append(code(
"""# 5. Fine-tune. Edit VOICES and EXTRA_STEPS as you like.
#    EXTRA_STEPS ~2000-4000 is plenty to clone a clean, consistent TTS voice.
import glob, os, subprocess, sys

VOICES = ['david']            # e.g. ['david','zira','mark','heera','ravi']
EXTRA_STEPS = 3000
BATCH_SIZE = 16

os.makedirs('/content/out', exist_ok=True)

def finetune(voice):
    dsdir = f'/content/dataset/{voice}'
    work = f'/content/work/{voice}'
    os.makedirs(work, exist_ok=True)
    cmd = [sys.executable, '-m', 'piper.train', 'fit',
           '--data.voice_name', voice,
           '--data.csv_path', f'{dsdir}/metadata.csv',
           '--data.audio_dir', dsdir,
           '--data.cache_dir', f'/content/cache/{voice}',
           '--data.config_path', f'{work}/config.json',
           '--data.espeak_voice', 'en-us',
           '--data.batch_size', str(BATCH_SIZE),
           '--model.sample_rate', str(SAMPLE_RATE),
           '--ckpt_path', BASE_CKPT,
           '--trainer.max_steps', str(base_step + EXTRA_STEPS),
           '--trainer.default_root_dir', work,
           '--trainer.precision', '16-mixed',
           '--trainer.log_every_n_steps', '20']
    print('>>>', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return work

for v in VOICES:
    finetune(v)
print('training done for:', VOICES)"""))

cells.append(code(
"""# 6. Export each trained voice to ONNX (+ .onnx.json) and zip for download.
import glob, os, shutil, subprocess, sys
from google.colab import files

os.makedirs('/content/out', exist_ok=True)

def export(voice):
    work = f'/content/work/{voice}'
    ckpts = glob.glob(f'{work}/**/*.ckpt', recursive=True)
    assert ckpts, f'no checkpoint produced for {voice}'
    latest = max(ckpts, key=os.path.getmtime)
    onnx_path = f'/content/out/{voice}.onnx'
    subprocess.run([sys.executable, '-m', 'piper.train.export_onnx',
                    '--checkpoint', latest, '--output-file', onnx_path], check=True)
    # The training config.json is the inference .onnx.json
    shutil.copyfile(f'{work}/config.json', onnx_path + '.json')
    print(f'{voice}: {os.path.getsize(onnx_path)/1e6:.1f} MB  <- {os.path.basename(latest)}')

for v in VOICES:
    export(v)

shutil.make_archive('/content/piper_onnx_models', 'zip', '/content/out')
print('zipped ->', '/content/piper_onnx_models.zip')
files.download('/content/piper_onnx_models.zip')"""))

cells.append(md(
"""## Next steps
- Download `piper_onnx_models.zip` — each voice has `<voice>.onnx` + `<voice>.onnx.json`.
- Back on the Windows machine these are tested and packaged for Android (sherpa-onnx)
  with a runtime voice picker.
- To add the other voices, set `VOICES = ['zira','mark','heera','ravi']` in cell 5,
  re-run cells 5–6. (Re-running cell 4 is not needed.)
- If `load_from_checkpoint` errors on key mismatch, tell me the error — we'll pick a
  different base checkpoint."""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
