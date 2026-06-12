"""Launch a local CPU fine-tune of Piper for one voice.

Optimized for a modest CPU:
  - uses all physical cores (OMP/MKL threads),
  - reduces the training segment_size (less vocoder/discriminator compute per
    step; does NOT change weight shapes, so checkpoint loading still works),
  - auto-detects sample_rate + mel params from the base checkpoint so the model
    architecture matches the weights being loaded,
  - periodic checkpointing so a long run survives interruption.

Usage (after placing the base .ckpt in checkpoints/base/):
  python scripts/train_local.py --voice david --base-ckpt checkpoints/base/<file>.ckpt --extra-steps 1500

Smoke test of the launch itself (from scratch, no checkpoint):
  python scripts/train_local.py --voice david --smoke
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def detect_arch(base_ckpt: Path):
    """Read sample_rate + mel params from the base checkpoint's hyperparameters
    so the instantiated model matches the weights."""
    ck = torch.load(base_ckpt, map_location="cpu")
    hp = ck.get("hyper_parameters", {}) or {}
    base_step = int(ck.get("global_step", 0))
    arch = {
        "sample_rate": hp.get("sample_rate", 22050),
        "filter_length": hp.get("filter_length", 1024),
        "hop_length": hp.get("hop_length", 256),
        "win_length": hp.get("win_length", 1024),
    }
    return arch, base_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--base-ckpt", default=None, help="base checkpoint to fine-tune from")
    ap.add_argument("--extra-steps", type=int, default=1500, help="training steps beyond the base checkpoint")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--segment", type=int, default=4096, help="training audio segment (smaller = faster on CPU)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=200, help="checkpoint every N steps")
    ap.add_argument("--smoke", action="store_true", help="2-step from-scratch run to validate the launch")
    args = ap.parse_args()

    vdir = ROOT / "data" / args.voice
    csv = vdir / "piper_metadata.csv"
    audio = vdir / "wav"
    assert csv.exists(), f"missing {csv} (run make_piper_csv.py)"

    work = ROOT / "checkpoints" / ("work_smoke" if args.smoke else "work") / args.voice
    work.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "checkpoints" / ("cache_smoke" if args.smoke else "cache") / args.voice

    if args.smoke:
        # tiny BOM-free subset for a fast launch/syntax validation
        lines = csv.read_text(encoding="utf-8").splitlines()[:30]
        csv = work / "_smoke.csv"
        csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        sample_rate, filter_length, hop_length, win_length = 22050, 1024, 256, 1024
        max_steps = 2
        base_ckpt = None
    else:
        assert args.base_ckpt, "--base-ckpt is required (or use --smoke)"
        base_ckpt = Path(args.base_ckpt)
        assert base_ckpt.exists(), f"missing base checkpoint: {base_ckpt}"
        arch, base_step = detect_arch(base_ckpt)
        sample_rate = arch["sample_rate"]
        filter_length = arch["filter_length"]
        hop_length = arch["hop_length"]
        win_length = arch["win_length"]
        max_steps = base_step + args.extra_steps
        print(f"base global_step={base_step} -> training to max_steps={max_steps} "
              f"(+{args.extra_steps}); sample_rate={sample_rate}")

    cmd = [
        PY, "-m", "piper.train", "fit",
        "--data.voice_name", args.voice,
        "--data.csv_path", str(csv),
        "--data.audio_dir", str(audio),
        "--data.cache_dir", str(cache),
        "--data.config_path", str(work / "config.json"),
        "--data.espeak_voice", "en-us",
        "--data.batch_size", str(args.batch),
        "--model.sample_rate", str(sample_rate),
        "--model.filter_length", str(filter_length),
        "--model.hop_length", str(hop_length),
        "--model.win_length", str(win_length),
        "--model.segment_size", str(args.segment),
        "--trainer.default_root_dir", str(work),
        "--trainer.accelerator", "cpu",
        "--trainer.precision", "32",
        "--trainer.max_steps", str(max_steps),
        "--trainer.num_sanity_val_steps", "0",
        "--trainer.limit_val_batches", "0",
        "--trainer.log_every_n_steps", "10",
        # periodic checkpointing so a long CPU run survives interruption
        "--trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint",
        f"--trainer.callbacks.dirpath={work / 'ckpts'}",
        f"--trainer.callbacks.every_n_train_steps={args.save_every}",
        "--trainer.callbacks.save_top_k=-1",
        "--trainer.callbacks.save_last=true",
    ]
    if base_ckpt is not None:
        cmd += ["--ckpt_path", str(base_ckpt)]

    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(args.threads)

    print(">>>", " ".join(cmd))
    raise SystemExit(subprocess.run(cmd, env=env).returncode)


if __name__ == "__main__":
    main()
