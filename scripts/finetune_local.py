"""Fine-tune Piper locally on CPU from a (sanitized, weights-only) checkpoint.

The base checkpoint is weights-only (state_dict + empty hyper_parameters, no
optimizer/global_step), so we instantiate VitsModel with the lessac-medium
config, load the weights, and train with a fresh optimizer. The checkpoint is
loaded with weights_only=True (no code execution).

CPU optimizations: all physical cores, reduced segment_size, num_workers=0
(avoids Windows DataLoader spawn issues), periodic checkpointing.

Usage:
  python scripts/finetune_local.py --voice david --weights checkpoints/base/en_US-lessac-medium-SAFE.ckpt --dry-run
  python scripts/finetune_local.py --voice david --weights checkpoints/base/en_US-lessac-medium-SAFE.ckpt --steps 1500
"""
import argparse
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--weights", required=True, help="sanitized weights-only .ckpt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--segment", type=int, default=4096)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="build model + load weights + exit")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from piper.train.vits.lightning import VitsModel
    from piper.train.vits.dataset import VitsDataModule

    vdir = ROOT / "data" / args.voice
    csv = vdir / "piper_metadata.csv"
    audio = vdir / "wav"
    assert csv.exists(), f"missing {csv}"
    work = ROOT / "checkpoints" / "work" / args.voice
    work.mkdir(parents=True, exist_ok=True)

    # lessac-medium architecture = VitsModel defaults; just set the variable bits.
    model = VitsModel(
        batch_size=args.batch,
        sample_rate=args.sr,
        num_symbols=256,
        num_speakers=1,
        segment_size=args.segment,
    )
    state = torch.load(args.weights, map_location="cpu", weights_only=True)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded weights: {len(state)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  first missing:", missing[:5])
    if unexpected:
        print("  first unexpected:", unexpected[:5])

    if args.dry_run:
        print("dry-run OK: model built and weights loaded.")
        return

    dm = VitsDataModule(
        csv_path=str(csv),
        audio_dir=str(audio),
        cache_dir=str(ROOT / "checkpoints" / "cache" / args.voice),
        config_path=str(work / "config.json"),
        voice_name=args.voice,
        espeak_voice="en-us",
        sample_rate=args.sr,
        batch_size=args.batch,
        segment_size=args.segment,
        num_workers=0,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(work / "ckpts"),
        every_n_train_steps=args.save_every,
        save_top_k=-1,
        save_last=True,
    )
    trainer = L.Trainer(
        max_steps=args.steps,
        accelerator="cpu",
        precision="32-true",
        default_root_dir=str(work),
        callbacks=[ckpt_cb],
        num_sanity_val_steps=0,
        limit_val_batches=0,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    trainer.fit(model, dm)
    print("training finished.")


if __name__ == "__main__":
    main()
