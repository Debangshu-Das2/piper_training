"""Safely inspect/sanitize an untrusted Piper checkpoint.

Loads with weights_only=True (restricted unpickler — cannot execute arbitrary
code). If that succeeds, the file contains only safe objects, and we re-save a
sanitized copy (built from the safe-loaded data) that is safe to use normally.

Usage:
  python scripts/verify_ckpt.py checkpoints/base/en_US-lessac-medium-pl2.ckpt
"""
import sys
from pathlib import Path

import torch


def main():
    src = Path(sys.argv[1])
    print(f"Inspecting {src} ({src.stat().st_size/1e6:.1f} MB)")

    try:
        ckpt = torch.load(src, map_location="cpu", weights_only=True)
        print("weights_only=True load: SUCCESS -> file contains only safe objects")
    except Exception as exc:
        print("weights_only=True load: FAILED")
        print("  reason:", repr(exc)[:500])
        print("=> Will NOT do a normal (code-executing) load. "
              "Next step would be a state_dict-only extraction.")
        return

    keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)
    print("top-level keys:", keys)
    if isinstance(ckpt, dict):
        print("  global_step:", ckpt.get("global_step"))
        print("  epoch:", ckpt.get("epoch"))
        hp = ckpt.get("hyper_parameters", {})
        if isinstance(hp, dict):
            print("  hp.sample_rate:", hp.get("sample_rate"),
                  "| filter_length:", hp.get("filter_length"),
                  "| hop_length:", hp.get("hop_length"),
                  "| num_speakers:", hp.get("num_speakers"))
        sd = ckpt.get("state_dict", {})
        print("  state_dict tensors:", len(sd))
        print("  has optimizer_states:", "optimizer_states" in ckpt)

    # Re-save a sanitized copy from the safe-loaded data.
    out = src.with_name(src.stem.replace("-pl2", "") + "-SAFE.ckpt")
    torch.save(ckpt, out)
    print(f"sanitized copy written: {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
