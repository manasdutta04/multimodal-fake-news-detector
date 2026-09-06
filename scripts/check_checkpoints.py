"""Validate that Module 01–03 checkpoint folders exist and look complete.

Usage (from repo root):
  python scripts/check_checkpoints.py
  python scripts/check_checkpoints.py --dir path/to/checkpoints
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED = [
    ("module01_distilbert/config.json", "Module 01 DistilBERT config"),
    ("module01_distilbert/model.safetensors", "Module 01 DistilBERT weights"),
    ("module01_distilbert/tokenizer.json", "Module 01 tokenizer"),
    ("module02_resnet50/model.pt", "Module 02 ResNet50 checkpoint"),
    ("module03_fusion/fusion.pt", "Module 03 fusion heads"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default=None,
        help="Checkpoints directory (default: CHECKPOINTS_DIR, else dataset/checkpoints)",
    )
    args = parser.parse_args()

    import os

    root = Path(
        args.dir
        or os.environ.get("CHECKPOINTS_DIR")
        or ("dataset/checkpoints" if Path("dataset/checkpoints").exists() else "checkpoints")
    )
    print(f"Checking: {root.resolve()}")
    if not root.exists():
        print("FAIL: directory does not exist")
        print("Expected layout: dataset/checkpoints/  (same as Google Drive)")
        print("Or set CHECKPOINTS_DIR to that folder.")
        return 1

    ok = True
    for rel, label in REQUIRED:
        path = root / rel
        # Allow pytorch_model.bin as alternate DistilBERT weight name
        if not path.exists() and rel.endswith("model.safetensors"):
            alt = root / rel.replace("model.safetensors", "pytorch_model.bin")
            if alt.exists():
                print(f"OK   {label}: {alt.name}")
                continue
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"OK   {label} ({size_mb:.1f} MB)")
        else:
            print(f"MISS {label}: expected {path}")
            ok = False

    if ok:
        print("\nAll required checkpoints found. You can run:")
        print("  python -m streamlit run app/streamlit_app.py")
        return 0

    print("\nIncomplete checkpoints — demo will not load until the MISS lines are fixed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
