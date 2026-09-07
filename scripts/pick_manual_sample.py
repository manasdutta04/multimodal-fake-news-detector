"""Print 1 FAKE + 1 REAL sample that has a cached image (for manual Streamlit tests).

Usage (from repo root):
  python scripts/pick_manual_sample.py
  python scripts/pick_manual_sample.py --split test --index 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--split", choices=["test", "validate", "train"], default="test")
    parser.add_argument("--index", type=int, default=0, help="Which cached row per label (0, 1, 2, …)")
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    tsv_name = {
        "test": "multimodal_test_public.tsv",
        "validate": "multimodal_validate.tsv",
        "train": "multimodal_train.tsv",
    }[args.split]
    tsv = dataset / tsv_name
    cache = dataset / "image_cache"

    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    df["id"] = df["id"].astype(str)
    df["path"] = df["id"].map(lambda i: cache / f"{i.replace('/', '_')}.jpg")
    df = df[df["path"].map(lambda p: p.exists())].copy()
    df = df[df["clean_title"].notna() & (df["clean_title"].astype(str).str.strip() != "")]

    if len(df) == 0:
        raise SystemExit(
            f"No cached images under {cache}.\n"
            "Run: python scripts/batch_eval_from_tsv.py --n 40 --download"
        )

    for lab, name in [(0, "FAKE"), (1, "REAL")]:
        sub = df[df["2_way_label"] == lab]
        if len(sub) == 0:
            print(f"No cached rows for {name}")
            continue
        i = min(args.index, len(sub) - 1)
        r = sub.iloc[i]
        print("=" * 60)
        print(f"TRUE LABEL: {name}  (2_way_label={lab})")
        print(f"id:    {r['id']}")
        print(f"title: {r['clean_title']}")
        print(f"image: {Path(r['path']).resolve()}")
    print("=" * 60)
    print("In Streamlit: paste title, upload that image file, click Predict.")


if __name__ == "__main__":
    main()
