"""Batch-test the demo pipeline against Fakeddit TSV + image_cache.

Runs text-only, image-only, and multimodal on N labeled rows and reports:
  - accuracy vs 2_way_label (0=fake, 1=real)
  - per-sample errors (so intermittent crashes are visible)

Usage (from repo root, on the machine that has dataset/):

  python scripts/batch_eval_from_tsv.py
  python scripts/batch_eval_from_tsv.py --n 50 --split test
  python scripts/batch_eval_from_tsv.py --checkpoints dataset/checkpoints --n 30

Needs:
  dataset/multimodal_*.tsv
  dataset/image_cache/{id}.jpg   (from Module 02)
  dataset/checkpoints/module01_distilbert, module02_resnet50, module03_fusion
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference_pipeline import FakeNewsPipeline, default_checkpoints_dir
from app.preprocessing import LABEL_NAMES


def cache_path(image_cache: Path, sample_id: str) -> Path:
    return image_cache / f"{str(sample_id).replace('/', '_')}.jpg"


def load_paired(tsv: Path, image_cache: Path, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    df["id"] = df["id"].astype(str)
    df["image_path"] = df["id"].map(lambda i: str(cache_path(image_cache, i)))
    df = df[df["image_path"].map(lambda p: Path(p).exists())].copy()
    df = df[df["clean_title"].notna() & (df["clean_title"].astype(str).str.strip() != "")]
    df = df[df["2_way_label"].isin([0, 1])]
    if len(df) == 0:
        raise SystemExit(
            f"No rows with both title and cached image under {image_cache}.\n"
            "Run Module 02 download first, or lower --n after confirming image_cache has files."
        )
    # stratified-ish sample
    parts = []
    per = max(1, n // 2)
    for _, g in df.groupby("2_way_label"):
        parts.append(g.sample(n=min(len(g), per), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset", help="Folder with TSVs + image_cache")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Checkpoints dir (default: CHECKPOINTS_DIR or dataset/checkpoints)",
    )
    parser.add_argument("--split", choices=["test", "validate", "train"], default="test")
    parser.add_argument("--n", type=int, default=40, help="Number of paired samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        default="text,image,multimodal",
        help="Comma list: text,image,multimodal",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    tsv_name = {
        "test": "multimodal_test_public.tsv",
        "validate": "multimodal_validate.tsv",
        "train": "multimodal_train.tsv",
    }[args.split]
    tsv = dataset_dir / tsv_name
    image_cache = dataset_dir / "image_cache"
    ckpt = Path(args.checkpoints or os.environ.get("CHECKPOINTS_DIR") or default_checkpoints_dir())

    print(f"dataset_dir = {dataset_dir.resolve()}")
    print(f"tsv         = {tsv}")
    print(f"image_cache = {image_cache}  exists={image_cache.exists()}")
    print(f"checkpoints = {ckpt.resolve()}")

    if not tsv.exists():
        raise SystemExit(f"Missing TSV: {tsv}")
    if not ckpt.exists():
        raise SystemExit(f"Missing checkpoints: {ckpt}")

    df = load_paired(tsv, image_cache, n=args.n, seed=args.seed)
    print(f"Evaluating {len(df)} paired rows  "
          f"(fake={(df['2_way_label']==0).sum()}, real={(df['2_way_label']==1).sum()})")

    print("Loading models…")
    pipe = FakeNewsPipeline(ckpt)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    stats = {
        m: {"ok": 0, "err": 0, "correct": 0, "total": 0, "errors": []}
        for m in modes
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="batch_eval"):
        title = str(row["clean_title"])
        y = int(row["2_way_label"])
        true_label = LABEL_NAMES[y]
        img_path = row["image_path"]

        for mode in modes:
            stats[mode]["total"] += 1
            try:
                if mode == "text":
                    result = pipe.predict(text=title, image=None)
                elif mode == "image":
                    with Image.open(img_path) as im:
                        result = pipe.predict(text=None, image=im.convert("RGB"))
                elif mode == "multimodal":
                    with Image.open(img_path) as im:
                        result = pipe.predict(text=title, image=im.convert("RGB"))
                else:
                    raise ValueError(f"Unknown mode {mode}")

                stats[mode]["ok"] += 1
                if result.label == true_label:
                    stats[mode]["correct"] += 1
            except Exception as e:
                stats[mode]["err"] += 1
                if len(stats[mode]["errors"]) < 5:
                    stats[mode]["errors"].append(
                        {"id": row["id"], "error": repr(e), "trace": traceback.format_exc(limit=2)}
                    )

    print("\n======== RESULTS ========")
    all_ok = True
    for mode in modes:
        s = stats[mode]
        scored = s["ok"]
        acc = (s["correct"] / scored) if scored else float("nan")
        print(
            f"{mode:12s}  ran_ok={s['ok']}/{s['total']}  "
            f"errors={s['err']}  accuracy_vs_label={acc:.3f}  "
            f"(correct={s['correct']}/{scored})"
        )
        if s["err"]:
            all_ok = False
            print(f"  sample errors for {mode}:")
            for err in s["errors"]:
                print(f"    id={err['id']}  {err['error']}")

    print("=========================")
    if all_ok:
        print("Pipeline ran without crashes on this sample.")
        print("Accuracy is vs Fakeddit labels on THIS subset — not a full test-set score.")
        return 0

    print("Some modes crashed — fix those errors before trusting the Streamlit demo.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
