"""Batch-test the demo pipeline against Fakeddit TSV (+ optional image download).

Runs text-only, image-only, and multimodal on N labeled rows and reports:
  - accuracy vs 2_way_label (0=fake, 1=real)
  - per-sample errors (so intermittent crashes are visible)

Usage (from repo root):

  # Text-only (works with just TSVs + text checkpoint)
  python scripts/batch_eval_from_tsv.py --modes text --n 40

  # Full multimodal — download a small image sample if image_cache is missing
  python scripts/batch_eval_from_tsv.py --n 40 --download

  # If you already have dataset/image_cache from Module 02 / Colab Drive:
  python scripts/batch_eval_from_tsv.py --n 40 --split test

Needs:
  dataset/multimodal_*.tsv
  dataset/checkpoints/...
  For image/multimodal: dataset/image_cache/{id}.jpg  OR pass --download
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference_pipeline import FakeNewsPipeline, default_checkpoints_dir
from app.preprocessing import LABEL_NAMES

USER_AGENT = (
    "Mozilla/5.0 (compatible; MultimodalFakeNewsDetector/0.1; "
    "+https://github.com/manasdutta04/multimodal-fake-news-detector)"
)
REQUEST_TIMEOUT = 12
MIN_SIDE_PX = 32


def cache_path(image_cache: Path, sample_id: str) -> Path:
    return image_cache / f"{str(sample_id).replace('/', '_')}.jpg"


def is_valid_image_file(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            return w >= MIN_SIDE_PX and h >= MIN_SIDE_PX
    except Exception:
        return False


def download_one(sample_id: str, url: str, image_cache: Path) -> tuple[str, bool, str]:
    path = cache_path(image_cache, sample_id)
    if path.exists() and is_valid_image_file(path):
        return sample_id, True, "cached"
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        resp.raise_for_status()
        data = resp.content
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
                return sample_id, False, "too_small"
            image_cache.mkdir(parents=True, exist_ok=True)
            im.save(path, format="JPEG", quality=90)
        if not is_valid_image_file(path):
            path.unlink(missing_ok=True)
            return sample_id, False, "invalid_after_save"
        return sample_id, True, "downloaded"
    except Exception as e:
        if path.exists():
            path.unlink(missing_ok=True)
        return sample_id, False, repr(e)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    parts = []
    per = max(1, n // 2)
    for _, g in df.groupby("2_way_label"):
        parts.append(g.sample(n=min(len(g), per), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_text_rows(tsv: Path, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    df["id"] = df["id"].astype(str)
    df = df[df["clean_title"].notna() & (df["clean_title"].astype(str).str.strip() != "")]
    df = df[df["2_way_label"].isin([0, 1])].copy()
    if len(df) == 0:
        raise SystemExit(f"No usable text rows in {tsv}")
    out = stratified_sample(df, n=n, seed=seed)
    out["image_path"] = None
    return out


def load_paired(
    tsv: Path,
    image_cache: Path,
    n: int,
    seed: int,
    download: bool,
    download_workers: int,
    download_pool: int,
) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    df["id"] = df["id"].astype(str)
    df = df[df["clean_title"].notna() & (df["clean_title"].astype(str).str.strip() != "")]
    df = df[df["2_way_label"].isin([0, 1])].copy()
    df["image_path"] = df["id"].map(lambda i: str(cache_path(image_cache, i)))

    cached = df[df["image_path"].map(lambda p: Path(p).exists())].copy()
    if len(cached) >= n:
        return stratified_sample(cached, n=n, seed=seed)

    if not download:
        raise SystemExit(
            f"image_cache has {len(cached)} usable files under {image_cache} "
            f"(need ~{n}).\n\n"
            "Fix one of these:\n"
            "  1) Copy image_cache from Colab/Drive (Module 02 output) into dataset/image_cache\n"
            "  2) Download a small sample now:\n"
            "       python scripts/batch_eval_from_tsv.py --n 40 --download\n"
            "  3) Text-only smoke test (no images):\n"
            "       python scripts/batch_eval_from_tsv.py --modes text --n 40\n"
        )

    # Prefer rows with http(s) image_url; try more than n because many URLs fail
    urls = df.copy()
    if "image_url" not in urls.columns:
        raise SystemExit("TSV has no image_url column — cannot --download")
    urls["image_url"] = urls["image_url"].astype(str).str.strip()
    urls = urls[
        urls["image_url"].str.startswith(("http://", "https://"))
        & (urls["image_url"].str.lower() != "nan")
    ]
    # already-cached first, then candidates to download
    need = max(n * 4, download_pool)  # oversample; many Reddit/Imgur links die
    candidates = stratified_sample(urls, n=min(len(urls), need), seed=seed)

    to_fetch = [
        (row["id"], row["image_url"])
        for _, row in candidates.iterrows()
        if not Path(row["image_path"]).exists()
    ]
    print(f"Downloading up to {len(to_fetch)} images into {image_cache} …")
    image_cache.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=download_workers) as ex:
        futs = [ex.submit(download_one, sid, url, image_cache) for sid, url in to_fetch]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="download"):
            _, success, _ = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
    print(f"download done: ok={ok} fail={fail}")

    df["image_path"] = df["id"].map(lambda i: str(cache_path(image_cache, i)))
    cached = df[df["image_path"].map(lambda p: Path(p).exists() and is_valid_image_file(Path(p)))].copy()
    if len(cached) == 0:
        raise SystemExit(
            "Download finished but no valid images were saved.\n"
            "Many Fakeddit image_url links are dead. Copy image_cache from the "
            "Colab/Drive machine that ran Module 02, or use --modes text."
        )
    if len(cached) < n:
        print(f"Warning: only {len(cached)} images available; evaluating that many.")
        n = len(cached)
    return stratified_sample(cached, n=n, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset", help="Folder with TSVs + image_cache")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Checkpoints dir (default: CHECKPOINTS_DIR or dataset/checkpoints)",
    )
    parser.add_argument("--split", choices=["test", "validate", "train"], default="test")
    parser.add_argument("--n", type=int, default=40, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modes",
        default="text,image,multimodal",
        help="Comma list: text,image,multimodal",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="If image_cache is missing/sparse, download images from TSV image_url",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--download-pool",
        type=int,
        default=200,
        help="How many URL candidates to try when --download (many fail)",
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

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    needs_images = any(m in ("image", "multimodal") for m in modes)

    if needs_images:
        df = load_paired(
            tsv,
            image_cache,
            n=args.n,
            seed=args.seed,
            download=args.download,
            download_workers=args.download_workers,
            download_pool=args.download_pool,
        )
        print(
            f"Evaluating {len(df)} paired rows  "
            f"(fake={(df['2_way_label']==0).sum()}, real={(df['2_way_label']==1).sum()})"
        )
    else:
        df = load_text_rows(tsv, n=args.n, seed=args.seed)
        print(
            f"Evaluating {len(df)} text rows  "
            f"(fake={(df['2_way_label']==0).sum()}, real={(df['2_way_label']==1).sum()})"
        )

    print("Loading models…")
    pipe = FakeNewsPipeline(ckpt)

    stats = {
        m: {"ok": 0, "err": 0, "correct": 0, "total": 0, "errors": []}
        for m in modes
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="batch_eval"):
        title = str(row["clean_title"])
        y = int(row["2_way_label"])
        true_label = LABEL_NAMES[y]
        img_path = row.get("image_path")

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
