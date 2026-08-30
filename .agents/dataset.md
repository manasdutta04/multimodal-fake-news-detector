# Dataset

## Location

- Local path: `dataset/` (gitignored — do not commit TSVs or downloaded images)
- Files:
  - `multimodal_train.tsv` (~564k rows)
  - `multimodal_validate.tsv` (~59k rows)
  - `multimodal_test_public.tsv` (~59k rows)

This is the **Fakeddit multimodal-only** subset (`hasImage=True` on all rows). Images are **URLs**, not local files, until downloaded separately.

## Key columns

| Column | Role |
|--------|------|
| `clean_title` | Primary text feature for modeling |
| `title` | Original title (reference) |
| `image_url` | Image download / load source |
| `hasImage` | Always true in this subset |
| `2_way_label` | Binary: start here (0 = fake, 1 = real) |
| `3_way_label` | Coarser multi-class (stretch) |
| `6_way_label` | Fine-grained multi-class (stretch) |
| `id`, `subreddit`, `domain`, … | Metadata / EDA |

Confirm label semantics against Fakeddit docs if anything looks off; treat 2-way as the v1 target.

## Known stats (local copy)

- 2-way train skew: ~61% fake (0) / ~39% real (1) — report per-class Precision/Recall/F1, not accuracy alone
- ~1.5k train / ~170 val / ~156 test rows have **null `image_url`** despite `hasImage=True` — drop before image/multimodal training
- `clean_title` is never empty in the local splits; median length ~34 chars

## Agent rules for data work

1. Use the provided train / validate / test splits — do not reshuffle them together into a new random split
2. Fit preprocessing (TF-IDF, tokenizers, image stats) on **train only**; apply to val/test
3. Stratify only if creating internal subsets from train; never leak test into training
4. Filter broken/missing/corrupt images **before** the training loop
5. Check dataset license before publishing models or demos that redistribute data
6. For early iteration, a stratified train subsample is fine — document the subsample size in experiment logs

## Suggested EDA checklist

- Class counts for 2/3/6-way
- Text length distribution
- Fraction of missing/broken `image_url`
- Top subreddits (domain shift risk: heavy `psbattle_artwork`, etc.)
