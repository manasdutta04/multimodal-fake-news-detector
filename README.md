# Multimodal Fake News Detector with Explainability

[![GitHub](https://img.shields.io/badge/GitHub-manasdutta04%2Fmultimodal--fake--news--detector-181717?logo=github&logoColor=white)](https://github.com/manasdutta04/multimodal-fake-news-detector)

Research prototype that classifies news-style **text + image** posts as real/fake and explains *why* — at word level, image-region level, and text–image agreement.

> **Not a deployed fact-checker.** Predictions are model scores on Fakeddit-style Reddit posts. The system does not verify claims against a live knowledge base.

## Architecture

```
Text  → DistilBERT  ─┐
                     ├→ Late / Cross-attn fusion → Real/Fake + confidence
Image → ResNet50    ─┘

Explainability:
  Text   → token occlusion / SHAP
  Image  → Grad-CAM
  Fusion → cosine agreement + rule-based modality summary
```

## Results (from training notebooks)

### Unimodal + fusion (test)

| Model | Accuracy | F1 macro |
|-------|----------|----------|
| TF-IDF + LogReg | 0.794 | 0.790 |
| DistilBERT | 0.850 | 0.844 |
| ResNet50 | 0.817 | 0.816 |
| DistilBERT (paired subset) | 0.847 | 0.847 |
| ResNet50 (paired subset) | 0.823 | 0.821 |
| **Late fusion** | **0.890** | **0.890** |
| **Cross-attn fusion** | **0.893** | **0.893** |

### Faithfulness & robustness (Module 04)

| Check | Result |
|-------|--------|
| Text SHAP deletion (mean conf drop @ top-5 tokens) | **0.259** |
| Image Grad-CAM deletion (mean conf drop @ top-20% area) | **0.176** |
| Late fusion text+image acc | 0.871 |
| Late fusion text-only (image=0) Δacc | −0.023 |
| Late fusion image-only (text=0) Δacc | −0.103 |
| Cross-attn text+image acc | 0.887 |
| Cross-attn text-only Δacc | −0.029 |
| Cross-attn image-only Δacc | −0.141 |

Fusion beats both unimodal baselines. Explanations move model confidence when important evidence is removed. Text carries more signal under missing-modality stress.

## Repository layout

```
app/                 # Streamlit demo + inference pipeline
artifacts/           # Small metrics CSVs (safe for git)
notebooks/           # Colab modules 01–04
docs/                # PRD + roadmap
dataset/             # gitignored — Fakeddit TSVs + Drive caches
checkpoints/         # gitignored locally / kept on Drive
```

## Demo (Module 05)

### 1. Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Point at checkpoints

Weights live under **`dataset/checkpoints/`** (same as Google Drive / teammate PC):

```text
dataset/
  multimodal_train.tsv
  multimodal_validate.tsv
  multimodal_test_public.tsv
  image_cache/                    # optional for demo
  checkpoints/
    module01_distilbert/
    module02_resnet50/model.pt
    module03_fusion/fusion.pt
```

Validate from repo root:

```powershell
python scripts/check_checkpoints.py
# or explicitly:
python scripts/check_checkpoints.py --dir dataset\checkpoints
```

### 3. Run the app

```powershell
# If dataset\checkpoints exists, this finds it automatically:
.\scripts\run_demo.ps1

# Or:
$env:CHECKPOINTS_DIR="dataset\checkpoints"
python -m streamlit run app/streamlit_app.py
```

Sidebar also defaults to `dataset/checkpoints` when that folder exists.

### 4. Use the app

- Paste a headline and/or upload an image → **Predict**
- Text-only / image-only inputs are supported
- Output: label, confidence, agreement (if both), explanation, token highlights, Grad-CAM

### 5. Batch-test from the TSV (recommended)

On the machine that has `dataset/` (TSVs + `image_cache/` + checkpoints):

```powershell
python scripts/batch_eval_from_tsv.py --n 40 --split test
```

This runs **text / image / multimodal** on paired Fakeddit rows and prints accuracy + any crashes. Use this instead of random clickbait headlines for “does it work?” checks.

See [artifacts/RESULTS.md](artifacts/RESULTS.md) for the full metrics tables.

## Training notebooks (Colab)

| Notebook | Purpose |
|----------|---------|
| `notebooks/module01_text_baseline.ipynb` | Data, TF-IDF, DistilBERT, SHAP |
| `notebooks/module02_image_baseline.ipynb` | Image download, ResNet50, Grad-CAM |
| `notebooks/module03_fusion.ipynb` | Late + cross-attn fusion, agreement |
| `notebooks/module04_evaluation.ipynb` | Faithfulness deletion + missing modality |

Dataset: **Fakeddit** multimodal-only TSVs (`2_way_label`). Place under Drive `MyDrive/dataset/` for Colab.

## Limitations

- English, static images only (no video/audio)
- Domain skew toward Reddit / photoshop-battle style content
- Agreement score is consistency between modalities, not ground-truth verification
- Hosted-GPU runs often use stratified subsamples; full-corpus numbers need `SUBSAMPLE_*=None`

## License

MIT — see [LICENSE](LICENSE).
