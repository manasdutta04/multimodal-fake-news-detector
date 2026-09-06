# Build roadmap

Status: **Modules 01–05 complete** (training notebooks + evaluation + Streamlit demo).

Build in order. Each module depends on the previous.

## Module 01 — Data + text-only baseline — DONE

- Notebook: `notebooks/module01_text_baseline.ipynb`
- Metrics: `artifacts/module01_metrics.csv`
- Checkpoint (Drive): `checkpoints/module01_distilbert/`

**Done when:** held-out accuracy/F1 reported, and top-5 SHAP words printable for ≥1 example.

## Module 02 — Image-only branch — DONE

- Notebook: `notebooks/module02_image_baseline.ipynb`
- Metrics: `artifacts/module02_metrics.csv`
- Checkpoint (Drive): `checkpoints/module02_resnet50/model.pt`

**Done when:** image-only metrics reported + Grad-CAM overlay on ≥1 image.

## Module 03 — Fusion + cross-modal explainability — DONE

- Notebook: `notebooks/module03_fusion.ipynb`
- Metrics: `artifacts/module03_metrics.csv`
- Checkpoint (Drive): `checkpoints/module03_fusion/fusion.pt`

**Done when:** multimodal beats both unimodal baselines; one-line modality-driven explanation.

## Module 04 — Faithfulness + robustness — DONE

- Notebook: `notebooks/module04_evaluation.ipynb`
- Metrics: `artifacts/module04_metrics.csv`
- Summary: `artifacts/RESULTS.md`

**Done when:** deletion-curve plots, missing-modality table, comparison table exist.

## Module 05 — Demo + writeup — DONE

- App: `app/streamlit_app.py` + `app/inference_pipeline.py`
- Writeup: `README.md`
- Helpers: `scripts/check_checkpoints.py`, `scripts/run_demo.ps1`

**Done when:** a stranger can open the app, paste a headline / upload a photo, and get prediction + explanation without hand-holding.

```powershell
# Teammate layout: dataset\checkpoints\
python scripts/check_checkpoints.py
python -m streamlit run app/streamlit_app.py
```

## Priority rule for agents

If asked to “improve the project” without a module specified: prefer demo polish, README clarity, or bugfixes over stretch features (6-way labels, ViT, live deploy, VLM API) unless explicitly requested.
