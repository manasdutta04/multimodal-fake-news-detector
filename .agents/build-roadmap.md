# Build roadmap

Build in order. Each module depends on the previous. Total ~6–7.5 weeks if following the human roadmap.

## Module 01 — Data + text-only baseline (~1–2 weeks)

- Env: Python 3.10+, venv; Colab/Kaggle GPU when fine-tuning transformers
- EDA on Fakeddit multimodal TSVs
- Baseline: TF-IDF + Logistic Regression
- Upgrade: DistilBERT/BERT fine-tune
- First explainability: SHAP (baseline) and/or attention weights (transformer)

**Done when:** held-out accuracy/F1 reported, and top-5 SHAP “fake-indicating” words printable for ≥1 example.

## Module 02 — Image-only branch (~1–1.5 weeks)

- Download/filter images; correct resize + normalize
- ResNet50 fine-tune (freeze early layers)
- Light augmentation
- Grad-CAM heatmaps

**Done when:** image-only metrics reported + Grad-CAM overlay on ≥1 image.

## Module 03 — Fusion + cross-modal explainability (~1.5–2 weeks)

- Extract embeddings from both branches
- Late fusion MLP first; compare to unimodal
- Optional: cross-attention fusion
- Text–image agreement score + short rule-based NL explanation

**Done when:** multimodal beats both unimodal baselines; one-line modality-driven explanation for sample cases.

## Module 04 — Faithfulness + robustness (~1 week)

- Deletion tests for text (mask top SHAP tokens) and image (blur/blank Grad-CAM region)
- Deletion curves
- Missing-modality ablation (zero/mask image or text embedding)
- Comparison table: text / image / late fusion / (cross-attn) / optional VLM

**Done when:** deletion-curve plot + missing-modality table + combined comparison table exist.

## Module 05 — Demo + writeup (~1 week)

- Checkpoint all branches
- Single inference pipeline function
- Streamlit: text + image upload → label, confidence, text highlights, Grad-CAM, agreement sentence
- Graceful missing input; loading spinner
- README: problem, architecture, data, results, limitations

**Done when:** a stranger can run the demo without hand-holding.

## Priority rule for agents

If asked to “improve the project” without a module specified: finish the earliest incomplete module’s **done criteria** before adding stretch features (6-way labels, ViT, live deploy, etc.).
