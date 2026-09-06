# Consolidated results

Values copied from `artifacts/module0{1,2,3,4}_metrics.csv` after Colab runs.

## Classification (test)

| Model | Accuracy | F1 macro | Source |
|-------|----------|----------|--------|
| TF-IDF + LogReg | 0.794 | 0.790 | module01 |
| DistilBERT | 0.850 | 0.844 | module01 |
| ResNet50 | 0.817 | 0.816 | module02 |
| DistilBERT (paired) | 0.847 | 0.847 | module03 |
| ResNet50 (paired) | 0.823 | 0.821 | module03 |
| Late fusion | **0.890** | **0.890** | module03 |
| Cross-attn fusion | **0.893** | **0.893** | module03 |

Multimodal fusion beats both unimodal baselines on the paired test subset.

## Faithfulness (Module 04)

| Protocol | Mean confidence drop | N |
|----------|----------------------|---|
| Text: delete top-5 SHAP tokens (TF-IDF+LogReg) | 0.259 | 80 |
| Image: blank top-20% Grad-CAM area (ResNet50) | 0.176 | 80 |

Higher drop ⇒ more faithful explanation (removing “important” evidence hurts the original prediction).

## Missing-modality robustness (Module 04 eval subset)

| Model | Setting | Accuracy | F1 macro | Δacc vs both |
|-------|---------|----------|----------|--------------|
| Late fusion | text+image | 0.871 | 0.870 | — |
| Late fusion | text-only (image=0) | 0.849 | 0.849 | −0.023 |
| Late fusion | image-only (text=0) | 0.768 | 0.759 | −0.103 |
| Cross-attn fusion | text+image | 0.887 | 0.887 | — |
| Cross-attn fusion | text-only (image=0) | 0.859 | 0.858 | −0.029 |
| Cross-attn fusion | image-only (text=0) | 0.746 | 0.745 | −0.141 |

Text carries more signal when one modality is missing.

## Artifacts in this folder

| File | Contents |
|------|----------|
| `module01_metrics.csv` | TF-IDF + DistilBERT val/test |
| `module02_metrics.csv` | ResNet50 val/test |
| `module03_metrics.csv` | Paired unimodal + fusion |
| `module04_metrics.csv` | Faithfulness + missing-modality rows |

Large weights (`*.pt`, DistilBERT folders, `image_cache/`) stay on Drive — not in git.
