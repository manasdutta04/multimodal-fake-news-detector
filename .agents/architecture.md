# Architecture

```
Text  → Text Encoder (NLP)  ─┐
                             ├→ Fusion → Classifier → Real/Fake + confidence
Image → Image Encoder (CV)  ─┘

Explainability (parallel):
  Text  → SHAP / attention        → word-level highlights
  Image → Grad-CAM                 → region heatmap
  Fusion → agreement / cross-attn  → consistency score + short NL summary
```

## Branches

### Text (NLP)

- Start: TF-IDF + Logistic Regression baseline
- Upgrade: DistilBERT / BERT fine-tune (HuggingFace)
- Do not over-clean text — punctuation, clickbait style, and emotional language can be signal

### Image (CV)

- Start: ResNet50 (torchvision), freeze early layers, fine-tune last block + head
- Optional later: ViT
- Preprocess with the **same** resize + ImageNet mean/std the backbone expects

### Fusion

1. **Late fusion first** — concat embeddings → small MLP (debug-friendly baseline)
2. **Cross-attention later** — stronger, and attention weights double as explainability
3. Multimodal must beat both unimodal baselines; if not, fix fusion before adding features

## Demo serving

- Preferred v1: single Streamlit app
- Optional: FastAPI backend + Streamlit frontend
- Load model weights **once** at app startup, not per prediction
- Inference target for demo: under ~5 seconds on CPU (best-effort)

## Modularity rules

- Separate packages/modules for: data loading, text model, image model, fusion, explainability, inference pipeline, UI
- Checkpoint each branch independently (`state_dict`)
- One inference function: raw text + image path → preprocess → branches → fusion → explanations → display dict
