# Project Requirements Document
## Multimodal Fake News & Misinformation Detector with Explainability

---

## 1. Project Overview

**Problem statement:** News and social media posts spread misinformation through a combination of text, images, and sometimes video/audio. A model that only looks at text (or only at images) misses cases where the mismatch *between* modalities is the actual signal — e.g. a real photo captioned with a false claim, or a fabricated image paired with real text.

**Goal:** Build a system that takes a news item (text + optional image) and outputs:
1. A prediction — real / fake (or a fake-probability score)
2. A human-readable explanation of *why* the model reached that decision

**Why explainability matters here specifically:** a black-box "fake" label is not actionable — a user, journalist, or moderator needs to know *which words/phrases* and *which image regions* drove the decision, and whether text and image agree or contradict each other.

---

## 2. Objectives

- O1: Classify news content as real/fake using text alone, image alone, and combined (multimodal)
- O2: Fuse text and image signals to catch cross-modal inconsistency (text-image mismatch)
- O3: Produce explanations at three levels: word-level (text), region-level (image), and modality-agreement level (does text support the image, or contradict it?)
- O4: Evaluate not just accuracy but *faithfulness* of explanations
- O5: Ship a working demo (Streamlit/FastAPI) where a user pastes text + uploads an image and gets prediction + visual explanation

---

## 3. Scope

### In scope
- Binary classification (real/fake); optionally extend to multi-class (satire, misleading, fabricated, real)
- English-language text
- Static images (not video)
- Explainability for both modalities + fusion layer

### Out of scope (v1)
- Video/audio misinformation
- Multilingual support (can be a stretch goal)
- Real-time web crawling / fact-checking against live databases (can be simulated with a static claim-verification dataset instead)
- Deepfake-specific pixel forensics (different, deeper CV problem — mention as future work, don't attempt in v1)

---

## 4. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | System accepts text input, image input, or both |
| FR2 | System outputs a fake/real label with confidence score |
| FR3 | System highlights influential words/phrases in the text (e.g. via attention or SHAP) |
| FR4 | System highlights influential regions in the image (e.g. via Grad-CAM) |
| FR5 | System reports a text-image consistency score (do the two modalities agree?) |
| FR6 | System handles missing modality gracefully (text-only or image-only input) |
| FR7 | System logs predictions + explanations for later review/audit |
| FR8 | UI displays original input, prediction, confidence, and explanation overlays side by side |

---

## 5. Non-Functional Requirements

- **Interpretability:** every prediction must ship with an explanation — no silent black-box output
- **Latency:** single-item inference should return in under ~5 seconds on CPU (reasonable for a demo, not production SLA)
- **Reproducibility:** fixed seeds, versioned datasets, logged hyperparameters
- **Modularity:** text encoder, image encoder, fusion module, and explainability module should be swappable independently
- **Honesty about limitations:** the demo should clearly state it's a research prototype, not a deployed fact-checker

---

## 6. System Architecture

```
                ┌───────────────┐
   Text ──────▶ │ Text Encoder  │──┐
                │ (NLP branch)  │  │
                └───────────────┘  │
                                   ▼
                            ┌──────────────┐       ┌────────────────┐
                            │ Fusion Layer │──────▶│ Classifier Head │──▶ Real/Fake
                            └──────────────┘       └────────────────┘
                                   ▲
                ┌───────────────┐  │
  Image ──────▶ │ Image Encoder │──┘
                │ (DL/CV branch)│
                └───────────────┘

Parallel explainability branch:
  Text Encoder  ──▶ SHAP / LIME / Attention weights ──▶ word-level heatmap
  Image Encoder ──▶ Grad-CAM / Integrated Gradients ──▶ region-level heatmap
  Fusion Layer  ──▶ cross-modal attention weights ──▶ agreement/mismatch score
```

### 6.1 Text branch (NLP)
- Preprocessing: cleaning, tokenization, stopword handling (careful — don't over-clean, style cues like exaggeration/emotional language are signal for fake news)
- Feature options (pick one to start, compare later):
  - Classical: TF-IDF + Logistic Regression / SVM (baseline)
  - Deep: BiLSTM with attention
  - Transformer: fine-tuned BERT/RoBERTa (best accuracy, also gives attention weights for explainability almost for free)

### 6.2 Image branch (DL/CV)
- Pretrained CNN backbone (ResNet50 / EfficientNet) fine-tuned, or a Vision Transformer (ViT)
- Used to detect visual manipulation cues, style inconsistency, or simply to produce an embedding for fusion

### 6.3 Fusion strategies (pick one, document why)
- **Early fusion:** concatenate text + image embeddings before classification (simplest, good baseline)
- **Late fusion:** train separate classifiers, combine their outputs (easy to explain "text says X, image says Y")
- **Cross-attention fusion:** text and image embeddings attend to each other (more powerful, and cross-attention weights directly double as an explainability signal for modality agreement)

Recommendation: start with late fusion for a working v1, then upgrade to cross-attention once the pipeline works end to end — that upgrade path also gives you a natural "ablation study" section for your report/paper.

---

## 7. Explainability Requirements

| Modality | Technique | Output |
|----------|-----------|--------|
| Text | SHAP or LIME on classifier, or native attention weights if using a Transformer | Word/phrase importance heatmap |
| Image | Grad-CAM / Grad-CAM++ | Class-activation heatmap over image regions |
| Fusion | Cross-modal attention weights, or a simple rule-based agreement score (embedding cosine similarity between text and image) | Agreement/contradiction score + short natural-language summary |

**Faithfulness check (important, often skipped):** don't just generate pretty heatmaps — verify they're meaningful. E.g. mask the top-highlighted words/regions and confirm the model's confidence actually drops. This is a common ask in reviews/interviews for explainability projects.

---

## 8. Data Requirements

- **Text-only fake news datasets:** LIAR, FakeNewsNet, ISOT Fake News Dataset
- **Multimodal datasets:** Fakeddit (Reddit posts with text+image+label, large and well-suited to this exact project), MediaEval / weibo-based multimodal misinformation datasets, or Twitter multimodal misinformation datasets
- **Recommendation:** start with Fakeddit — it's built specifically for multimodal fake news detection, has millions of samples, and includes 2-way/3-way/6-way label granularity so you can start binary and extend later
- Note: check dataset licenses before any public sharing of the trained model/demo

---

## 9. Evaluation Metrics

- Classification: Accuracy, Precision, Recall, F1 (report per-class, not just overall — fake news datasets are often imbalanced)
- Confusion matrix for error analysis
- ROC-AUC for threshold tuning
- Explainability-specific: qualitative review of heatmaps on a held-out sample set + the faithfulness/masking check from Section 7
- Ablation: text-only vs image-only vs multimodal performance comparison (this table is usually the most convincing part of the writeup)

---

## 10. Tech Stack (suggested)

- **Language:** Python
- **NLP:** HuggingFace Transformers (BERT/RoBERTa), NLTK/spaCy for preprocessing
- **CV:** torchvision (pretrained ResNet/ViT)
- **Explainability:** `shap`, `lime`, `captum` (for Grad-CAM/Integrated Gradients), `pytorch-grad-cam`
- **Training:** PyTorch
- **Serving/demo:** FastAPI backend + Streamlit frontend (or a single Streamlit app for a faster demo)
- **Experiment tracking:** simple CSV/JSON logs to start; MLflow or Weights & Biases if you want it to look more production-grade for interviews

---

## 11. Suggested Milestones

1. **Week 1–2:** Data collection + EDA (class balance, text length distribution, image quality check)
2. **Week 3:** Text-only baseline (TF-IDF+LogReg, then BERT fine-tune) + SHAP/attention explainability
3. **Week 4:** Image-only baseline (CNN fine-tune) + Grad-CAM explainability
4. **Week 5:** Fusion model (start late fusion, then try cross-attention) + agreement-score explainability
5. **Week 6:** Evaluation, ablation study, faithfulness checks
6. **Week 7:** Streamlit/FastAPI demo + writeup/report

---

## 12. Deliverables

- Trained models (text, image, multimodal) with saved weights
- Evaluation report with metrics table + ablation study
- Explainability module producing heatmaps + agreement scores
- Working demo app (Streamlit/FastAPI)
- Project report / README documenting architecture, dataset, results, and limitations

---

## 13. Risks / Open Questions to Resolve Early

- Dataset imbalance (most datasets skew toward one class) — decide on class-weighting or resampling strategy upfront
- Compute budget — fine-tuning BERT + a CNN together needs a GPU; plan for Colab/Kaggle GPU quotas or a smaller distilled model (DistilBERT) if compute-constrained
- Explainability output volume — decide early how many samples get manual heatmap review vs. automated faithfulness scoring, since manual review doesn't scale
