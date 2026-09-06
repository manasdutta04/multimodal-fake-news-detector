# Coding standards

## Stack (preferred)

- Python 3.10+
- pandas, numpy, scikit-learn
- PyTorch + torchvision
- HuggingFace `transformers`
- `shap`, `pytorch-grad-cam`
- Streamlit for demo; FastAPI optional
- matplotlib for plots/curves

## Repo conventions

- Keep `dataset/` gitignored; never commit raw Fakeddit dumps or bulk images
- Put durable docs under `docs/`; put agent guidance under `.agents/`
- Prefer a clear package layout as code appears, e.g.:

```text
app/                 # Streamlit demo + inference
  streamlit_app.py
  inference_pipeline.py
  models.py
  preprocessing.py
notebooks/           # Colab modules 01–04
artifacts/           # metrics CSVs + RESULTS.md (git-safe)
docs/                # PRD + roadmap
scripts/             # check_checkpoints.py, run_demo.ps1
checkpoints/         # local/Drive weights (gitignored)
dataset/             # Fakeddit TSVs + image_cache (gitignored)
```

- Large weights/checkpoints: gitignore by default; document how to download/reproduce
- Demo loads checkpoints from `CHECKPOINTS_DIR` (env or sidebar) — never hard-code machine paths

## ML hygiene

- Fixed random seeds; log hyperparameters and split names
- Metrics: Accuracy, Precision, Recall, F1 **per class**, confusion matrix; ROC-AUC when useful
- Always compare new models against the current best unimodal/multimodal baseline
- Freeze pretrained backbones first; unfreeze carefully to avoid overfitting
- Match train/inference preprocessing exactly (tokenizer, image size, normalize)

## Code style

- Small, testable functions over notebook-only sprawl for anything that will ship in the demo
- Type hints on public functions (inference, dataset loaders, explainers)
- Fail loudly on missing files/URLs during dataset prep; fail softly in the UI (user-facing message)
- No secrets in repo; API keys for optional VLM comparison via env vars

## What not to build in v1

- Video/audio pipelines
- Live crawlers / live fact-check DBs
- Deepfake forensics
- Multilingual models

Mention those as future work in the README if relevant — do not implement unless explicitly requested.
