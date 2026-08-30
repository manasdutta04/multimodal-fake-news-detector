# Project context

## Problem

Misinformation often lives in the *combination* of text and image — e.g. a real photo with a false caption. Text-only or image-only models miss that cross-modal signal.

## Goal

Given a news-like item (text + optional image), the system must output:

1. **Prediction** — real / fake (or fake-probability)
2. **Explanation** — why: influential words, influential image regions, and whether text and image agree or conflict

A bare "fake" label without explanation is out of product intent.

## Objectives (O1–O5)

- Classify with text-only, image-only, and multimodal paths
- Fuse modalities to catch text–image inconsistency
- Explain at word, region, and modality-agreement levels
- Evaluate explanation **faithfulness**, not only accuracy
- Ship a demo (Streamlit / FastAPI) with prediction + visual explanation

## Scope (v1)

**In:** English text, static images, binary real/fake (optional later: 3-way / 6-way), explainability on both modalities + fusion.

**Out:** video/audio, multilingual, live web fact-checking, deepfake pixel forensics.

## Non-negotiables

- Every prediction ships with an explanation (no silent black-box output)
- Missing modality must work (text-only or image-only) without crashing
- Demo must state it is a research prototype, not a deployed fact-checker
- Prefer modularity: text encoder, image encoder, fusion, explainability should be swappable

## Success signal

A stranger can paste a headline, upload a photo, and get label + confidence + heatmaps + a plain-English agreement line — without the author explaining the UI.
