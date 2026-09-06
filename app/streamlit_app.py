"""Streamlit demo for Multimodal Fake News Detector with Explainability.

Run from repo root:
  set CHECKPOINTS_DIR=path/to/checkpoints
  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when launched via `streamlit run app/...`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from PIL import Image

from app.inference_pipeline import FakeNewsPipeline, default_checkpoints_dir


st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="📰",
    layout="wide",
)

st.title("Multimodal Fake News Detector")
st.caption(
    "Research prototype — not a deployed fact-checker. "
    "Trained on Fakeddit-style Reddit posts (text + image)."
)

with st.sidebar:
    st.header("Setup")
    default_dir = str(default_checkpoints_dir())
    ckpt_dir = st.text_input(
        "Checkpoints directory",
        value=os.environ.get("CHECKPOINTS_DIR", default_dir),
        help="Folder containing module01_distilbert/, module02_resnet50/, module03_fusion/",
    )
    fusion_choice = st.selectbox(
        "Fusion model",
        options=["cross_attn", "late"],
        format_func=lambda x: "Cross-attention (best)" if x == "cross_attn" else "Late fusion",
    )
    st.markdown(
        """
**Expected layout**
```
checkpoints/
  module01_distilbert/
  module02_resnet50/model.pt
  module03_fusion/fusion.pt
```
"""
    )
    st.markdown(
        "[GitHub](https://github.com/manasdutta04/multimodal-fake-news-detector)"
    )


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline(checkpoints_dir: str) -> FakeNewsPipeline:
    return FakeNewsPipeline(checkpoints_dir)


col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("Input")
    text = st.text_area(
        "Headline / claim",
        height=120,
        placeholder="Paste a news-style headline or Reddit title…",
    )
    uploaded = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png", "webp"])
    run = st.button("Predict", type="primary", use_container_width=True)

with col_out:
    st.subheader("Result")
    result_box = st.empty()

if run:
    if not text.strip() and uploaded is None:
        st.error("Provide text, an image, or both.")
        st.stop()

    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        st.error(f"Checkpoints directory not found: `{ckpt_path}`")
        st.stop()

    try:
        pipe = load_pipeline(str(ckpt_path))
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

    image = Image.open(uploaded).convert("RGB") if uploaded is not None else None

    with st.spinner("Running inference…"):
        try:
            result = pipe.predict(text=text or None, image=image, fusion=fusion_choice)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    with result_box.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", result.label.upper())
        c2.metric("Confidence", f"{result.confidence:.1%}")
        c3.metric("Mode", result.mode)
        if result.agreement is not None:
            st.metric("Text–image agreement", f"{result.agreement:.3f}")

        st.info(result.explanation)

        st.write("**Class probabilities**")
        st.json(result.probs)

        if result.unimodal:
            st.write("**Unimodal branch probs**")
            st.json(result.unimodal)

        if result.text_highlights:
            st.write("**Influential tokens** (occlusion drop on DistilBERT)")
            for tok, score in result.text_highlights:
                st.write(f"- `{tok}` — Δconf={score:+.4f}")

        if result.gradcam_overlay is not None:
            st.write("**Grad-CAM** (image regions influencing the vision branch)")
            st.image(result.gradcam_overlay, caption="Grad-CAM overlay", use_container_width=True)

        if image is not None:
            st.write("**Original image**")
            st.image(image, use_container_width=True)

st.divider()
st.markdown(
    """
### Limitations
- Prototype trained on **Fakeddit** (Reddit-style posts), not general web news.
- Does **not** verify claims against a live knowledge base.
- Missing-modality mode uses the matching unimodal branch (or zeroed fusion embeddings in training eval).
- Explanations are model attributions, not proof of truth.
"""
)
