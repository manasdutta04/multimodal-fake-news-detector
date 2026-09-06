"""Streamlit demo for Multimodal Fake News Detector with Explainability.

Run from repo root:
  set CHECKPOINTS_DIR=path/to/checkpoints
  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from PIL import Image

from app.inference_pipeline import FakeNewsPipeline, default_checkpoints_dir


st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon=":newspaper:",
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
**Expected layout** (same as Drive)
```
dataset/checkpoints/
  module01_distilbert/
  module02_resnet50/model.pt
  module03_fusion/fusion.pt
```
"""
    )
    if st.button("Clear model cache"):
        st.cache_resource.clear()
        st.success("Cache cleared — next Predict will reload weights.")
    st.markdown(
        "[GitHub](https://github.com/manasdutta04/multimodal-fake-news-detector)"
    )


@st.cache_resource(show_spinner="Loading models…")
def load_pipeline(checkpoints_dir: str) -> FakeNewsPipeline:
    return FakeNewsPipeline(checkpoints_dir)


def render_highlighted_text(text: str, highlights: list) -> str:
    """Bold/color the top occlusion tokens inside the cleaned text."""
    if not text:
        return ""
    scores = {t.lower(): s for t, s in highlights}
    parts = []
    for tok in text.split():
        key = tok.lower().strip(".,!?;:\"'()[]")
        if key in scores and scores[key] > 0:
            parts.append(
                f'<mark style="background:#ff7a5933;padding:0 2px;border-radius:2px">'
                f"{html.escape(tok)}</mark>"
            )
        else:
            parts.append(html.escape(tok))
    return " ".join(parts)


col_in, col_out = st.columns([1, 1])

with col_in:
    st.subheader("Input")
    text = st.text_area(
        "Headline / claim",
        height=120,
        placeholder="Paste a news-style headline or Reddit title…",
    )
    uploaded = st.file_uploader(
        "Image (optional)", type=["jpg", "jpeg", "png", "webp"]
    )
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
            result = pipe.predict(
                text=text or None, image=image, fusion=fusion_choice
            )
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
        st.caption(f"fake  {result.probs['fake']:.1%}")
        st.progress(min(max(float(result.probs["fake"]), 0.0), 1.0))
        st.caption(f"real  {result.probs['real']:.1%}")
        st.progress(min(max(float(result.probs["real"]), 0.0), 1.0))

        if result.unimodal:
            with st.expander("Unimodal branch probabilities"):
                st.json(result.unimodal)

        if text.strip() and result.text_highlights:
            st.write("**Text with influential tokens highlighted**")
            from app.preprocessing import clean_text

            st.markdown(
                render_highlighted_text(clean_text(text), result.text_highlights),
                unsafe_allow_html=True,
            )
            with st.expander("Token occlusion scores"):
                for tok, score in result.text_highlights:
                    st.write(f"- `{tok}` — Δconf={score:+.4f}")

        if result.gradcam_overlay is not None:
            st.write("**Grad-CAM** (regions influencing the vision branch)")
            left, right = st.columns(2)
            if image is not None:
                left.image(image, caption="Original", use_container_width=True)
            right.image(
                result.gradcam_overlay,
                caption="Grad-CAM overlay",
                use_container_width=True,
            )
        elif image is not None:
            st.write("**Original image**")
            st.image(image, use_container_width=True)

st.divider()
st.markdown(
    """
### Limitations
- Prototype trained on **Fakeddit** (Reddit-style posts), not general web news.
- Does **not** verify claims against a live knowledge base.
- Missing-modality mode uses the matching unimodal branch.
- Explanations are model attributions, not proof of truth.

See [artifacts/RESULTS.md](artifacts/RESULTS.md) for metrics from Modules 01–04.
"""
)
