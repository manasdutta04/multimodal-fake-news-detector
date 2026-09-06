"""End-to-end inference for the Streamlit demo.

Loads Module 01–03 checkpoints once, then predicts from text and/or image.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.models import CrossAttentionFusion, LateFusionMLP, build_resnet50
from app.preprocessing import (
    EVAL_TRANSFORM,
    IMG_SIZE,
    LABEL_NAMES,
    MAX_LEN,
    clean_text,
    load_image,
)


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probs: dict
    mode: str  # "text" | "image" | "multimodal"
    agreement: Optional[float] = None
    explanation: str = ""
    text_highlights: list = field(default_factory=list)  # [(token, score), ...]
    gradcam_overlay: Optional[np.ndarray] = None  # RGB uint8
    unimodal: dict = field(default_factory=dict)


class FakeNewsPipeline:
    def __init__(self, checkpoints_dir: str | Path, device: Optional[str] = None):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_ckpt = self.checkpoints_dir / "module01_distilbert"
        self.image_ckpt = self.checkpoints_dir / "module02_resnet50" / "model.pt"
        self.fusion_ckpt = self.checkpoints_dir / "module03_fusion" / "fusion.pt"

        self._validate_paths()
        self._load_models()

    def _validate_paths(self) -> None:
        missing = [
            p
            for p in (self.text_ckpt, self.image_ckpt, self.fusion_ckpt)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoint(s):\n"
                + "\n".join(f"  - {p}" for p in missing)
                + "\nPoint CHECKPOINTS_DIR at `dataset/checkpoints` "
                "(module01_distilbert, module02_resnet50, module03_fusion)."
            )

    def _load_models(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.text_ckpt))
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            str(self.text_ckpt)
        )
        self.text_model.to(self.device).eval()
        for p in self.text_model.parameters():
            p.requires_grad_(False)

        try:
            blob = torch.load(self.image_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(self.image_ckpt, map_location="cpu")

        self.image_clf = build_resnet50(2)
        self.image_clf.load_state_dict(blob["model_state_dict"])
        self.image_clf.to(self.device).eval()

        self.image_encoder = build_resnet50(2)
        self.image_encoder.load_state_dict(blob["model_state_dict"])
        self.image_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = torch.nn.Identity()
        self.image_encoder.to(self.device).eval()

        for p in list(self.image_clf.parameters()) + list(self.image_encoder.parameters()):
            p.requires_grad_(False)

        try:
            fusion = torch.load(self.fusion_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            fusion = torch.load(self.fusion_ckpt, map_location="cpu")

        self.text_dim = int(fusion.get("text_dim", 768))
        self.late = LateFusionMLP(self.text_dim, self.image_dim)
        self.xattn = CrossAttentionFusion(self.text_dim, self.image_dim)
        self.late.load_state_dict(fusion["late_fusion_state_dict"])
        self.xattn.load_state_dict(fusion["cross_attn_state_dict"])
        self.late.to(self.device).eval()
        self.xattn.to(self.device).eval()
        for p in list(self.late.parameters()) + list(self.xattn.parameters()):
            p.requires_grad_(False)

        self._cam = None

    @torch.no_grad()
    def encode_text(self, text: str):
        text = clean_text(text)
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.text_model.distilbert(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
        )
        hidden = out.last_hidden_state[:, 0]
        pooled = F.relu(self.text_model.pre_classifier(hidden))
        logits = self.text_model.classifier(pooled)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        return pooled.squeeze(0), probs, text

    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        x = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.image_encoder(x).squeeze(0)
        logits = self.image_clf(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)
        return feat, probs, x

    def text_token_highlights(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Occlusion ranks for DistilBERT predicted class (demo explainability)."""
        cleaned = clean_text(text)
        tokens = cleaned.split()
        if not tokens:
            return []
        with torch.no_grad():
            _, base_probs, _ = self.encode_text(cleaned)
            pred = int(torch.argmax(base_probs).item())
            base_conf = float(base_probs[pred].item())

        scored: list[tuple[str, float]] = []
        for i, tok in enumerate(tokens):
            masked = " ".join(t if j != i else "[MASK]" for j, t in enumerate(tokens))
            with torch.no_grad():
                _, probs, _ = self.encode_text(masked)
            drop = base_conf - float(probs[pred].item())
            scored.append((tok, drop))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def gradcam_overlay(self, image: Image.Image, pred: int) -> np.ndarray:
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError as e:
            raise ImportError(
                "Install grad-cam: pip install grad-cam"
            ) from e

        # Grad-CAM needs gradients through the CNN
        for p in self.image_clf.parameters():
            p.requires_grad_(True)
        self.image_clf.eval()

        if self._cam is None:
            self._cam = GradCAM(
                model=self.image_clf, target_layers=[self.image_clf.layer4[-1]]
            )

        x = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        x = x.detach().requires_grad_(True)
        grayscale = self._cam(
            input_tensor=x, targets=[ClassifierOutputTarget(int(pred))]
        )[0]

        # Denormalize for overlay
        img = x.detach().cpu().squeeze(0).clone()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        for p in self.image_clf.parameters():
            p.requires_grad_(False)
        return overlay

    @staticmethod
    def agreement_score(text_emb: torch.Tensor, image_emb: torch.Tensor, xattn: CrossAttentionFusion) -> float:
        with torch.no_grad():
            t = xattn.text_proj(text_emb.unsqueeze(0))
            i = xattn.image_proj(image_emb.unsqueeze(0))
            return float(F.cosine_similarity(t, i, dim=-1).item())

    @staticmethod
    def explain(
        fusion_pred: int,
        fusion_prob: np.ndarray,
        text_prob: Optional[np.ndarray],
        image_prob: Optional[np.ndarray],
        agreement: Optional[float],
        mode: str,
    ) -> str:
        label = LABEL_NAMES[fusion_pred]
        conf = float(fusion_prob[fusion_pred])

        if mode == "text":
            return (
                f"Predicted {label} (conf={conf:.2f}) from text only — "
                "no image was provided, so the DistilBERT branch was used."
            )
        if mode == "image":
            return (
                f"Predicted {label} (conf={conf:.2f}) from image only — "
                "no text was provided, so the ResNet50 branch was used."
            )

        assert text_prob is not None and image_prob is not None and agreement is not None
        text_pred = int(np.argmax(text_prob))
        image_pred = int(np.argmax(image_prob))
        text_conf = float(text_prob[text_pred])
        image_conf = float(image_prob[image_pred])
        low = agreement < 0.15

        if text_pred == image_pred == fusion_pred:
            return (
                f"Predicted {label} (conf={conf:.2f}). Text and image agree "
                f"(agreement={agreement:.2f}); both support {LABEL_NAMES[text_pred]}."
            )
        if low and image_conf >= text_conf:
            return (
                f"Predicted {label} (conf={conf:.2f}). Low text–image agreement ({agreement:.2f}); "
                f"decision driven mainly by the image "
                f"(image→{LABEL_NAMES[image_pred]} @ {image_conf:.2f})."
            )
        if low and text_conf > image_conf:
            return (
                f"Predicted {label} (conf={conf:.2f}). Low text–image agreement ({agreement:.2f}); "
                f"decision driven mainly by wording/style "
                f"(text→{LABEL_NAMES[text_pred]} @ {text_conf:.2f})."
            )
        if text_pred != image_pred:
            driver = "image" if image_conf >= text_conf else "text"
            return (
                f"Predicted {label} (conf={conf:.2f}). Modalities disagree "
                f"(text→{LABEL_NAMES[text_pred]}, image→{LABEL_NAMES[image_pred]}, "
                f"agreement={agreement:.2f}); fused decision leans on the {driver} branch."
            )
        return (
            f"Predicted {label} (conf={conf:.2f}) with agreement={agreement:.2f}. "
            f"Text={LABEL_NAMES[text_pred]} ({text_conf:.2f}), "
            f"image={LABEL_NAMES[image_pred]} ({image_conf:.2f})."
        )

    def predict(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        fusion: str = "cross_attn",
    ) -> PredictionResult:
        has_text = bool(text and clean_text(text))
        has_image = image is not None

        if not has_text and not has_image:
            raise ValueError("Provide text, an image, or both.")

        # --- text only ---
        if has_text and not has_image:
            _, probs, cleaned = self.encode_text(text)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
            pnp = probs.detach().cpu().numpy()
            return PredictionResult(
                label=LABEL_NAMES[pred],
                confidence=conf,
                probs={"fake": float(pnp[0]), "real": float(pnp[1])},
                mode="text",
                explanation=self.explain(pred, pnp, pnp, None, None, "text"),
                text_highlights=self.text_token_highlights(cleaned),
                unimodal={"text": {"fake": float(pnp[0]), "real": float(pnp[1])}},
            )

        # --- image only ---
        if has_image and not has_text:
            _, probs, _ = self.encode_image(image)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
            pnp = probs.detach().cpu().numpy()
            overlay = self.gradcam_overlay(image, pred)
            return PredictionResult(
                label=LABEL_NAMES[pred],
                confidence=conf,
                probs={"fake": float(pnp[0]), "real": float(pnp[1])},
                mode="image",
                explanation=self.explain(pred, pnp, None, pnp, None, "image"),
                gradcam_overlay=overlay,
                unimodal={"image": {"fake": float(pnp[0]), "real": float(pnp[1])}},
            )

        # --- multimodal ---
        text_emb, text_probs, cleaned = self.encode_text(text)
        image_emb, image_probs, _ = self.encode_image(image)
        model = self.xattn if fusion == "cross_attn" else self.late
        with torch.no_grad():
            logits = model(text_emb.unsqueeze(0), image_emb.unsqueeze(0))
            fusion_probs = F.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(fusion_probs).item())
        conf = float(fusion_probs[pred].item())
        fp = fusion_probs.detach().cpu().numpy()
        tp = text_probs.detach().cpu().numpy()
        ip = image_probs.detach().cpu().numpy()
        agree = self.agreement_score(text_emb, image_emb, self.xattn)
        overlay = self.gradcam_overlay(image, int(np.argmax(ip)))

        return PredictionResult(
            label=LABEL_NAMES[pred],
            confidence=conf,
            probs={"fake": float(fp[0]), "real": float(fp[1])},
            mode="multimodal",
            agreement=agree,
            explanation=self.explain(pred, fp, tp, ip, agree, "multimodal"),
            text_highlights=self.text_token_highlights(cleaned),
            gradcam_overlay=overlay,
            unimodal={
                "text": {"fake": float(tp[0]), "real": float(tp[1])},
                "image": {"fake": float(ip[0]), "real": float(ip[1])},
            },
        )


def default_checkpoints_dir() -> Path:
    """Resolve checkpoints folder. Prefer `dataset/checkpoints` (Drive / teammate layout)."""
    env = os.environ.get("CHECKPOINTS_DIR")
    if env:
        return Path(env)
    candidates = [
        Path("dataset/checkpoints"),  # Colab Drive + teammate local layout
        Path("checkpoints"),
        Path("/content/drive/MyDrive/dataset/checkpoints"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]
