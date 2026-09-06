"""Shared preprocessing for text and images (must match training notebooks)."""
from __future__ import annotations

import re
from typing import Optional

from PIL import Image
from torchvision import transforms

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
MAX_LEN = 64
LABEL_NAMES = ("fake", "real")


def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_image(path_or_file) -> Image.Image:
    img = Image.open(path_or_file).convert("RGB")
    return img
