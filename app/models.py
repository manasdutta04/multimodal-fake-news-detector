"""Model constructors matching Modules 01–03 checkpoints."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


class LateFusionMLP(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, text_emb, image_emb):
        return self.net(torch.cat([text_emb, image_emb], dim=-1))


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, d_model)
        self.image_proj = nn.Linear(image_dim, d_model)
        self.text_to_image = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.image_to_text = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb).unsqueeze(1)
        i = self.image_proj(image_emb).unsqueeze(1)
        t2i, _ = self.text_to_image(t, i, i, need_weights=False)
        i2t, _ = self.image_to_text(i, t, t, need_weights=False)
        fused = torch.cat(
            [t.squeeze(1), i.squeeze(1), t2i.squeeze(1), i2t.squeeze(1)], dim=-1
        )
        return self.classifier(fused)


def build_resnet50(num_classes: int = 2) -> nn.Module:
    try:
        model = resnet50(weights=None)
    except TypeError:
        model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
