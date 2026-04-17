from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ResNetVisionBackbone(nn.Module):
    def __init__(
        self,
        token_dim: int = 512,
        backbone: str = "resnet50",
        pretrained: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.freeze = freeze
        self.encoder, in_dim = self._build_encoder(backbone, pretrained)
        self.project = nn.Linear(in_dim, token_dim)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = _normalize_rgb(rgb)
        features = self.encoder(rgb)
        if features.ndim == 2:
            global_raw = features
            patch_raw = features.unsqueeze(1)
        else:
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            global_raw = pooled
            patch_raw = features.flatten(2).transpose(1, 2)
        global_token = self.project(global_raw)
        pooled_patch_token = self.project(patch_raw.mean(dim=1))
        return global_token, pooled_patch_token

    def _build_encoder(self, backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
        try:
            import torchvision.models as tv_models

            if backbone == "resnet18":
                model = tv_models.resnet18(weights=None)
                in_dim = model.fc.in_features
            else:
                model = tv_models.resnet50(weights=None)
                in_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_dim
        except Exception:
            return _SmallCNN(), 256


class _SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.shape[-1] == 3:
        rgb = rgb.permute(0, 3, 1, 2)
    rgb = rgb.float()
    if rgb.max() > 2:
        rgb = rgb / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std

