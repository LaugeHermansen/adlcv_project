from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from src.hidden_objects_dataset import HiddenObjectsHeatmap


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLM(nn.Module):
    def __init__(self, text_dim: int, feat_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Sequential(
            nn.Linear(text_dim, 2 * feat_dim),
            nn.GELU(),
            nn.Linear(2 * feat_dim, 2 * feat_dim),
        )

    def forward(self, x: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        text_feat: (B, D)
        """
        b, c, _, _ = x.shape
        scale, shift = self.to_scale_shift(text_feat).chunk(2, dim=-1)
        scale = scale.view(b, c, 1, 1)
        shift = shift.view(b, c, 1, 1)
        return x * (1.0 + scale) + shift


class SimplePlacementModel(nn.Module):
    """
    Simple first baseline:
    - frozen DINOv2-small backbone
    - learned class embedding from dataset class names
    - FiLM conditioning
    - lightweight conv decoder
    - single heatmap output

    Input:
        image: (B, 3, 512, 512), float in [0, 1]
        text: list[str] length B

    Output:
        heatmap: (B, 512, 512)
    """

    def __init__(
        self,
        image_backbone_name: str = "facebook/dinov2-small",
        backbone_image_size: int = 448,
        text_dim: int = 128,
        feat_dim: int = 128,
    ):
        super().__init__()

        classes = sorted(np.unique(HiddenObjectsHeatmap("test").anno_dataset.class_arr))
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = self.classes
        num_classes = len(self.classes)

        self.image_backbone = AutoModel.from_pretrained(image_backbone_name)
        self.image_processor = AutoImageProcessor.from_pretrained(image_backbone_name)
        freeze_module(self.image_backbone)

        self.backbone_image_size = backbone_image_size
        self.patch_size = int(self.image_backbone.config.patch_size)
        backbone_dim = int(self.image_backbone.config.hidden_size)

        self.class_embed = nn.Embedding(num_classes, text_dim)

        self.feat_proj = nn.Conv2d(backbone_dim, feat_dim, kernel_size=1)
        self.film = FiLM(text_dim=text_dim, feat_dim=feat_dim)

        self.dec1 = ConvBlock(feat_dim, feat_dim)
        self.dec2 = ConvBlock(feat_dim, feat_dim // 2)
        self.dec3 = ConvBlock(feat_dim // 2, feat_dim // 4)
        self.dec4 = ConvBlock(feat_dim // 4, feat_dim // 4)

        self.head = nn.Conv2d(feat_dim // 4, 1, kernel_size=1)

    def _prepare_images(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, 3, H, W), got {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[1]}")
        if image.shape[-2:] != (512, 512):
            raise ValueError(f"Expected image shape (B, 3, 512, 512), got {tuple(image.shape)}")

        x = F.interpolate(
            image,
            size=(self.backbone_image_size, self.backbone_image_size),
            mode="bilinear",
            align_corners=False,
        )

        mean = torch.tensor(
            self.image_processor.image_mean,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.image_processor.image_std,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)

        x = (x - mean) / std
        return x

    def _extract_backbone_map(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Returns a single spatial feature map from the final DINO layer.
        """
        h = pixel_values.shape[-2] // self.patch_size
        w = pixel_values.shape[-1] // self.patch_size

        with torch.inference_mode():
            outputs = self.image_backbone(
                pixel_values=pixel_values,
                return_dict=True,
            )
            tokens = outputs.last_hidden_state  # (B, 1 + HW, C) or (B, HW, C)

        if tokens.shape[1] == h * w + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != h * w:
            raise ValueError(
                f"Unexpected token count: got {tokens.shape[1]}, expected {h*w} or {h*w+1}"
            )

        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)
        return feat

    def _text_to_indices(self, text: Sequence[str], device: torch.device) -> torch.Tensor:
        if not isinstance(text, (list, tuple)):
            raise TypeError(f"Expected text to be a list/tuple of strings, got {type(text)}")
        if not all(isinstance(t, str) for t in text):
            raise TypeError("Expected every element of text to be a string")

        try:
            idx = [self.class_to_idx[t] for t in text]
        except KeyError as e:
            raise ValueError(f"Unknown class name: {e.args[0]}")

        return torch.tensor(idx, dtype=torch.long, device=device)

    def forward(self, image: torch.Tensor, text: Sequence[str]) -> torch.Tensor:
        b = image.shape[0]
        if len(text) != b:
            raise ValueError(f"Expected {b} text items, got {len(text)}")

        class_idx = self._text_to_indices(text, image.device)

        pixel_values = self._prepare_images(image)
        feat = self._extract_backbone_map(pixel_values).float()   # (B, Cb, Hb, Wb)
        feat = self.feat_proj(feat)                               # (B, Cf, Hb, Wb)

        text_feat = self.class_embed(class_idx)                   # (B, Dt)
        feat = self.film(feat, text_feat)

        x = self.dec1(feat)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec3(x)

        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = self.dec4(x)

        heatmap = torch.sigmoid(self.head(x).squeeze(1))
        return heatmap
