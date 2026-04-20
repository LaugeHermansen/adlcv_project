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
        b, c, _, _ = x.shape
        scale, shift = self.to_scale_shift(text_feat).chunk(2, dim=-1)
        scale = scale.view(b, c, 1, 1)
        shift = shift.view(b, c, 1, 1)
        return x * (1.0 + scale) + shift


class DecodeBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, text_dim: int):
        super().__init__()
        self.conv = ConvBlock(c_in, c_out)
        self.film = FiLM(text_dim=text_dim, feat_dim=c_out)

    def forward(self, x: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.film(x, text_feat)
        return x


class SimplePlacementModel(nn.Module):
    """
    Slightly improved baseline:
    - frozen DINOv2 backbone
    - learned class embedding
    - FiLM conditioning at every decode stage
    - uniform decode path: 32 -> 64 -> 128 -> 256 -> 512
    - single heatmap output in [0, 1]
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
        self.input_film = FiLM(text_dim=text_dim, feat_dim=feat_dim)

        self.dec32 = DecodeBlock(feat_dim, feat_dim, text_dim=text_dim)
        self.dec64 = DecodeBlock(feat_dim, feat_dim // 2, text_dim=text_dim)
        self.dec128 = DecodeBlock(feat_dim // 2, feat_dim // 4, text_dim=text_dim)
        self.dec256 = DecodeBlock(feat_dim // 4, feat_dim // 4, text_dim=text_dim)
        self.dec512 = DecodeBlock(feat_dim // 4, feat_dim // 4, text_dim=text_dim)

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
        h = pixel_values.shape[-2] // self.patch_size
        w = pixel_values.shape[-1] // self.patch_size

        with torch.inference_mode():
            outputs = self.image_backbone(
                pixel_values=pixel_values,
                return_dict=True,
            )
            tokens = outputs.last_hidden_state

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
        text_feat = self.class_embed(class_idx)

        pixel_values = self._prepare_images(image)
        x = self._extract_backbone_map(pixel_values).float()
        x = self.feat_proj(x)
        x = self.input_film(x, text_feat)

        x = self.dec32(x, text_feat)

        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        x = self.dec64(x, text_feat)

        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.dec128(x, text_feat)

        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = self.dec256(x, text_feat)

        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = self.dec512(x, text_feat)

        heatmap = torch.sigmoid(self.head(x).squeeze(1))
        return heatmap
