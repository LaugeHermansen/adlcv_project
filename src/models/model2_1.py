from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation

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


class DinoPatchFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-small",
        image_size: int = 448,
        freeze: bool = True,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)

        if freeze:
            freeze_module(self.backbone)

        self.image_size = image_size
        self.patch_size = int(self.backbone.config.patch_size)
        self.hidden_dim = int(self.backbone.config.hidden_size)
        self.freeze = freeze

    def _prepare_images(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, 3, H, W), got {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[1]}")
        if image.shape[-2:] != (512, 512):
            raise ValueError(f"Expected image shape (B, 3, 512, 512), got {tuple(image.shape)}")

        x = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        mean = torch.tensor(
            self.processor.image_mean,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            self.processor.image_std,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)

        return (x - mean) / std

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_images(image)

        h = pixel_values.shape[-2] // self.patch_size
        w = pixel_values.shape[-1] // self.patch_size

        if self.freeze:
            with torch.inference_mode():
                outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)

        tokens = outputs.last_hidden_state

        if tokens.shape[1] == h * w + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != h * w:
            raise ValueError(
                f"Unexpected token count: got {tokens.shape[1]}, expected {h * w} or {h * w + 1}"
            )

        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)
        return feat.float()


class InternalRawDepthFeatureExtractor(nn.Module):
    """
    RGB image -> frozen monocular depth feature maps.

    No normalization.
    No quantile clipping.
    Keeps raw predicted depth values from the pretrained depth model.

    Input:
        image: (B, 3, 512, 512), expected in [0, 1]

    Output:
        {
            64:  (B, depth_dim, 64, 64),
            128: (B, depth_dim, 128, 128),
            256: (B, depth_dim, 256, 256),
            512: (B, depth_dim, 512, 512),
        }
    """

    def __init__(
        self,
        backbone_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        image_size: int = 518,
        depth_dim: int = 32,
        freeze: bool = True,
        use_log_depth: bool = False,
    ):
        super().__init__()

        self.depth_model = AutoModelForDepthEstimation.from_pretrained(backbone_name)
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)

        if freeze:
            freeze_module(self.depth_model)

        self.image_size = image_size
        self.depth_dim = depth_dim
        self.freeze = freeze
        self.use_log_depth = use_log_depth

        self.enc64 = ConvBlock(1, depth_dim)
        self.enc128 = ConvBlock(1, depth_dim)
        self.enc256 = ConvBlock(1, depth_dim)
        self.enc512 = ConvBlock(1, depth_dim)

    def _prepare_images(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, 3, H, W), got {tuple(image.shape)}")
        if image.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {image.shape[1]}")
        if image.shape[-2:] != (512, 512):
            raise ValueError(f"Expected image shape (B, 3, 512, 512), got {tuple(image.shape)}")

        x = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        mean = torch.tensor(
            self.processor.image_mean,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            self.processor.image_std,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)

        return (x - mean) / std

    def _predict_depth(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_images(image)

        if self.freeze:
            with torch.inference_mode():
                outputs = self.depth_model(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.depth_model(pixel_values=pixel_values, return_dict=True)

        if not hasattr(outputs, "predicted_depth"):
            raise RuntimeError("Depth model output does not contain `predicted_depth`.")

        depth = outputs.predicted_depth

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        depth = F.interpolate(
            depth,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )

        if self.use_log_depth:
            depth = torch.log1p(torch.relu(depth))

        return depth

    def forward(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        depth = self._predict_depth(image)

        d64 = F.interpolate(depth, size=(64, 64), mode="bilinear", align_corners=False)
        d128 = F.interpolate(depth, size=(128, 128), mode="bilinear", align_corners=False)
        d256 = F.interpolate(depth, size=(256, 256), mode="bilinear", align_corners=False)
        d512 = depth

        return {
            64: self.enc64(d64),
            128: self.enc128(d128),
            256: self.enc256(d256),
            512: self.enc512(d512),
        }


class GatedDepthFusion(nn.Module):
    def __init__(self, rgb_dim: int, depth_dim: int):
        super().__init__()
        self.depth_proj = nn.Conv2d(depth_dim, rgb_dim, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(rgb_dim + depth_dim, rgb_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        d_proj = self.depth_proj(d)
        gate = self.gate(torch.cat([x, d], dim=1))
        return x + gate * d_proj


class PatchFeatureFiLMDecoderHeatmapRawDepthModel(nn.Module):
    """
    Same model structure, but depth is raw:
    - no robust normalization
    - no quantile clipping
    - no dx/dy gradient channels
    """

    def __init__(
        self,
        image_backbone_name: str = "facebook/dinov2-small",
        depth_backbone_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        backbone_image_size: int = 448,
        depth_image_size: int = 518,
        text_dim: int = 128,
        feat_dim: int = 128,
        depth_dim: int = 32,
        use_log_depth: bool = False,
    ):
        super().__init__()

        classes = sorted(np.unique(HiddenObjectsHeatmap("test").anno_dataset.class_arr))
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = self.classes
        num_classes = len(self.classes)

        self.patch_features = DinoPatchFeatureExtractor(
            backbone_name=image_backbone_name,
            image_size=backbone_image_size,
            freeze=True,
        )

        self.depth_features = InternalRawDepthFeatureExtractor(
            backbone_name=depth_backbone_name,
            image_size=depth_image_size,
            depth_dim=depth_dim,
            freeze=True,
            use_log_depth=use_log_depth,
        )

        backbone_dim = self.patch_features.hidden_dim

        self.class_embed = nn.Embedding(num_classes, text_dim)

        self.feat_proj = nn.Conv2d(backbone_dim, feat_dim, kernel_size=1)
        self.input_film = FiLM(text_dim=text_dim, feat_dim=feat_dim)

        self.dec32 = DecodeBlock(feat_dim, feat_dim, text_dim=text_dim)

        self.fuse64 = GatedDepthFusion(rgb_dim=feat_dim, depth_dim=depth_dim)
        self.dec64 = DecodeBlock(feat_dim, feat_dim // 2, text_dim=text_dim)

        self.fuse128 = GatedDepthFusion(rgb_dim=feat_dim // 2, depth_dim=depth_dim)
        self.dec128 = DecodeBlock(feat_dim // 2, feat_dim // 4, text_dim=text_dim)

        self.fuse256 = GatedDepthFusion(rgb_dim=feat_dim // 4, depth_dim=depth_dim)
        self.dec256 = DecodeBlock(feat_dim // 4, feat_dim // 4, text_dim=text_dim)

        self.fuse512 = GatedDepthFusion(rgb_dim=feat_dim // 4, depth_dim=depth_dim)
        self.dec512 = DecodeBlock(feat_dim // 4, feat_dim // 4, text_dim=text_dim)

        self.head = nn.Conv2d(feat_dim // 4, 1, kernel_size=1)

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

        x = self.patch_features(image)
        x = self.feat_proj(x)
        x = self.input_film(x, text_feat)

        depth_feats = self.depth_features(image)

        x = self.dec32(x, text_feat)

        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        x = self.fuse64(x, depth_feats[64])
        x = self.dec64(x, text_feat)

        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.fuse128(x, depth_feats[128])
        x = self.dec128(x, text_feat)

        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = self.fuse256(x, depth_feats[256])
        x = self.dec256(x, text_feat)

        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = self.fuse512(x, depth_feats[512])
        x = self.dec512(x, text_feat)

        heatmap = torch.sigmoid(self.head(x).squeeze(1))
        return heatmap