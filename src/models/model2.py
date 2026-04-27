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


class ConditionedDecodeBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, text_dim: int, depth_dim: int = 0):
        super().__init__()
        self.conv = ConvBlock(c_in + depth_dim, c_out)
        self.film = FiLM(text_dim=text_dim, feat_dim=c_out)

    def forward(
        self,
        x: torch.Tensor,
        text_feat: torch.Tensor,
        depth_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth_feat is not None:
            if depth_feat.shape[-2:] != x.shape[-2:]:
                depth_feat = F.interpolate(
                    depth_feat,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat([x, depth_feat], dim=1)

        x = self.conv(x)
        x = self.film(x, text_feat)
        return x


class PatchEncoder(nn.Module):
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


class DepthPyramid(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        image_size: int,
        dims_by_scale: dict[int, int],
        freeze: bool = True,
        use_log_depth: bool = False,
    ):
        super().__init__()

        required = {32, 64, 128, 256, 512}
        if dims_by_scale is None or (required - set(dims_by_scale)):
            raise ValueError("DepthPyramid requires dims_by_scale with keys {32,64,128,256,512}")

        self.dims_by_scale = dict(dims_by_scale)

        self.backbone = AutoModelForDepthEstimation.from_pretrained(backbone_name)
        self.processor = AutoImageProcessor.from_pretrained(backbone_name)

        if freeze:
            freeze_module(self.backbone)

        self.image_size = image_size
        self.freeze = freeze
        self.use_log_depth = use_log_depth

        self.enc512 = ConvBlock(1, self.dims_by_scale[512])
        self.enc256 = ConvBlock(self.dims_by_scale[512], self.dims_by_scale[256])
        self.enc128 = ConvBlock(self.dims_by_scale[256], self.dims_by_scale[128])
        self.enc64 = ConvBlock(self.dims_by_scale[128], self.dims_by_scale[64])
        self.enc32 = ConvBlock(self.dims_by_scale[64], self.dims_by_scale[32])

    def _prepare_images(self, image: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        mean = torch.tensor(self.processor.image_mean, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)

        return (x - mean) / std

    def _predict_depth(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_images(image)

        if self.freeze:
            with torch.inference_mode():
                outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)

        depth = outputs.predicted_depth

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        depth = F.interpolate(depth, size=(512, 512), mode="bilinear", align_corners=False)

        if self.use_log_depth:
            depth = torch.log1p(torch.relu(depth))

        return depth

    def forward(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        depth = self._predict_depth(image)

        d512 = self.enc512(depth)
        d256 = self.enc256(F.avg_pool2d(d512, 2))
        d128 = self.enc128(F.avg_pool2d(d256, 2))
        d64 = self.enc64(F.avg_pool2d(d128, 2))
        d32 = self.enc32(F.avg_pool2d(d64, 2))

        return {32: d32, 64: d64, 128: d128, 256: d256, 512: d512}


class DepthFiLMHeatmapModel(nn.Module):
    def __init__(
        self,
        image_backbone_name: str = "facebook/dinov2-small",
        depth_backbone_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        backbone_image_size: int = 448,
        depth_image_size: int = 518,
        text_dim: int = 128,
        feat_dim: int = 128,
        depth_dims_by_scale: dict[int, int] | None = None,
        use_log_depth: bool = False,
    ):
        super().__init__()

        if depth_dims_by_scale is None:
            depth_dims_by_scale = {512: 16, 256: 32, 128: 64, 64: 96, 32: 128}

        self.depth_dims_by_scale = dict(depth_dims_by_scale)

        classes = sorted(np.unique(HiddenObjectsHeatmap("test").anno_dataset.class_arr))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.rgb_encoder = PatchEncoder(image_backbone_name, backbone_image_size, freeze=True)

        self.depth_encoder = DepthPyramid(
            backbone_name=depth_backbone_name,
            image_size=depth_image_size,
            dims_by_scale=self.depth_dims_by_scale,
            freeze=True,
            use_log_depth=use_log_depth,
        )

        backbone_dim = self.rgb_encoder.hidden_dim

        self.class_embed = nn.Embedding(len(classes), text_dim)

        self.feat_proj = nn.Conv2d(backbone_dim, feat_dim, 1)

        self.input_fusion = nn.Conv2d(
            feat_dim + self.depth_dims_by_scale[32],
            feat_dim,
            1,
        )

        self.input_film = FiLM(text_dim, feat_dim)

        self.dec32 = ConditionedDecodeBlock(feat_dim, feat_dim, text_dim, self.depth_dims_by_scale[32])
        self.dec64 = ConditionedDecodeBlock(feat_dim, feat_dim // 2, text_dim, self.depth_dims_by_scale[64])
        self.dec128 = ConditionedDecodeBlock(feat_dim // 2, feat_dim // 4, text_dim, self.depth_dims_by_scale[128])
        self.dec256 = ConditionedDecodeBlock(feat_dim // 4, feat_dim // 4, text_dim, self.depth_dims_by_scale[256])
        self.dec512 = ConditionedDecodeBlock(feat_dim // 4, feat_dim // 4, text_dim, self.depth_dims_by_scale[512])

        self.head = nn.Conv2d(feat_dim // 4, 1, 1)

    def _text_to_indices(self, text: Sequence[str], device: torch.device) -> torch.Tensor:
        idx = [self.class_to_idx[t] for t in text]
        return torch.tensor(idx, dtype=torch.long, device=device)

    def forward(self, image: torch.Tensor, text: Sequence[str]) -> torch.Tensor:
        text_feat = self.class_embed(self._text_to_indices(text, image.device))

        depth_feats = self.depth_encoder(image)

        x = self.rgb_encoder(image)
        x = self.feat_proj(x)

        x = torch.cat([x, depth_feats[32]], dim=1)
        x = self.input_fusion(x)
        x = self.input_film(x, text_feat)

        x = self.dec32(x, text_feat, depth_feats[32])

        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        x = self.dec64(x, text_feat, depth_feats[64])

        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.dec128(x, text_feat, depth_feats[128])

        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = self.dec256(x, text_feat, depth_feats[256])

        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        x = self.dec512(x, text_feat, depth_feats[512])

        return torch.sigmoid(self.head(x).squeeze(1))