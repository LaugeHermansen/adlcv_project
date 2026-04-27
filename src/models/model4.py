from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoBackbone,
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    CLIPTextModel,
    CLIPTokenizer,
)

from src.hidden_objects_dataset import HiddenObjectsHeatmap


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()

        self.to_gamma_beta = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)

        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return x * (1.0 + gamma) + beta


class FusionBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int):
        super().__init__()

        self.proj = ConvBNAct(cin, cout)
        self.film = FiLM(cond_dim, cout)

        self.refine = nn.Sequential(
            ConvBNAct(cout, cout),
            ConvBNAct(cout, cout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.film(x, cond)
        x = self.refine(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, cin: int, skip_c: int, cout: int, cond_dim: int):
        super().__init__()

        self.proj = ConvBNAct(cin + skip_c, cout)
        self.film = FiLM(cond_dim, cout)

        self.refine = nn.Sequential(
            ConvBNAct(cout, cout),
            ConvBNAct(cout, cout),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.film(x, cond)
        x = self.refine(x)

        return x


class ObjectPlacementHeatmapModel(nn.Module):
    def __init__(
        self,
        image_size: int = 512,
        width: int = 128,
        class_dim: int = 64,
        cond_dim: int = 256,
        freeze_pretrained: bool = True,
        dino_name: str = "facebook/dinov2-small",
        backbone_name: str = "microsoft/swin-tiny-patch4-window7-224",
        depth_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        text_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()

        self.image_size = image_size
        self.width = width

        classes = sorted(
            np.unique(HiddenObjectsHeatmap("test").anno_dataset.class_arr)
        )

        self.class_names = [str(c) for c in classes]
        self.class_to_idx = {str(c): i for i, c in enumerate(classes)}
        self.num_classes = len(self.class_names)

        self.dino = AutoModel.from_pretrained(dino_name)

        self.backbone = AutoBackbone.from_pretrained(
            backbone_name,
            out_indices=(1, 2, 3, 4),
        )

        self.depth_processor = AutoImageProcessor.from_pretrained(depth_name)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_name)

        self.text_tokenizer = CLIPTokenizer.from_pretrained(text_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_name)

        if freeze_pretrained:
            self.freeze_module(self.dino)
            self.freeze_module(self.backbone)
            self.freeze_module(self.depth_model)
            self.freeze_module(self.text_encoder)

        dino_dim = self.dino.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size

        self.class_embedding = nn.Embedding(self.num_classes, class_dim)

        self.cond_proj = nn.Sequential(
            nn.Linear(class_dim + text_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.f4_proj = ConvBNAct(96, width // 2)
        self.f8_proj = ConvBNAct(192, width)
        self.f16_proj = ConvBNAct(384, width)
        self.f32_proj = ConvBNAct(768, width)

        self.dino_proj = ConvBNAct(dino_dim, width)
        self.depth_proj = ConvBNAct(1, width // 2)

        self.fuse32 = FusionBlock(
            cin=width + width,
            cout=width,
            cond_dim=cond_dim,
        )

        self.fuse16 = FusionBlock(
            cin=width + width,
            cout=width,
            cond_dim=cond_dim,
        )

        self.fuse8 = FusionBlock(
            cin=width + width + width // 2,
            cout=width,
            cond_dim=cond_dim,
        )

        self.fuse4 = FusionBlock(
            cin=width // 2 + width // 2,
            cout=width // 2,
            cond_dim=cond_dim,
        )

        self.up16 = UpBlock(
            cin=width,
            skip_c=width,
            cout=width,
            cond_dim=cond_dim,
        )

        self.up8 = UpBlock(
            cin=width,
            skip_c=width,
            cout=width,
            cond_dim=cond_dim,
        )

        self.up4 = UpBlock(
            cin=width,
            skip_c=width // 2,
            cout=width // 2,
            cond_dim=cond_dim,
        )

        self.heatmap_head = nn.Sequential(
            ConvBNAct(width // 2, width // 2),
            nn.Conv2d(width // 2, 1, kernel_size=1),
        )

    @staticmethod
    def freeze_module(module: nn.Module) -> None:
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        self.dino.eval()
        self.backbone.eval()
        self.depth_model.eval()
        self.text_encoder.eval()

        return self

    def infer_class_ids_from_text(
        self,
        text: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        class_ids = []

        for t in text:
            t_lower = str(t).lower()
            matches = []

            for class_name in self.class_names:
                name = class_name.lower()
                if name in t_lower:
                    matches.append(class_name)

            if len(matches) == 0:
                raise ValueError(
                    f"No class found in text: {t}. "
                    f"Known classes: {self.class_names}"
                )

            if len(matches) > 1:
                raise ValueError(
                    f"Multiple classes found in text: {t}. "
                    f"Matches: {matches}"
                )

            class_ids.append(self.class_to_idx[matches[0]])

        return torch.tensor(class_ids, dtype=torch.long, device=device)

    def encode_text(
        self,
        text: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        tokens = self.text_tokenizer(
            list(text),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self.text_encoder(**tokens)

        return outputs.pooler_output

    def extract_dino_features(self, image: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            image,
            size=(518, 518),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            outputs = self.dino(
                pixel_values=x,
                interpolate_pos_encoding=True,
            )

        tokens = outputs.last_hidden_state
        patch_tokens = tokens[:, 1:, :]

        b, n, c = patch_tokens.shape
        h = w = int(n ** 0.5)

        dino = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
        dino = self.dino_proj(dino)

        return dino

    def extract_depth_features(self, image: torch.Tensor) -> torch.Tensor:
        inputs = self.depth_processor(
            images=image,
            return_tensors="pt",
            do_rescale=False,
        ).to(image.device)

        with torch.no_grad():
            outputs = self.depth_model(**inputs)

        depth = outputs.predicted_depth

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        depth = F.interpolate(
            depth,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        d_min = depth.amin(dim=(2, 3), keepdim=True)
        d_max = depth.amax(dim=(2, 3), keepdim=True)

        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        depth = self.depth_proj(depth)

        return depth

    def extract_backbone_features(self, image: torch.Tensor):
        with torch.no_grad():
            outputs = self.backbone(image)

        f4, f8, f16, f32 = outputs.feature_maps

        f4 = self.f4_proj(f4)
        f8 = self.f8_proj(f8)
        f16 = self.f16_proj(f16)
        f32 = self.f32_proj(f32)

        return f4, f8, f16, f32

    def forward(
        self,
        image: torch.Tensor,
        text: Sequence[str],
    ) -> torch.Tensor:
        b, _, h, w = image.shape
        device = image.device

        if len(text) != b:
            raise ValueError(
                f"Batch size mismatch: image batch has {b}, "
                f"but text has {len(text)} items."
            )

        class_id = self.infer_class_ids_from_text(text, device)

        class_vec = self.class_embedding(class_id)
        text_vec = self.encode_text(text, device)

        cond = torch.cat([class_vec, text_vec], dim=-1)
        cond = self.cond_proj(cond)

        f4, f8, f16, f32 = self.extract_backbone_features(image)

        dino = self.extract_dino_features(image)
        depth = self.extract_depth_features(image)

        dino32 = F.interpolate(
            dino,
            size=f32.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        dino16 = F.interpolate(
            dino,
            size=f16.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        dino8 = F.interpolate(
            dino,
            size=f8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        depth8 = F.interpolate(
            depth,
            size=f8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        depth4 = F.interpolate(
            depth,
            size=f4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x32 = self.fuse32(
            torch.cat([f32, dino32], dim=1),
            cond,
        )

        x16 = self.fuse16(
            torch.cat([f16, dino16], dim=1),
            cond,
        )

        x8 = self.fuse8(
            torch.cat([f8, dino8, depth8], dim=1),
            cond,
        )

        x4 = self.fuse4(
            torch.cat([f4, depth4], dim=1),
            cond,
        )

        x = self.up16(x32, x16, cond)
        x = self.up8(x, x8, cond)
        x = self.up4(x, x4, cond)

        logits = self.heatmap_head(x)

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        return torch.sigmoid(logits)


def placement_heatmap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, target)

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    intersection = (pred_flat * target_flat).sum(dim=1)

    dice = 1.0 - (2.0 * intersection + eps) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps
    )

    return bce + dice.mean()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ObjectPlacementHeatmapModel(
        freeze_pretrained=True,
    ).to(device)

    image = torch.rand(2, 3, 512, 512, device=device)

    text = [
        "place a chair in the room",
        "place a lamp on the table",
    ]

    with torch.no_grad():
        heatmap = model(image, text)

    print(heatmap.shape)
    print(heatmap.min().item(), heatmap.max().item())