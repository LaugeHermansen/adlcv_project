"""
Joint augmentations for image + heatmap pairs.

All geometric transforms are applied identically to both tensors so that
the heatmap stays spatially aligned with the image.

Rotation handling
-----------------
After rotating by angle θ the canvas corners go black.  We crop out those
black borders by keeping the largest axis-aligned square that fits inside
the rotated image, then resize back to the original 512x512.

For a square of side S rotated by θ the crop side length is:
    crop_size = S / (cos θ + sin θ)
which guarantees zero black pixels for any |θ| ≤ 45°.
"""

from __future__ import annotations

import math
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, Dataset

from src.hidden_objects_dataset import HiddenObjectsHeatmap


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _inscribed_crop_size(size: int, angle_deg: float) -> int:
    """
    Largest square crop (no black) for a square image of `size` px
    rotated by `angle_deg` degrees.
    """
    theta = math.radians(abs(angle_deg) % 90)
    if theta > math.pi / 4:
        theta = math.pi / 2 - theta
    return max(1, int(size / (math.cos(theta) + math.sin(theta))))


# ------------------------------------------------------------------ #
# Individual transforms (all accept and return (image, heatmap))      #
# ------------------------------------------------------------------ #

class JointRandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: torch.Tensor, heatmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            image = TF.hflip(image)
            heatmap = TF.hflip(heatmap)
        return image, heatmap


class JointRandomRotation:
    """
    Rotates both tensors by the same random angle, then crops to remove
    black border pixels and resizes back to the original spatial size.
    """

    def __init__(self, max_degrees: float = 15.0):
        self.max_degrees = max_degrees

    def __call__(
        self, image: torch.Tensor, heatmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angle = random.uniform(-self.max_degrees, self.max_degrees)
        H, W = image.shape[-2], image.shape[-1]

        # Rotate (black fill = 0)
        image = TF.rotate(image, angle, fill=0)
        heatmap = TF.rotate(heatmap, angle, fill=0)

        # Crop to the largest inscribed square (no black pixels)
        crop = _inscribed_crop_size(H, angle)
        image = TF.center_crop(image, [crop, crop])
        heatmap = TF.center_crop(heatmap, [crop, crop])

        # Resize back to original size
        image = TF.resize(
            image, [H, W],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        heatmap = TF.resize(
            heatmap, [H, W],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )

        return image, heatmap


class ImageOnlyColorJitter:
    """Color jitter applied to image only; heatmap passes through unchanged."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(
        self, image: torch.Tensor, heatmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.jitter(image)
        return image, heatmap


class ImageOnlyGaussianBlur:
    """Random Gaussian blur applied to image only."""

    def __init__(self, kernel_size: int = 5, sigma: tuple[float, float] = (0.1, 1.5), p: float = 0.3):
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = p

    def __call__(
        self, image: torch.Tensor, heatmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            image = self.blur(image)
        return image, heatmap


# ------------------------------------------------------------------ #
# Composed pipeline                                                    #
# ------------------------------------------------------------------ #

class JointAugmentation:
    """
    Full augmentation pipeline for (image, heatmap) pairs.

    Geometric transforms are applied to both tensors identically.
    Photometric transforms are applied to the image only.
    """

    def __init__(
        self,
        flip_p: float = 0.5,
        max_rotation: float = 15.0,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05,
        blur_p: float = 0.3,
    ):
        self.transforms = [
            JointRandomHorizontalFlip(p=flip_p),
            JointRandomRotation(max_degrees=max_rotation),
            ImageOnlyColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            ImageOnlyGaussianBlur(p=blur_p),
        ]

    def __call__(
        self, image: torch.Tensor, heatmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            image, heatmap = t(image, heatmap)
        return image, heatmap


# ------------------------------------------------------------------ #
# Augmented dataset wrapper                                            #
# ------------------------------------------------------------------ #

class AugmentedHiddenObjectsHeatmap(Dataset):
    """
    Wraps an existing HiddenObjectsHeatmap and applies joint augmentations
    at __getitem__ time.  Intended to be concatenated with the base dataset
    so that the training set is larger without duplicating disk reads.
    """

    def __init__(
        self,
        base_dataset: HiddenObjectsHeatmap,
        augmentation: JointAugmentation | None = None,
    ):
        self.base = base_dataset
        self.augmentation = augmentation or JointAugmentation()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base[idx]
        image, heatmap = self.augmentation(sample["image"], sample["heatmap"])
        return {**sample, "image": image, "heatmap": heatmap}


# ------------------------------------------------------------------ #
# Factory used by train.py                                             #
# ------------------------------------------------------------------ #

def build_augmented_train_dataset(
    base_dataset: HiddenObjectsHeatmap,
    n_extra_copies: int = 1,
    augmentation: JointAugmentation | None = None,
) -> ConcatDataset:
    """
    Returns ConcatDataset([base, aug_copy_1, ..., aug_copy_n]).

    The base dataset is shared across all augmented copies — only one
    HiddenObjectsHeatmap instance is created regardless of n_extra_copies.

    Args:
        base_dataset:   Already-constructed HiddenObjectsHeatmap(split="train").
        n_extra_copies: Number of augmented copies to append.
                        Total dataset size = (1 + n_extra_copies) * len(base).
        augmentation:   Custom JointAugmentation; defaults to JointAugmentation().
    """
    aug = augmentation or JointAugmentation()
    augmented_copies = [
        AugmentedHiddenObjectsHeatmap(base_dataset, augmentation=aug)
        for _ in range(n_extra_copies)
    ]
    return ConcatDataset([base_dataset] + augmented_copies)
