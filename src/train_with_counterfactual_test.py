from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.hidden_objects_dataset import HiddenObjectsHeatmap
from src.train import (
    HeatmapLightningModule,
    FixedInspectionPanelCallback,
    _class_to_import_path,
    _ensure_heatmap_2d,
    _normalize_image_for_display,
)


# ============================================================
# Counterfactual class pairs
# ============================================================

COUNTERFACTUAL_CLASSES: dict[str, list[str]] = {
    "kite": ["car", "microwave", "toothbrush", "parking meter"],
    "car": ["kite", "vase", "toothbrush", "fork"],
    "boat": ["microwave", "toothbrush", "bed", "keyboard"],
    "airplane": ["fork", "vase", "toothbrush", "microwave"],
    "bench": ["kite", "toothbrush", "microwave", "wine glass"],
    "bed": ["traffic light", "kite", "parking meter", "fire hydrant"],
    "vase": ["car", "airplane", "boat", "motorcycle"],
    "fork": ["car", "airplane", "bench", "bed"],
    "spoon": ["car", "airplane", "bench", "bed"],
    "toothbrush": ["car", "boat", "bench", "bed"],
    "traffic light": ["bed", "cake", "pizza", "toothbrush"],
    "parking meter": ["bed", "cake", "pizza", "toothbrush"],
    "fire hydrant": ["bed", "cake", "pizza", "toothbrush"],
    "surfboard": ["microwave", "bed", "toothbrush", "keyboard"],
    "skateboard": ["microwave", "bed", "toothbrush", "vase"],
    "microwave": ["kite", "boat", "airplane", "surfboard"],
    "keyboard": ["kite", "boat", "airplane", "bench"],
    "laptop": ["kite", "boat", "fire hydrant", "traffic light"],
    "tv": ["kite", "boat", "toothbrush", "parking meter"],
}


# ============================================================
# Collate function for train/val/test
# ============================================================

def heatmap_counterfactual_collate(batch: list[dict]) -> dict:
    """
    Collates samples while preserving bbox annotations.

    boxes are ragged because each image-class pair can have a different number
    of candidate placement boxes. Therefore, boxes are kept as a list of tensors.
    """
    return {
        "image": torch.stack([sample["image"] for sample in batch], dim=0),
        "heatmap": torch.stack([sample["heatmap"] for sample in batch], dim=0),
        "class": [sample["class"] for sample in batch],
        "boxes": [sample["boxes"] for sample in batch],
        "labels": [sample["labels"] for sample in batch],
        "confidences": [sample["confidences"] for sample in batch],
        "entry_id": [sample["entry_id"] for sample in batch],
        "bg_path": [sample["bg_path"] for sample in batch],
    }


# ============================================================
# Dataset splitting
# ============================================================

def deterministic_subset_split(
    dataset,
    fractions: tuple[float, float],
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """
    Deterministically splits a dataset into two subsets.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")

    n = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_first = int(round(fractions[0] * n))

    first_indices = indices[:n_first].tolist()
    second_indices = indices[n_first:].tolist()

    return Subset(dataset, first_indices), Subset(dataset, second_indices)


# ============================================================
# Box scoring
# ============================================================

def score_heatmap_inside_boxes(
    heatmap: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor | None = None,
    confidences: torch.Tensor | None = None,
    use_only_positive_boxes: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes average predicted heatmap value inside annotated boxes.

    heatmap:
        Tensor of shape (H, W)

    boxes:
        Tensor of shape (N, 4), using x, y, w, h format.
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Expected heatmap shape (H, W), got {tuple(heatmap.shape)}")

    H, W = heatmap.shape
    device = heatmap.device

    boxes = boxes.to(device=device, dtype=torch.float32)

    if labels is not None:
        labels = labels.to(device=device)

    if confidences is not None:
        confidences = confidences.to(device=device, dtype=torch.float32)

    per_box_scores = []
    per_box_weights = []

    for i, box in enumerate(boxes):
        if labels is not None and use_only_positive_boxes and labels[i].item() <= 0:
            continue

        x, y, w, h = box

        x0 = int(torch.floor(x).item())
        y0 = int(torch.floor(y).item())
        x1 = int(torch.ceil(x + w).item())
        y1 = int(torch.ceil(y + h).item())

        x0 = max(0, min(W, x0))
        x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0))
        y1 = max(0, min(H, y1))

        if x1 <= x0 or y1 <= y0:
            continue

        patch = heatmap[y0:y1, x0:x1]
        box_score = patch.mean()

        if confidences is not None:
            box_weight = confidences[i].clamp_min(0.0)
        else:
            box_weight = torch.tensor(1.0, device=device)

        per_box_scores.append(box_score)
        per_box_weights.append(box_weight)

    if len(per_box_scores) == 0:
        return torch.tensor(float("nan"), device=device)

    scores = torch.stack(per_box_scores)
    weights = torch.stack(per_box_weights)

    return (scores * weights).sum() / (weights.sum() + eps)


# ============================================================
# Visualization helpers
# ============================================================

def to_numpy(x):
    """
    Converts tensors/lists/arrays to a NumPy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalizes an array.

    If the array is constant, returns zeros with the same shape.
    """
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin) if xmax > xmin else np.zeros_like(x)


def prepare_image(image) -> np.ndarray:
    """
    Converts an image tensor/array into displayable HWC RGB format in [0, 1].

    Accepts:
        CHW tensor
        HWC array
        grayscale channel image
    """
    image = to_numpy(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.max() > 1.0:
        image = image / 255.0

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    return image


def plot_boxes_with_confidence(
    *,
    image,
    boxes,
    labels,
    confidences,
    ax,
    max_boxes: int | None = 20,
    selection: Literal["top", "random"] = "top",
    seed: int | None = None,
    linewidth: float = 2.0,
    use_only_positives: bool = True,
) -> None:
    """
    Plots image with bbox edges colored by confidence.

    Color convention:
        low confidence  -> red
        high confidence -> green

    Only positive-label boxes are shown by default.
    """
    image = prepare_image(image)
    boxes = to_numpy(boxes)
    labels = to_numpy(labels)
    confidences = to_numpy(confidences)

    if use_only_positives:
        keep = labels == 1
        boxes = boxes[keep]
        confidences = confidences[keep]

    if max_boxes is not None and len(boxes) > max_boxes:
        if selection == "random":
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(boxes), size=max_boxes, replace=False)
        else:
            keep = np.argsort(confidences)[-max_boxes:]

        boxes = boxes[keep]
        confidences = confidences[keep]

    ax.imshow(image)
    ax.axis("off")

    if len(boxes) == 0:
        return

    conf = normalize_minmax(confidences)

    patches = []
    colors = []

    for box, c in zip(boxes, conf):
        x, y, w, h = box
        patches.append(Rectangle((x, y), w, h))
        colors.append((1.0 - float(c), float(c), 0.0, 1.0))

    ax.add_collection(
        PatchCollection(
            patches,
            facecolor="none",
            edgecolors=colors,
            linewidths=linewidth,
        )
    )


# ============================================================
# Counterfactual visual callback
# ============================================================

class CounterfactualInspectionPanelCallback(L.Callback):
    """
    Saves fixed visual examples from a validation or test split.

    Each row shows:

        image
        GT heatmap
        prediction with true prompt
        prediction with first wrong prompt
        image with positive boxes colored by confidence
    """

    def __init__(
        self,
        dataset,
        counterfactual_classes: dict[str, list[str]],
        split_name: Literal["val", "test"],
        max_examples: int = 12,
        seed: int = 17,
        max_boxes: int | None = 20,
        box_selection: Literal["top", "random"] = "top",
    ):
        super().__init__()

        self.dataset = dataset
        self.counterfactual_classes = counterfactual_classes
        self.split_name = split_name
        self.max_examples = max_examples
        self.seed = seed
        self.max_boxes = max_boxes
        self.box_selection = box_selection

        self.fixed_batch = None
        self.fixed_indices = None

    def setup(self, trainer, pl_module, stage):
        """
        Pre-samples fixed examples.

        For validation visuals, setup runs before fitting.
        For test visuals, setup runs before testing.
        """
        if self.fixed_batch is not None:
            return

        if self.split_name == "val" and stage not in {"fit", "validate"}:
            return

        if self.split_name == "test" and stage != "test":
            return

        valid_indices = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            true_class = sample["class"]

            if true_class in self.counterfactual_classes:
                valid_indices.append(idx)

        if len(valid_indices) == 0:
            raise ValueError(
                f"No {self.split_name} examples found whose class exists in counterfactual_classes."
            )

        rng = np.random.default_rng(self.seed)
        n = min(self.max_examples, len(valid_indices))
        self.fixed_indices = rng.choice(valid_indices, size=n, replace=False).tolist()

        samples = [self.dataset[i] for i in self.fixed_indices]
        self.fixed_batch = heatmap_counterfactual_collate(samples)

    @torch.no_grad()
    def _save_panel(self, trainer, pl_module, epoch_label: str):
        if self.fixed_batch is None:
            return

        device = pl_module.device
        batch = self.fixed_batch

        image = batch["image"].to(device)
        gt_heatmap = _ensure_heatmap_2d(batch["heatmap"].to(device))
        true_classes = batch["class"]

        true_pred = _ensure_heatmap_2d(pl_module(image, true_classes))

        wrong_classes = []

        for true_class in true_classes:
            candidates = self.counterfactual_classes.get(true_class, None)

            if not candidates:
                wrong_classes.append(true_class)
            else:
                wrong_classes.append(candidates[0])

        wrong_pred = _ensure_heatmap_2d(pl_module(image, wrong_classes))

        image_cpu = image.detach().cpu()
        gt_cpu = gt_heatmap.detach().cpu()
        true_pred_cpu = true_pred.detach().cpu().clamp(0.0, 1.0)
        wrong_pred_cpu = wrong_pred.detach().cpu().clamp(0.0, 1.0)

        n = image_cpu.shape[0]

        fig, axes = plt.subplots(
            n,
            5,
            figsize=(20, 4 * n),
            squeeze=False,
        )

        for i in range(n):
            img = _normalize_image_for_display(image_cpu[i])

            gt = gt_cpu[i].numpy()
            pred_true = true_pred_cpu[i].numpy()
            pred_wrong = wrong_pred_cpu[i].numpy()

            true_class = true_classes[i]
            wrong_class = wrong_classes[i]

            boxes = batch["boxes"][i]
            labels = batch["labels"][i]
            confidences = batch["confidences"][i]

            matched_score = score_heatmap_inside_boxes(
                heatmap=true_pred_cpu[i],
                boxes=boxes,
                labels=labels,
                confidences=confidences,
                use_only_positive_boxes=True,
            )

            wrong_score = score_heatmap_inside_boxes(
                heatmap=wrong_pred_cpu[i],
                boxes=boxes,
                labels=labels,
                confidences=confidences,
                use_only_positive_boxes=True,
            )

            gap = matched_score - wrong_score

            ax0, ax1, ax2, ax3, ax4 = axes[i]

            ax0.imshow(img)
            ax0.set_title(
                f"image\n"
                f"idx={self.fixed_indices[i]}\n"
                f"class={true_class}"
            )
            ax0.axis("off")

            ax1.imshow(gt, cmap="jet", vmin=0.0, vmax=1.0)
            ax1.set_title("GT heatmap")
            ax1.axis("off")

            ax2.imshow(pred_true, cmap="jet", vmin=0.0, vmax=1.0)
            ax2.set_title(
                f'correct: "{true_class}"\n'
                f"box score={matched_score.item():.4f}"
            )
            ax2.axis("off")

            ax3.imshow(pred_wrong, cmap="jet", vmin=0.0, vmax=1.0)
            ax3.set_title(
                f'wrong: "{wrong_class}"\n'
                f"box score={wrong_score.item():.4f}\n"
                f"gap={gap.item():.4f}"
            )
            ax3.axis("off")

            plot_boxes_with_confidence(
                image=image_cpu[i],
                boxes=boxes,
                labels=labels,
                confidences=confidences,
                ax=ax4,
                max_boxes=self.max_boxes,
                selection=self.box_selection,
                seed=self.seed + i,
                linewidth=2.0,
                use_only_positives=True,
            )
            ax4.set_title(
                f"positive boxes by confidence\n"
                f"red=low, green=high"
            )

        fig.suptitle(
            f"Counterfactual {self.split_name}-set inspection — {epoch_label}",
            fontsize=16,
        )
        fig.tight_layout()

        out_dir = Path(trainer.log_dir) / f"counterfactual_{self.split_name}_panels"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{epoch_label}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved counterfactual {self.split_name} panel to: {out_path}")

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.split_name != "val":
            return

        if trainer.sanity_checking:
            return

        self._save_panel(
            trainer=trainer,
            pl_module=pl_module,
            epoch_label=f"epoch_{trainer.current_epoch:03d}",
        )

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        if self.split_name != "test":
            return

        self._save_panel(
            trainer=trainer,
            pl_module=pl_module,
            epoch_label="test",
        )


# ============================================================
# Lightning module with counterfactual validation/test logic
# ============================================================

class CounterfactualHeatmapLightningModule(HeatmapLightningModule):
    """
    Extends your existing HeatmapLightningModule.

    Training:
        standard supervised heatmap loss

    Validation:
        standard supervised heatmap metrics
        + counterfactual validation metrics

    Test:
        counterfactual test metrics
    """

    def __init__(
        self,
        model_class_path: str,
        model_config: dict[str, Any],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        counterfactual_classes: dict[str, list[str]] | None = None,
    ):
        super().__init__(
            model_class_path=model_class_path,
            model_config=model_config,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.counterfactual_classes = counterfactual_classes or COUNTERFACTUAL_CLASSES

    def _counterfactual_eval_step(
        self,
        batch,
        stage: Literal["val", "test"],
    ):
        """
        Shared counterfactual scoring for validation and test.

        For each sample:
            1. predict with the true class prompt;
            2. score the predicted heatmap inside the true-class positive boxes;
            3. predict with deliberately wrong class prompts;
            4. score those wrong-prompt heatmaps inside the same true-class boxes;
            5. log matched-vs-wrong gap.
        """
        image = batch["image"]
        true_classes = batch["class"]

        true_pred = _ensure_heatmap_2d(self(image, true_classes))

        if true_pred.ndim != 3:
            raise ValueError(f"Expected true_pred shape (B,H,W), got {tuple(true_pred.shape)}")

        matched_scores = []
        counterfactual_mean_scores = []
        counterfactual_max_scores = []
        mean_gaps = []
        max_gaps = []
        mean_ratios = []
        max_ratios = []

        B = image.shape[0]

        for i in range(B):
            true_class = true_classes[i]
            wrong_classes = self.counterfactual_classes.get(true_class, None)

            if not wrong_classes:
                continue

            boxes = batch["boxes"][i]
            labels = batch["labels"][i]
            confidences = batch["confidences"][i]

            matched_score = score_heatmap_inside_boxes(
                heatmap=true_pred[i],
                boxes=boxes,
                labels=labels,
                confidences=confidences,
                use_only_positive_boxes=True,
            )

            if torch.isnan(matched_score):
                continue

            wrong_class_scores = []

            for wrong_class in wrong_classes:
                wrong_pred = _ensure_heatmap_2d(
                    self(image[i : i + 1], [wrong_class])
                )

                if wrong_pred.ndim == 3:
                    wrong_heatmap = wrong_pred[0]
                elif wrong_pred.ndim == 2:
                    wrong_heatmap = wrong_pred
                else:
                    raise ValueError(
                        f"Expected wrong_pred shape (1,H,W) or (H,W), got {tuple(wrong_pred.shape)}"
                    )

                wrong_score = score_heatmap_inside_boxes(
                    heatmap=wrong_heatmap,
                    boxes=boxes,
                    labels=labels,
                    confidences=confidences,
                    use_only_positive_boxes=True,
                )

                if not torch.isnan(wrong_score):
                    wrong_class_scores.append(wrong_score)

            if len(wrong_class_scores) == 0:
                continue

            wrong_scores = torch.stack(wrong_class_scores)

            counterfactual_mean_score = wrong_scores.mean()
            counterfactual_max_score = wrong_scores.max()

            mean_gap = matched_score - counterfactual_mean_score
            max_gap = matched_score - counterfactual_max_score

            mean_ratio = matched_score / (counterfactual_mean_score + 1e-8)
            max_ratio = matched_score / (counterfactual_max_score + 1e-8)

            matched_scores.append(matched_score)
            counterfactual_mean_scores.append(counterfactual_mean_score)
            counterfactual_max_scores.append(counterfactual_max_score)
            mean_gaps.append(mean_gap)
            max_gaps.append(max_gap)
            mean_ratios.append(mean_ratio)
            max_ratios.append(max_ratio)

        if len(matched_scores) == 0:
            return None

        matched_scores = torch.stack(matched_scores)
        counterfactual_mean_scores = torch.stack(counterfactual_mean_scores)
        counterfactual_max_scores = torch.stack(counterfactual_max_scores)
        mean_gaps = torch.stack(mean_gaps)
        max_gaps = torch.stack(max_gaps)
        mean_ratios = torch.stack(mean_ratios)
        max_ratios = torch.stack(max_ratios)

        mean_accuracy = (mean_gaps > 0).float().mean()
        max_accuracy = (max_gaps > 0).float().mean()

        self.log(
            f"{stage}_cf_matched_box_score",
            matched_scores.mean(),
            prog_bar=True,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_wrong_mean_box_score",
            counterfactual_mean_scores.mean(),
            prog_bar=True,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_wrong_max_box_score",
            counterfactual_max_scores.mean(),
            prog_bar=False,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_mean_gap",
            mean_gaps.mean(),
            prog_bar=True,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_max_gap",
            max_gaps.mean(),
            prog_bar=False,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_mean_ratio",
            mean_ratios.mean(),
            prog_bar=False,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_max_ratio",
            max_ratios.mean(),
            prog_bar=False,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_mean_accuracy",
            mean_accuracy,
            prog_bar=True,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_cf_max_accuracy",
            max_accuracy,
            prog_bar=False,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )

        return {
            "matched_score": matched_scores.mean(),
            "wrong_mean_score": counterfactual_mean_scores.mean(),
            "wrong_max_score": counterfactual_max_scores.mean(),
            "mean_gap": mean_gaps.mean(),
            "max_gap": max_gaps.mean(),
            "mean_ratio": mean_ratios.mean(),
            "max_ratio": max_ratios.mean(),
            "mean_accuracy": mean_accuracy,
            "max_accuracy": max_accuracy,
        }

    def validation_step(self, batch, batch_idx):
        """
        Runs the original supervised validation metrics and the new
        counterfactual validation metrics.
        """
        self._shared_step(batch, "val")
        return self._counterfactual_eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._counterfactual_eval_step(batch, "test")


# ============================================================
# Main experiment
# ============================================================

def train_heatmap_experiment_with_counterfactual_test(
    *,
    model_class: type[nn.Module],
    model_config: dict[str, Any],
    experiment_name: str,
    max_epochs: int = 10,
    batch_size: int = 8,
    num_workers: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_inspection_examples: int = 10,
    inspection_seed: int = 3,
    split_seed: int = 123,
    resume_from_checkpoint: str | Path | None = None,
):
    train_ds = HiddenObjectsHeatmap(split="train")

    official_test_ds = HiddenObjectsHeatmap(split="test")

    val_ds, counterfactual_test_ds = deterministic_subset_split(
        official_test_ds,
        fractions=(0.5, 0.5),
        seed=split_seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=heatmap_counterfactual_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=heatmap_counterfactual_collate,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        counterfactual_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=heatmap_counterfactual_collate,
        pin_memory=torch.cuda.is_available(),
    )

    lit_model = CounterfactualHeatmapLightningModule(
        model_class_path=_class_to_import_path(model_class),
        model_config=model_config,
        lr=lr,
        weight_decay=weight_decay,
        counterfactual_classes=COUNTERFACTUAL_CLASSES,
    )

    logger = TensorBoardLogger(
        save_dir="runs_counterfactual",
        name=experiment_name,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=Path(logger.log_dir) / "checkpoints",
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    standard_vis_cb = FixedInspectionPanelCallback(
        train_dataset=train_ds,
        val_dataset=val_ds,
        max_examples=num_inspection_examples,
        train_seed=inspection_seed,
        val_seed=inspection_seed + 1,
    )

    counterfactual_val_vis_cb = CounterfactualInspectionPanelCallback(
        dataset=val_ds,
        counterfactual_classes=COUNTERFACTUAL_CLASSES,
        split_name="val",
        max_examples=num_inspection_examples,
        seed=inspection_seed + 2,
        max_boxes=20,
        box_selection="top",
    )

    counterfactual_test_vis_cb = CounterfactualInspectionPanelCallback(
        dataset=counterfactual_test_ds,
        counterfactual_classes=COUNTERFACTUAL_CLASSES,
        split_name="test",
        max_examples=num_inspection_examples,
        seed=inspection_seed + 3,
        max_boxes=20,
        box_selection="top",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            standard_vis_cb,
            counterfactual_val_vis_cb,
            counterfactual_test_vis_cb,
        ],
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
    )

    best_ckpt_path = checkpoint_cb.best_model_path

    if best_ckpt_path:
        test_model = CounterfactualHeatmapLightningModule.load_from_checkpoint(
            best_ckpt_path,
            counterfactual_classes=COUNTERFACTUAL_CLASSES,
        )
    else:
        test_model = lit_model

    test_results = trainer.test(
        test_model,
        dataloaders=test_loader,
    )

    return trainer, lit_model, test_results