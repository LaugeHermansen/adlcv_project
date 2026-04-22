from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional, Sequence
import re

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.hidden_objects_dataset import HiddenObjectsHeatmap, heatmap_collate


# ============================================================
# Utilities
# ============================================================


def _normalize_image_for_display(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().float()
    arr = image.permute(1, 2, 0).numpy()
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def _ensure_heatmap_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Strips only a singleton channel dimension.

    Accepted examples:
        (B,1,H,W) -> (B,H,W)
        (1,H,W)   -> (H,W)
        (B,H,W)   -> unchanged
        (H,W)     -> unchanged
    """
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3 and x.shape[0] == 1:
        return x[0]
    return x

def _class_to_import_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _import_from_path(path: str) -> type:
    """
    Imports a class from a fully qualified path like:
        'src.models.ClipWithTransformerFusion'
    """
    if "." not in path:
        raise ValueError(f"Expected fully qualified import path, got {path!r}")

    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Could not find class {class_name!r} in module {module_name!r}") from e
    return cls

def _extract_logger_version_from_ckpt(ckpt_path: str | Path) -> int | None:
    ckpt_path = Path(ckpt_path)
    for part in ckpt_path.parts:
        m = re.fullmatch(r"version_(\d+)", part)
        if m:
            return int(m.group(1))
    return None

# ============================================================
# Lightning module
# ============================================================

def weighted_heatmap_loss(
    pred,                  # predicted heatmap after sigmoid, shape (B, H, W)
    target,                # in [0, 1], shape (B, H, W)
    alpha=0.5,             # inverse-mass strength
    beta=2.0,              # extra weight for target-active pixels
    eps=1e-6,
):
    ### quick idea. maybe needs some more work or better ideas?
    B, H, W = target.shape

    mass = target.sum(dim=(1, 2), keepdim=True)
    sample_weight = ((H * W) / (mass + eps)).pow(alpha)

    # High target pixels get more weight; tiny positive regions get even more
    weight = 1.0 + beta * target * sample_weight

    loss = weight * (pred - target).pow(2)
    return loss.mean()

class HeatmapLightningModule(L.LightningModule):
    """
    Lightning wrapper that stores enough information to reconstruct the wrapped
    model automatically from a checkpoint.

    The checkpoint will contain:
        - model_class_path: fully qualified import path
        - model_config: constructor kwargs
        - lr, weight_decay

    Because these are saved as hyperparameters, the following works without the
    caller re-supplying model_class or model_config:

        HeatmapLightningModule.load_from_checkpoint(checkpoint_path)
    """

    def __init__(
        self,
        model_class_path: str,
        model_config: dict[str, Any],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        self.model_class_path = model_class_path
        self.model_config = dict(model_config)
        self.lr = lr
        self.weight_decay = weight_decay

        model_class = _import_from_path(model_class_path)
        self.model = model_class(**self.model_config)

        self.loss_fn = nn.MSELoss()

        # Save everything needed to rebuild the wrapped model on load.
        self.save_hyperparameters()

    def forward(self, image: torch.Tensor, class_name: str) -> torch.Tensor:
        return self.model(image, class_name)

    def _shared_step(self, batch, stage: str):
        image = batch["image"]
        target = _ensure_heatmap_2d(batch["heatmap"])
        cls = batch["class"]

        pred = _ensure_heatmap_2d(self(image, cls))

        if pred.ndim != 3:
            raise ValueError(f"Expected predicted heatmap shape (B,H,W), got {tuple(pred.shape)}")
        if target.ndim != 3:
            raise ValueError(f"Expected target heatmap shape (B,H,W), got {tuple(target.shape)}")
        if pred.shape != target.shape:
            raise ValueError(
                f"Predicted and target heatmaps must have the same shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )

        loss = self.loss_fn(pred, target)
        mae = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)

        bs = image.size(0)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_mae", mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_mse", mse, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ============================================================
# Visualization callback
# ============================================================

def _sample_fixed_indices(dataset_len: int, max_examples: int, seed: int) -> list[int]:
    """
    Randomly samples up to max_examples unique indices from [0, dataset_len),
    using a fixed seed for reproducibility.
    """
    n = min(max_examples, dataset_len)
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_len, size=n, replace=False).tolist()

class FixedInspectionPanelCallback(Callback):
    """
    Saves a fixed set of train and validation examples after each validation epoch.

    Each row contains:
        image | GT heatmap | predicted heatmap | overlay

    The inspected examples are sampled once, randomly but reproducibly, from
    each dataset using the provided seed(s).
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        max_examples: int = 10,
        train_seed: int = 0,
        val_seed: int = 1,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_examples = max_examples

        self.train_indices = _sample_fixed_indices(len(train_dataset), max_examples, train_seed)
        self.val_indices = _sample_fixed_indices(len(val_dataset), max_examples, val_seed)

        self.train_fixed_batch = None
        self.val_fixed_batch = None

    def setup(self, trainer, pl_module, stage):
        if self.train_fixed_batch is None:
            train_samples = [self.train_dataset[i] for i in self.train_indices]
            self.train_fixed_batch = heatmap_collate(train_samples)

        if self.val_fixed_batch is None:
            val_samples = [self.val_dataset[i] for i in self.val_indices]
            self.val_fixed_batch = heatmap_collate(val_samples)

    @torch.no_grad()
    def _save_panel(self, trainer, pl_module, batch, split_name: str, indices: list[int]):
        device = pl_module.device

        image = batch["image"].to(device)
        target = _ensure_heatmap_2d(batch["heatmap"].to(device))
        cls = batch["class"]

        pred = _ensure_heatmap_2d(pl_module(image, cls))

        image = image.cpu()
        target = target.cpu()
        pred = pred.cpu()

        n = min(self.max_examples, image.shape[0])
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = np.array([axes])

        for i in range(n):
            img = _normalize_image_for_display(image[i])
            gt = target[i].numpy()
            pr = np.clip(pred[i].numpy(), 0.0, 1.0)

            ax0, ax1, ax2, ax3 = axes[i]

            ax0.imshow(img)
            ax0.set_title(f"image\nclass={batch['class'][i]}\nidx={indices[i]}")
            ax0.axis("off")

            ax1.imshow(gt, cmap="jet", vmin=0.0, vmax=1.0)
            ax1.set_title("GT heatmap")
            ax1.axis("off")

            ax2.imshow(pr, cmap="jet", vmin=0.0, vmax=1.0)
            ax2.set_title("pred heatmap")
            ax2.axis("off")

            ax3.imshow(img)
            ax3.imshow(pr, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
            ax3.set_title("overlay")
            ax3.axis("off")

        fig.suptitle(f"{split_name.capitalize()} examples — epoch {trainer.current_epoch}")
        fig.tight_layout()

        out_dir = Path(trainer.log_dir) / f"{split_name}_panels"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"epoch_{trainer.current_epoch:03d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        self._save_panel(
            trainer=trainer,
            pl_module=pl_module,
            batch=self.train_fixed_batch,
            split_name="train",
            indices=self.train_indices,
        )
        self._save_panel(
            trainer=trainer,
            pl_module=pl_module,
            batch=self.val_fixed_batch,
            split_name="val",
            indices=self.val_indices,
        )


# ============================================================
# Main training entry point
# ============================================================

def train_heatmap_experiment(
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
    resume_from_checkpoint: str | Path | None = None,
    ### wandb options
    wandb_logger: bool = False,
    wandb_log_model: bool = False,
    ###
):
    train_ds = HiddenObjectsHeatmap(split="train")
    val_ds = HiddenObjectsHeatmap(split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=heatmap_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=heatmap_collate,
        pin_memory=torch.cuda.is_available(),
    )

    lit_model = HeatmapLightningModule(
        model_class_path=_class_to_import_path(model_class),
        model_config=model_config,
        lr=lr,
        weight_decay=weight_decay,
    )

    ### TensorBoard logger
    if not wandb_logger:
        logger_version = (
            _extract_logger_version_from_ckpt(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        )

        logger = TensorBoardLogger(
            save_dir="runs",
            name=experiment_name,
            version=logger_version,
        )
        checkpoint_cb = ModelCheckpoint(
            dirpath=Path(logger.log_dir) / "checkpoints",
            filename="{epoch:03d}-{val_mae:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    ### wandb logger (maybe needs some work)

    wandb_dir = Path("runs_wandb") / experiment_name
    if wandb_logger:
        logger = WandbLogger(
            project="heatmap-prediction",
            entity="adlcv-project",
            name=experiment_name,
            save_dir=wandb_dir,
            log_model=wandb_log_model,  # set True if you want checkpoints logged as W&B artifacts
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=wandb_dir / "checkpoints",
            filename="{epoch:03d}-{val_mae:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )
    ###

    

    vis_cb = FixedInspectionPanelCallback(
        train_dataset=train_ds,
        val_dataset=val_ds,
        max_examples=num_inspection_examples,
        train_seed=inspection_seed,
        val_seed=inspection_seed + 1,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_cb, vis_cb],
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
    return trainer, lit_model


# ============================================================
# Loading helper
# ============================================================


def load_trained_heatmap_module(
    checkpoint_path: str | Path,
    map_location: Optional[str | torch.device] = None,
) -> HeatmapLightningModule:
    """
    Loads a trained module directly from checkpoint, without requiring the caller
    to re-supply model_class or model_config.

    This works because the checkpoint stores:
        - model_class_path
        - model_config
        - lr
        - weight_decay

    Example:
        lit_model = load_trained_heatmap_module(
            "runs/clip_tf_fusion/version_0/checkpoints/last.ckpt"
        )
        lit_model.eval()
    """
    return HeatmapLightningModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        map_location=map_location,
    )
