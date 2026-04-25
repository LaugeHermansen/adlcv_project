import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from tqdm import tqdm
import torchvision.utils as vutils

from src.vae_dataset import VAEDataset, get_dataloader
from src.models.vae import get_pretrained_vae


class SpatialClassConditioner(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, height: int, width: int):
        super().__init__()
        # extra index for null/unconditional embedding
        self.num_classes = num_classes
        self.null_class_idx = num_classes
        self.embedding = nn.Embedding(num_classes + 1, emb_dim)
        self.height = height
        self.width = width

    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        x = self.embedding(class_labels)  # [B, emb_dim]
        return x[:, :, None, None].expand(-1, -1, self.height, self.width)

    # def forward(self, image_latents: torch.Tensor) -> torch.Tensor:
    #     # identity mapping for image latents (no conditioning)
    #     return image_latents

class UnconditionedImageLatents(nn.Module):
    def __init__(self, image_channels: int, size: int):
        super().__init__()
        self.image_channels = image_channels
        self.size = size
        self.latent = nn.Parameter(data=torch.randn(1, image_channels, size, size), requires_grad=True)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.latent.expand(batch_size, -1, -1, -1)

def build_class_vocab(train_ds: VAEDataset, test_ds: VAEDataset):
    classes = sorted(set(train_ds.classes) | set(test_ds.classes))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


@torch.no_grad()
def sample_with_cfg(
    model,
    scheduler: DDIMScheduler,
    class_cond: SpatialClassConditioner,
    image_latents: torch.Tensor,
    unconditioned_image_latents: UnconditionedImageLatents,
    class_labels: torch.Tensor,
    latent_shape,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
):
    model.eval()
    class_cond.eval()

    batch_size = image_latents.shape[0]
    sampled_latent = torch.randn(latent_shape, device=device, dtype=image_latents.dtype)

    scheduler.set_timesteps(num_inference_steps, device=device)

    null_labels = torch.full_like(class_labels, fill_value=class_cond.null_class_idx)

    for t in tqdm(scheduler.timesteps, leave=False):
        # unconditional branch
        uncond_class_map = class_cond(null_labels)
        uncond_image_latent = unconditioned_image_latents(batch_size)
        uncond_input = torch.cat([sampled_latent, uncond_image_latent, uncond_class_map], dim=1)
        eps_uncond = model(uncond_input, t).sample

        # conditional branch
        cond_class_map = class_cond(class_labels)
        cond_input = torch.cat([sampled_latent, image_latents, cond_class_map], dim=1)
        eps_cond = model(cond_input, t).sample

        # classifier-free guidance
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        sampled_latent = scheduler.step(eps, t, sampled_latent).prev_sample

    return sampled_latent


@torch.no_grad()
def visualize_test_samples(
    vae,
    model,
    scheduler: DDIMScheduler,
    unconditioned_image_latents: UnconditionedImageLatents,
    class_cond,
    class_to_idx,
    test_ds,
    output_path: Path,
    scaling_factor,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 100,
    device: str = "cuda",
):
    """
    For each sample:
      - decode original heatmap latent
      - decode original image latent
      - sample reconstructed heatmap latent with DDIM + CFG
      - decode reconstructed heatmap
      - save grid: [original_heatmap, reconstructed_heatmap, original_image]
    """

    vae.eval()
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = min(10, len(test_ds))

    batch_heatmap_latents = []
    batch_image_latents = []
    batch_class_labels = []

    for idx in range(num_samples):
        sample = test_ds[idx]

        # assume dataset latents are UNscaled VAE latents
        heatmap_latents = sample["encoded_heatmap"].to(device).float()
        image_latents = sample["encoded_image"].to(device).float()
        class_label = torch.tensor(class_to_idx[sample["class"]], device=device, dtype=torch.long)

        batch_heatmap_latents.append(heatmap_latents)
        batch_image_latents.append(image_latents)
        batch_class_labels.append(class_label)

    batch_heatmap_latents = torch.stack(batch_heatmap_latents, dim=0)
    batch_image_latents = torch.stack(batch_image_latents, dim=0)
    batch_class_labels = torch.stack(batch_class_labels, dim=0)

    # decode originals consistently
    original_heatmap = vae.decode(batch_heatmap_latents).sample
    original_image = vae.decode(batch_image_latents).sample

    # sample scaled heatmap latent with DDIM + CFG
    sampled_latent = sample_with_cfg(
        model=model,
        scheduler=scheduler,
        class_cond=class_cond,
        image_latents=batch_image_latents*scaling_factor,
        unconditioned_image_latents=unconditioned_image_latents,
        class_labels=batch_class_labels,
        latent_shape=batch_heatmap_latents.shape,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    reconstructed_heatmap = vae.decode(sampled_latent / scaling_factor).sample

    output_grid = []
    for idx in range(num_samples):
        grid = torch.stack(
            [
                original_heatmap[idx],
                reconstructed_heatmap[idx],
                original_image[idx],
            ],
            dim=0,
        )
        grid = (grid + 1) / 2
        output_grid.append(grid)

    big_grid = torch.cat(output_grid, dim=0)
    vutils.save_image(big_grid, output_path, nrow=3, normalize=False)


def run_epoch(
    loader,
    model: torch.nn.Module,
    scheduler,
    class_cond: torch.nn.Module,
    unconditioned_image_latents: UnconditionedImageLatents,
    optimizer,
    class_to_idx,
    device,
    train,
    scaling_factor,
    cond_dropout_prob: float = 0.1,
):
    model.train(train)
    class_cond.train(train)

    running_loss = 0.0
    pbar = tqdm(loader, leave=False)

    for step, batch in enumerate(pbar):
        # assume stored latents are unscaled, so scale for diffusion model
        heatmap_latents = batch["encoded_heatmap"].to(device).float() * scaling_factor
        image_latents = batch["encoded_image"].to(device).float() * scaling_factor
        class_labels = torch.tensor(
            [class_to_idx[c] for c in batch["class"]],
            device=device,
            dtype=torch.long,
        )

        noise = torch.randn_like(heatmap_latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (heatmap_latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_heatmap_latents = scheduler.add_noise(heatmap_latents, noise, timesteps)

        # classifier-free guidance training:
        # randomly replace class labels with null label
        if train and cond_dropout_prob > 0:
            drop_mask = torch.rand(class_labels.shape[0], device=device) < cond_dropout_prob
            effective_labels = class_labels.clone()
            effective_labels[drop_mask] = class_cond.null_class_idx
            effective_image_latents = image_latents.clone()
            effective_image_latents[drop_mask] = unconditioned_image_latents(1)
        else:
            effective_labels = class_labels
            effective_image_latents = image_latents

        class_map = class_cond(effective_labels)
        model_input = torch.cat([noisy_heatmap_latents, image_latents, class_map], dim=1)

        with torch.set_grad_enabled(train):
            pred = model(model_input, timesteps).sample
            loss = F.mse_loss(pred, noise)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")

    return running_loss / max(len(loader), 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = 512
    batch_size = 32
    num_epochs = 100
    lr = 1e-4

    num_train_timesteps = 1000
    num_inference_steps = 200
    guidance_scale = 5.0
    cond_dropout_prob = 0.1

    class_emb_channels = 32

    save_dir = Path("checkpoints/heatmap_spatial_diffusion_cfg")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_ds = VAEDataset(split="train", image_size=image_size)
    test_ds = VAEDataset(split="test", image_size=image_size)
    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    classes, class_to_idx = build_class_vocab(train_ds, test_ds)
    with open(save_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    sample = train_ds[0]
    heatmap_channels, H, W = sample["encoded_heatmap"].shape
    image_channels = sample["encoded_image"].shape[0]

    model_in_channels = heatmap_channels + image_channels + class_emb_channels

    vae = get_pretrained_vae(device=device)

    model = UNet2DModel(
        sample_size=(H, W),
        in_channels=model_in_channels,
        out_channels=heatmap_channels,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ).to(device)

    class_cond = SpatialClassConditioner(
        num_classes=len(classes),
        emb_dim=class_emb_channels,
        height=H,
        width=W,
    ).to(device)

    unconditioned_image_latents = UnconditionedImageLatents(
        image_channels=image_channels,
        size=H,
    ).to(device)

    train_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    sample_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(class_cond.parameters()) + list(unconditioned_image_latents.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    visualize_test_samples(
        vae=vae,
        model=model,
        scheduler=sample_scheduler,
        unconditioned_image_latents=unconditioned_image_latents,
        class_cond=class_cond,
        class_to_idx=class_to_idx,
        test_ds=test_ds,
        output_path=save_dir / "initial_samples.jpg",
        scaling_factor=vae.config.scaling_factor,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    best_val = float("inf")
    for epoch in range(num_epochs):
        train_loss = run_epoch(
            train_loader,
            model,
            train_scheduler,
            class_cond,
            unconditioned_image_latents,
            optimizer,
            class_to_idx,
            device,
            train=True,
            scaling_factor=vae.config.scaling_factor,
            cond_dropout_prob=cond_dropout_prob,
        )

        val_loss = run_epoch(
            val_loader,
            model,
            train_scheduler,
            class_cond,
            unconditioned_image_latents,
            optimizer,
            class_to_idx,
            device,
            train=False,
            scaling_factor=vae.config.scaling_factor,
            cond_dropout_prob=0.0,
        )

        train_output_path = save_dir / f"epoch_{epoch+1:03d}_train_samples.jpg"
        test_output_path = save_dir / f"epoch_{epoch+1:03d}_test_samples.jpg"

        visualize_test_samples(
            vae=vae,
            model=model,
            scheduler=sample_scheduler,
            unconditioned_image_latents=unconditioned_image_latents,
            class_cond=class_cond,
            class_to_idx=class_to_idx,
            test_ds=train_ds,
            output_path=train_output_path,
            scaling_factor=vae.config.scaling_factor,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
        )

        visualize_test_samples(
            vae=vae,
            model=model,
            scheduler=sample_scheduler,
            unconditioned_image_latents=unconditioned_image_latents,
            class_cond=class_cond,
            class_to_idx=class_to_idx,
            test_ds=test_ds,
            output_path=test_output_path,
            scaling_factor=vae.config.scaling_factor,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
        )

        ckpt = {
            "model": model.state_dict(),
            "class_cond": class_cond.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "class_to_idx": class_to_idx,
            "heatmap_channels": heatmap_channels,
            "image_channels": image_channels,
            "class_emb_channels": class_emb_channels,
            "latent_hw": [H, W],
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "cond_dropout_prob": cond_dropout_prob,
        }
        torch.save(ckpt, save_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")

        print(f"epoch={epoch+1:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    print(f"done. best_val={best_val:.6f}")


if __name__ == "__main__":
    main()