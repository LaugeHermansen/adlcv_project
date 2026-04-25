import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.utils as vutils

from src.vae_dataset import VAEDataset, get_dataloader
from src.models.vae import get_pretrained_vae


class SpatialClassConditioner(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, height: int, width: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim)
        self.height = height
        self.width = width

    def forward(self, class_labels: torch.Tensor) -> torch.Tensor:
        x = self.embedding(class_labels)  # [B, emb_dim]
        return x[:, :, None, None].expand(-1, -1, self.height, self.width)  # [B, emb_dim, H, W]


def build_class_vocab(train_ds: VAEDataset, test_ds: VAEDataset):
    classes = sorted(set(train_ds.classes) | set(test_ds.classes))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx

def visualize_test_samples(vae, model, scheduler, class_cond, class_to_idx, test_ds, output_path: Path, scaling_factor, device: str = "cuda"):
    """
    Visualize test samples using the pre-trained VAE.
    take in the test dataset
    loop over 10 sample
    for each sample conditionally sample latent heatmap using ddpm scheduler and unet model conditioning on the class and latent image representation
    decode the sampled latent heatmap and the original image latent representation using the vae decoder
    save the original heatmap, reconstructed heatmap, original image and reconstructed image in a grid
    """

    vae.eval()
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = 10

    with torch.no_grad():

        batch_heatmap_latents = []
        batch_image_latents = []
        batch_class_labels = []

        for idx in range(min(num_samples, len(test_ds))):
            sample = test_ds[idx]
            heatmap_latents = sample["encoded_heatmap"].to(device).float()
            image_latents = sample["encoded_image"].to(device).float() * scaling_factor
            class_label = torch.tensor(class_to_idx[sample["class"]], device=device, dtype=torch.long)
            
            batch_class_labels.append(class_label)
            batch_heatmap_latents.append(heatmap_latents)
            batch_image_latents.append(image_latents)

        batch_heatmap_latents = torch.stack(batch_heatmap_latents, dim=0)
        batch_image_latents = torch.stack(batch_image_latents, dim=0)
        batch_class_labels = torch.stack(batch_class_labels, dim=0)

        # Decode original latents
        original_heatmap = vae.decode(batch_heatmap_latents).sample
        original_image = vae.decode(batch_image_latents / scaling_factor).sample
        
        # Conditionally sample latent heatmap using DDPM
        class_map = class_cond(batch_class_labels)
        sampled_latent = torch.randn_like(batch_heatmap_latents)
            
        for t in tqdm(scheduler.timesteps, leave=False):
            model_input = torch.cat([sampled_latent, batch_image_latents, class_map], dim=1)
            pred = model(model_input, t).sample
            sampled_latent = scheduler.step(pred, t, sampled_latent).prev_sample
            
        # Decode sampled latent
        reconstructed_heatmap = vae.decode(sampled_latent / scaling_factor).sample
        
        output_grid = []
        for idx in range(min(num_samples, len(test_ds))):
            
            # Save visualization grid
            grid = torch.stack([original_heatmap[idx], reconstructed_heatmap[idx], original_image[idx]], dim=0)
            grid = (grid + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            output_grid.append(grid)
        big_grid = torch.cat(output_grid, dim=0)
        vutils.save_image(big_grid, output_path, nrow=3, normalize=False)


def run_epoch(loader, model: torch.nn.Module, scheduler, class_cond: torch.nn.Module, optimizer, class_to_idx, device, train, scaling_factor):
    model.train(train)
    class_cond.train(train)

    running_loss = 0.0
    pbar = tqdm(loader, leave=False)

    for step, batch in enumerate(pbar):
        heatmap_latents = batch["encoded_heatmap"].to(device).float() * scaling_factor   # [B, C_h, H, W]
        image_latents = batch["encoded_image"].to(device).float() * scaling_factor        # [B, C_i, H, W]
        class_labels = torch.tensor([class_to_idx[c] for c in batch["class"]], device=device, dtype=torch.long)

        noise = torch.randn_like(heatmap_latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (heatmap_latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_heatmap_latents = scheduler.add_noise(heatmap_latents, noise, timesteps)

        class_map = class_cond(class_labels)  # [B, C_cls, H, W]
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
    num_epochs = 2000
    lr = 4e-5
    num_train_timesteps = 1000
    num_test_timesteps = 1000
    class_emb_channels = 32

    save_dir = Path("checkpoints/heatmap_spatial_diffusion")
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


    train_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    test_scheduler = DDPMScheduler(num_train_timesteps=num_test_timesteps)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(class_cond.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    visualize_test_samples(vae, model, test_scheduler, class_cond, class_to_idx, test_ds, save_dir / "initial_samples.jpg", vae.config.scaling_factor, device=device)

    best_val = float("inf")
    for epoch in range(num_epochs):
        train_loss = run_epoch(train_loader, model, train_scheduler, class_cond, optimizer, class_to_idx, device, train=True, scaling_factor=vae.config.scaling_factor)
        val_loss = run_epoch(val_loader, model, test_scheduler, class_cond, optimizer, class_to_idx, device, train=False, scaling_factor=vae.config.scaling_factor)
        
        train_output_path = save_dir / f"epoch_{epoch+1:03d}_train_samples.jpg"
        test_output_path = save_dir / f"epoch_{epoch+1:03d}_test_samples.jpg"
        visualize_test_samples(vae, model, test_scheduler, class_cond, class_to_idx, train_ds, train_output_path, vae.config.scaling_factor, device=device)
        visualize_test_samples(vae, model, test_scheduler, class_cond, class_to_idx, test_ds, test_output_path, vae.config.scaling_factor, device=device)

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
        }
        torch.save(ckpt, save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")

        print(f"epoch={epoch+1:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    print(f"done. best_val={best_val:.6f}")


if __name__ == "__main__":
    main()