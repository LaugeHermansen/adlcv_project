

from pathlib import Path
from typing import cast

import torch
from diffusers import AutoencoderKL

def get_pretrained_vae(device: str = "cuda"):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", device_map=device)
    # vae.to(torch.device(device))
    return vae

def visualize_model_quality(vae: AutoencoderKL, sample, output_dir: Path, device: str = "cuda"):
    vae.eval()
    with torch.no_grad():
        heatmap = cast(torch.Tensor, sample["heatmap"]).to(device).unsqueeze(1)  # [1, 1, H, W]
        x = heatmap.repeat(1, 3, 1, 1) * 2.0 - 1.0  # [1, 3, H, W] in [-1, 1]

        posterior = vae.encode(x).latent_dist
        z = posterior.mean
        x_recon = vae.decode(z).sample

        image = sample["image"].to(device)  # [1, 3, H, W] in [0, 1]
        posterior = vae.encode((image*2.0 - 1.0)).latent_dist
        z = posterior.mean
        image_recon = vae.decode(z).sample


    import matplotlib.pyplot as plt

    batch_size = sample["heatmap"].shape[0]
    fig, axes = plt.subplots(batch_size, 4, figsize=(10, 4 * batch_size))
    
    
    for idx in range(batch_size):
        # Convert to CPU and numpy for visualization
        heatmap_np = heatmap[idx].cpu().numpy()  # [1, H, W]
        recon_np = x_recon[idx].squeeze(0).cpu().numpy()  # [3, H, W]


        axes[idx, 0].imshow(heatmap_np.transpose(1, 2, 0).repeat(3, axis=2))
        axes[idx, 0].set_title("Original Heatmap")
        axes[idx, 0].axis("off")

        recon_rgb = (recon_np.transpose(1, 2, 0) + 1.0) / 2.0
        recon_rgb = recon_rgb.clip(0, 1)
        axes[idx, 1].imshow(recon_rgb)
        axes[idx, 1].set_title("Reconstructed Heatmap (RGB)")
        axes[idx, 1].axis("off")

        image = sample["image"][idx].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        image = image.clip(0, 1)
        axes[idx, 2].imshow(image)
        axes[idx, 2].set_title("Original Image")
        axes[idx, 2].axis("off") 


        recon_image_rgb = (image_recon[idx].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) / 2.0
        recon_image_rgb = recon_image_rgb.clip(0, 1)
        axes[idx, 3].imshow(recon_image_rgb)
        axes[idx, 3].set_title("Reconstructed Image")
        axes[idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_example.png")
    plt.close()