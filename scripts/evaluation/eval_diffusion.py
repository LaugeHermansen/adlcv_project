from src.models.diffusion import load_checkpoint, sample_heatmap, DDPMScheduler, get_pretrained_vae
from src.evaluation_pipeline import mean_heatmap_evaluation_pipeline
from pathlib import Path
import torch

model, class_cond, optimizer, class_to_idx = load_checkpoint(Path("scripts/evaluation/diffusion_ckpt"), device="cuda")
scheduler = DDPMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(300)
vae = get_pretrained_vae(device="cuda")

model.eval()

def diffusion_heatmap_fn(images, class_labels):
    with torch.no_grad():
        batch_size = images.shape[0]
        batch_image_latents = vae.encode(images).latent_dist.sample()
        batch_class_labels = torch.tensor([class_to_idx[label] for label in class_labels], device=images.device)
        heatmaps = sample_heatmap(
             model=model,
             scheduler=scheduler,
             vae=vae,
             class_cond=class_cond,
             class_labels=batch_class_labels,
             image_latents=batch_image_latents,
             device=images.device,
        )
    return heatmaps.mean(dim=1, keepdim=True).clip(0, 1)

if __name__ == "__main__":
    mean_heatmap_evaluation_pipeline(
        "diffusion", diffusion_heatmap_fn, 512, batch_size=16, device="cuda",
        num_evaluation_steps=None)


