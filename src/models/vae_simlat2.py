
from typing import Any, cast
from pathlib import Path

import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.models.vae import get_pretrained_vae, AutoencoderKL
from src.vae_dataset import generate_dataset, VAEDataset, get_dataloader


class ClassConditionerAdd(nn.Module):
    def __init__(self, classes: list[str], emb_dim: int, out_shape: tuple[int, int, int]):
        super().__init__()
        self.embedding = nn.Embedding(len(classes), emb_dim)
        self.linear = nn.Linear(emb_dim, out_shape[0] * out_shape[1] * out_shape[2])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.out_shape = out_shape

    def forward(self, class_labels: list[str], device: str) -> torch.Tensor:
        class_idx = torch.tensor([self.class_to_idx[c] for c in class_labels], device=device, dtype=torch.long)
        x = self.embedding(class_idx)  # [B, emb_dim]
        x = self.linear(x)  # [B, hidden_dim]
        x = x.view(-1, *self.out_shape)  # [B, *out_shape]
        return x
    
class ClassConditionerConcat(nn.Module):
    def __init__(self, classes: list[str], emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(len(classes), emb_dim)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def forward(self, class_labels: list[str], device: str, image: torch.Tensor) -> torch.Tensor:
        class_idx = torch.tensor([self.class_to_idx[c] for c in class_labels], device=device, dtype=torch.long)
        x = self.embedding(class_idx)[..., None, None]  # [B, emb_dim, 1, 1]
        x = x.expand(-1, -1, image.shape[2], image.shape[3])  # [B, emb_dim, H, W]
        x = torch.cat([x, image], dim=1)  # [B, emb_dim + C, H, W]
        return x


def visualize_vae_reconstructions(
        small_vae: AutoencoderKL, 
        dataset: VAEDataset,
        output_path: Path,
        vae: AutoencoderKL,
        device: str = "cuda",
        num_samples: int = 4,
        image_size=512,):

    # Get a single sample
    batch = []
    for idx in range(num_samples):
        batch.append(dataset[idx])
    batch = VAEDataset.collate_fn(batch)

    small_vae.eval()
    vae.eval()
    with torch.no_grad():
        original_encoded_image = cast(torch.Tensor, batch["encoded_image"]).to(device)  # [B, 4, H, W]
        original_encoded_heatmap = cast(torch.Tensor, batch["encoded_heatmap"]).to(device)  # [B, 4, H, W]
        original_decoded_heatmap = ((vae.decode(original_encoded_heatmap).sample + 1) / 2).clip(0, 1)  # [B, 3, H, W]

        predicted_latent_heatmap = small_vae.encode(original_encoded_image).latent_dist.mean  # [B, 4, H, W]
        predicted_encoded_heatmap = small_vae.decode(predicted_latent_heatmap).sample  # [B, 4, H, W]
        predicted_decoded_heatmap = ((vae.decode(predicted_encoded_heatmap).sample + 1) / 2).clip(0, 1)  # [B, 3, H, W]
        original_decoded_image = ((vae.decode(original_encoded_image).sample + 1) / 2).clip(0, 1)  # [B, 3, H, W]

        # Create a grid of images
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 18))
        for i in range(num_samples):
            axes[0, i].imshow(original_decoded_image[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title(f"{batch['class'][i]} - Original Image")
            axes[0, i].axis("off")
            
            axes[1, i].imshow(predicted_decoded_heatmap[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].set_title(f"{batch['class'][i]} - Predicted Image")
            axes[1, i].axis("off")
            
            axes[2, i].imshow(original_decoded_heatmap[i].permute(1, 2, 0).cpu().numpy())
            axes[2, i].set_title(f"{batch['class'][i]} - Original Heatmap")
            axes[2, i].axis("off")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()



def main_vae(
        image_size=512,
        batch_size=128,
        num_epoch=20000,
        output_dir: Path = Path("outputs/vae2_reconstructions"),
        num_samples=10,
        learning_rate=1e-4
):
    
    train_dataset = VAEDataset(split="train", image_size=image_size)
    test_dataset  = VAEDataset(split="test",  image_size=image_size)
    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    encoded_image_size = train_dataset[0]["encoded_heatmap"].shape[1]

    vae_small = AutoencoderKL(
        in_channels=4,
        out_channels=4,
        latent_channels=64,
        sample_size=encoded_image_size,
    )


    class_conditioner = ClassConditionerAdd(
        classes=train_dataset.classes,
        emb_dim=32,
        out_shape=(4, encoded_image_size, encoded_image_size)
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_conditioner.to(device)

    optimizer = torch.optim.Adam(vae_small.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss(reduction="none")
    vae_small.to(device)
    vae = get_pretrained_vae(device=device)

    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_vae_reconstructions(
        vae_small, 
        test_dataset, 
        output_dir / "initial_test_reconstruction.png", 
        vae,
        device=device, 
        num_samples=num_samples,
        )

    visualize_vae_reconstructions(
        vae_small, 
        train_dataset, 
        output_dir / "initial_train_reconstruction.png", 
        vae,
        device=device, 
        num_samples=num_samples,
        )

    for epoch in tqdm(range(num_epoch)):
        
        vae_small.train()
        total_loss = 0.0
        
        for i, batch in enumerate(bar := tqdm(train_dataloader, leave=False)):

            encoded_heatmap = cast(torch.Tensor, batch["encoded_heatmap"]).to(device)
            encoded_image = cast(torch.Tensor, batch["encoded_image"]).to(device)
            class_labels = batch["class"]

            optimizer.zero_grad()

            # vae_small_input = class_conditioner(class_labels, device, encoded_image)

            cls_emb = class_conditioner(class_labels, device)  # [B, latent_dim]
            latent_dist = vae_small.encode(encoded_image + cls_emb).latent_dist
            decoded = vae_small.decode(latent_dist.sample()).sample  # [B, 4, H, W]

            recon_loss = mse_loss(decoded, encoded_heatmap).mean(dim=(1,2,3)).sum()
            kl_loss = latent_dist.kl().mean()
            loss = recon_loss + 0.05*kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * encoded_heatmap.size(0)

            bar.set_description(f"Epoch {epoch+1} - Loss: {loss.item():.4f} - Average Loss: {total_loss / ((i + 1) * batch_size):.4f}")

        visualize_vae_reconstructions(
            vae_small, 
            test_dataset, 
            output_dir / f"test_reconstruction_{epoch+1}.png", 
            vae,
            device=device, 
            num_samples=num_samples,
            )

        visualize_vae_reconstructions(
            vae_small, 
            train_dataset, 
            output_dir / f"train_reconstruction_{epoch+1}.png", 
            vae,
            device=device, 
            num_samples=num_samples,
            )


        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main_vae()