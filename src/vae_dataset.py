

from pathlib import Path
import sys
from typing import cast
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.hidden_objects_dataset import HiddenObjectsHeatmap, heatmap_collate
from src.models.vae import get_pretrained_vae
from src.globals import ENCODED_HEATMAPS_AND_IMAGES_ROOT

    
def generate_dataset(image_size=512, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = get_pretrained_vae(device=device)

    # dataset
    train_ds = HiddenObjectsHeatmap(
        split="train",
        image_size=image_size,
        use_fast_dataset=True,
        use_saved_heatmaps=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=heatmap_collate,
    )

    test_ds = HiddenObjectsHeatmap(
        split="test",
        image_size=image_size,
        use_fast_dataset=True,
        use_saved_heatmaps=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=heatmap_collate,
    )

    def _generate(loader: DataLoader, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        vae.eval()
        all_encoded_heatmaps = []
        all_encoded_images = []
        all_classes = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Encoding heatmaps"):
                heatmap = cast(torch.Tensor, batch["heatmap"]).to(device).unsqueeze(1)  # [B, 1, H, W]
                x = (heatmap.repeat(1, 3, 1, 1) * 2.0 - 1.0).to(device)  # [B, 3, H, W] in [-1, 1]
                image = batch["image"].to(device) * 2.0 - 1.0  # [B, 3, H, W] in [-1, 1]

                posterior = vae.encode(x).latent_dist
                z = posterior.mean
                all_encoded_heatmaps.append(z.detach().cpu())
                
                posterior_image = vae.encode(image).latent_dist
                z_image = posterior_image.mean
                all_encoded_images.append(z_image.detach().cpu())

                all_classes += batch["class"]
        
        all_encoded_heatmaps = torch.cat(all_encoded_heatmaps, dim=0)
        all_encoded_images = torch.cat(all_encoded_images, dim=0)
        torch.save(all_encoded_heatmaps, output_dir / "encoded_heatmaps.pt")
        torch.save(all_encoded_images, output_dir / "encoded_images.pt")
        with open(output_dir / "classes.pkl", "wb") as f:
            pickle.dump(all_classes, f)
        print(f"Saved encoded heatmaps and images to {output_dir}")

    _generate(test_loader, ENCODED_HEATMAPS_AND_IMAGES_ROOT / str(image_size) / "test")
    _generate(train_loader, ENCODED_HEATMAPS_AND_IMAGES_ROOT / str(image_size) / "train")

class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, image_size: int):
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.split = split
        self.image_size = image_size
        self.encoded_heatmaps = torch.load(ENCODED_HEATMAPS_AND_IMAGES_ROOT / str(image_size) / split / "encoded_heatmaps.pt")
        self.encoded_images = torch.load(ENCODED_HEATMAPS_AND_IMAGES_ROOT / str(image_size) / split / "encoded_images.pt")
        with open(ENCODED_HEATMAPS_AND_IMAGES_ROOT / str(image_size) / split / "classes.pkl", "rb") as f:
            self.classes = pickle.load(f)

    def __len__(self):
        return len(self.encoded_heatmaps)

    def __getitem__(self, idx):
        return {
            "encoded_heatmap": self.encoded_heatmaps[idx],
            "encoded_image": self.encoded_images[idx],
            "class": self.classes[idx],
        }

    @staticmethod
    def collate_fn(batch):
        encoded_heatmaps = torch.stack([item["encoded_heatmap"] for item in batch], dim=0)
        encoded_images = torch.stack([item["encoded_image"] for item in batch], dim=0)
        classes = [item["class"] for item in batch]
        return {
            "encoded_heatmap": encoded_heatmaps,
            "encoded_image": encoded_images,
            "class": classes,
        }
    
def get_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=VAEDataset.collate_fn,
        )

if __name__ == "__main__":
    try:
        image_size = int(sys.argv[1])
        batch_size = int(sys.argv[2])
    except (IndexError, ValueError):
        image_size = 256
        batch_size = 64
    generate_dataset(image_size, batch_size)