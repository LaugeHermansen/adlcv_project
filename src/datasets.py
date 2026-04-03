import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

from src.globals import HF_CACHE_DIR, PLACES365_ROOT


class HiddenObjectsBase(Dataset):
    def __init__(self, split="train", image_size=512):
        self.hf_data = load_dataset(
            "marco-schouten/hidden-objects",
            split=split,
            cache_dir=HF_CACHE_DIR,
            streaming=False,
        )

        self.image_size = image_size
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def _load_image(self, bg_path):
        img_path = PLACES365_ROOT / str(bg_path)
        return self.transform(Image.open(img_path).convert("RGB"))

class HiddenObjects(HiddenObjectsBase):
    def __init__(self, split="train", image_size=512):
        super().__init__(split=split, image_size=image_size)

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        image = self._load_image(item["bg_path"])
        bbox = torch.tensor(item["bbox"], dtype=torch.float32) * self.image_size

        return {
            "image": image,
            "bbox": bbox,
            "label": item["label"],
            "class": item["fg_class"],
            "image_reward_score": item["image_reward_score"],
            "confidence": item["confidence"],
        }

class HiddenObjectsImageLevel(HiddenObjectsBase):
    def __init__(self, split="train", image_size=512):
        super().__init__(split=split, image_size=image_size)

        bg_paths = np.asarray(self.hf_data["bg_path"])
        bg_codes, unique_bg_paths = pd.factorize(bg_paths, sort=False)

        order = np.argsort(bg_codes, kind="mergesort")
        sorted_codes = bg_codes[order]

        _, starts, counts = np.unique(
            sorted_codes,
            return_index=True,
            return_counts=True,
        )
        ends = starts + counts

        self.unique_bg_paths = unique_bg_paths
        self.order = order
        self.starts = starts
        self.ends = ends

    def __len__(self):
        return len(self.unique_bg_paths)

    def __getitem__(self, idx):
        bg_path = self.unique_bg_paths[idx]
        row_idx = self.order[self.starts[idx]:self.ends[idx]]

        anns = self.hf_data.select(row_idx.tolist())
        image = self._load_image(bg_path)

        boxes = torch.tensor(anns["bbox"], dtype=torch.float32) * self.image_size
        labels = torch.tensor(anns["label"], dtype=torch.long)
        confidences = torch.tensor(anns["confidence"], dtype=torch.float32)
        reward_scores = torch.tensor(anns["image_reward_score"], dtype=torch.float32)
        entry_ids = torch.tensor(anns["entry_id"], dtype=torch.long)

        return {
            "image": image,
            "bg_path": str(bg_path),
            "boxes": boxes,
            "labels": labels,
            "classes": anns["fg_class"],
            "confidences": confidences,
            "image_reward_scores": reward_scores,
            "sources": anns["source"],
            "entry_ids": entry_ids,
        }

class HiddenObjectsImageClassLevel(HiddenObjectsBase):
    def __init__(self, split="train", image_size=512):
        super().__init__(split=split, image_size=image_size)

        entry_ids = np.asarray(self.hf_data["entry_id"])
        entry_codes, unique_entry_ids = pd.factorize(entry_ids, sort=False)

        order = np.argsort(entry_codes, kind="mergesort")
        sorted_codes = entry_codes[order]

        _, starts, counts = np.unique(
            sorted_codes,
            return_index=True,
            return_counts=True,
        )
        ends = starts + counts

        self.unique_entry_ids = unique_entry_ids
        self.order = order
        self.starts = starts
        self.ends = ends

    def __len__(self):
        return len(self.unique_entry_ids)

    def __getitem__(self, idx):
        entry_id = self.unique_entry_ids[idx]
        row_idx = self.order[self.starts[idx]:self.ends[idx]]

        anns = self.hf_data.select(row_idx.tolist())

        # assumes one bg_path per entry_id group
        bg_path = anns["bg_path"][0]
        image = self._load_image(bg_path)

        boxes = torch.tensor(anns["bbox"], dtype=torch.float32) * self.image_size
        labels = torch.tensor(anns["label"], dtype=torch.long)
        confidences = torch.tensor(anns["confidence"], dtype=torch.float32)
        reward_scores = torch.tensor(anns["image_reward_score"], dtype=torch.float32)
        entry_ids = torch.tensor(anns["entry_id"], dtype=torch.long)

        return {
            "image": image,
            "bg_path": str(bg_path),
            "entry_id": int(entry_id),
            "boxes": boxes,
            "labels": labels,
            "classes": anns["fg_class"],
            "confidences": confidences,
            "image_reward_scores": reward_scores,
            "sources": anns["source"],
            "entry_ids": entry_ids,
        }