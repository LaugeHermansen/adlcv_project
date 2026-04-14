import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

from src.globals import HF_CACHE_DIR, PLACES365_ROOT, PLACES365_TRIMMED_ROOT


class HiddenObjectsBase(Dataset):
    def __init__(self, split="train", use_trimmed=None):
        """
        split: 'train' or 'test'
        """
        self.split = split
        self.hf_data = load_dataset(
            "marco-schouten/hidden-objects",
            split=split,
            cache_dir=HF_CACHE_DIR,
            streaming=False,
        )
        if use_trimmed is None:
            if os.path.exists(PLACES365_TRIMMED_ROOT):
                self.image_root = PLACES365_TRIMMED_ROOT
            elif os.path.exists(PLACES365_ROOT):
                self.image_root = PLACES365_ROOT
            else:
                raise FileNotFoundError

        self.image_size = 512
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
        ])

    def _load_image(self, bg_path):
        img_path = self.image_root / str(bg_path)
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)

class HiddenObjects(HiddenObjectsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # load all annotations in at once since arrow indexing turned out to be EXTREMELY slow, both in __init__ and __getitem__.
        # e.g. on the test split, even just this shows how ineffective the hf data is at retrieving values
        #   5.726876s:  np.asarray(self.hf_data["bg_path"])
        #   0.184582s:  self.df = self.hf_data.to_pandas()
        #   0.000072s:  np.asarray(self.df["bg_path"])
        # conclusion: get away from that HF format as soon as possible!
        self.df = self.hf_data.to_pandas()
        bg_paths = np.asarray(self.df["bg_path"])
        bg_codes, unique_bg_paths = pd.factorize(bg_paths, sort=False)

        order = np.argsort(bg_codes)
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
        anns = self.df.iloc[row_idx]

        image = self._load_image(bg_path)

        boxes = torch.tensor(np.vstack(anns["bbox"].to_numpy()), dtype=torch.float32) * self.image_size
        labels = torch.tensor(anns["label"].to_numpy(), dtype=torch.long)
        confidences = torch.tensor(anns["confidence"].to_numpy(), dtype=torch.float32)
        reward_scores = torch.tensor(anns["image_reward_score"].to_numpy(), dtype=torch.float32)
        entry_ids = torch.tensor(anns["entry_id"].to_numpy(), dtype=torch.long)

        return {
            "image": image,
            "bg_path": str(bg_path),
            "boxes": boxes,
            "labels": labels,
            "classes": anns["fg_class"].to_numpy(),
            "confidences": confidences,
            "image_reward_scores": reward_scores,
            "sources": anns["source"].to_numpy(),
            "entry_ids": entry_ids,
        }

class HiddenObjectsImageClassLevel(HiddenObjectsBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.df = self.hf_data.to_pandas()
        entry_ids = np.asarray(self.df["entry_id"])
        entry_codes, unique_entry_ids = pd.factorize(entry_ids, sort=False)

        order = np.argsort(entry_codes)
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

        anns = self.df.iloc[row_idx]

        # assumes one bg_path per entry_id group
        bg_path = anns.iloc[0]["bg_path"]
        image = self._load_image(bg_path)

        boxes = torch.tensor(np.vstack(anns["bbox"].to_numpy()), dtype=torch.float32) * self.image_size
        labels = torch.tensor(anns["label"].to_numpy(), dtype=torch.long)
        confidences = torch.tensor(anns["confidence"].to_numpy(), dtype=torch.float32)
        reward_scores = torch.tensor(anns["image_reward_score"].to_numpy(), dtype=torch.float32)
        entry_ids = torch.tensor(anns["entry_id"].to_numpy(), dtype=torch.long)

        return {
            "image": image,
            "bg_path": str(bg_path),
            "entry_id": int(entry_id),
            "boxes": boxes,
            "labels": labels,
            "class": anns.iloc[0]["fg_class"],
            "confidences": confidences,
            "image_reward_scores": reward_scores,
            "sources": anns["source"].to_numpy(),
            "entry_ids": entry_ids,
        }

def get_bbox_weights(
    sample,
    use_only_positives=True,
    use_reward_scores=False,
) -> torch.Tensor:
    weights = sample['confidences']

    if use_only_positives:
        weights = weights * sample['labels']

    if use_reward_scores:
        reward_scores = sample['image_reward_scores']
        vmin = reward_scores.min()
        vmax = reward_scores.max()
        if vmin == vmax:
            pass
        else:
            reward_scores = (reward_scores - vmin) / (vmax - vmin)
            weights = weights * reward_scores

    return weights

class NaiveHeatmap:
    def __call__(self, sample: dict) -> torch.Tensor:
        image = sample["image"]
        boxes = sample["boxes"]

        _, H, W = image.shape
        heatmap = torch.zeros((H, W))

        weights = get_bbox_weights(sample)
        boxes = boxes.round().to(int)

        for box, weight in zip(boxes, weights):
            if weight == 0:
                continue

            x, y, w, h = box
            x0 = x.item()
            y0 = y.item()
            x1 = x0 + w.item()
            y1 = y0 + h.item()
            heatmap[y0:y1, x0:x1] += weight

        return heatmap / (heatmap.max() + 1e-8)

class BoxGaussianHeatmap:
    def __init__(self, sigma_scale=0.35):
        self.sigma_scale = sigma_scale
        self.truncate_sigma = 4.0

    def __call__(self, sample: dict) -> torch.Tensor:
        image = sample["image"]
        boxes = sample["boxes"]

        _, H, W = image.shape
        heatmap = torch.zeros((H, W))

        weights = get_bbox_weights(sample)
        boxes = boxes.round()

        xs_full = torch.arange(W)
        ys_full = torch.arange(H)

        weights = get_bbox_weights(sample)

        for box, weight in zip(boxes, weights):
            if weight <= 0:
                continue

            x, y, w, h = box
            cx = x + 0.5 * w
            cy = y + 0.5 * h

            sigma_x = self.sigma_scale * w
            sigma_y = self.sigma_scale * h

            radius_x = int(np.ceil(self.truncate_sigma * sigma_x))
            radius_y = int(np.ceil(self.truncate_sigma * sigma_y))

            x0 = max(0, int(np.floor(float(cx) - radius_x)))
            x1 = min(W, int(np.ceil(float(cx) + radius_x + 1)))
            y0 = max(0, int(np.floor(float(cy) - radius_y)))
            y1 = min(H, int(np.ceil(float(cy) + radius_y + 1)))

            xs = xs_full[x0:x1].view(1, -1)
            ys = ys_full[y0:y1].view(-1, 1)

            g = torch.exp(
                -0.5 * (
                    ((xs - cx) / sigma_x) ** 2 +
                    ((ys - cy) / sigma_y) ** 2
                )
            )

            heatmap[y0:y1, x0:x1] += weight * g

        return heatmap / (heatmap.max() + 1e-8)

class HiddenObjectsHeatmap(Dataset):
    def __init__(
        self,
        split="train",
        heatmap_fn=BoxGaussianHeatmap(),
    ):
        self.anno_dataset = HiddenObjectsImageClassLevel(split=split)
        self.heatmap_fn = heatmap_fn

    def __len__(self):
        return len(self.anno_dataset)

    def __getitem__(self, idx):
        sample = self.anno_dataset[idx]
        target = self.heatmap_fn(sample)

        return {
            **sample,
            "heatmap": target,
        }

def heatmap_collate(batch):
    return {
        "image": torch.stack([sample["image"] for sample in batch], dim=0),
        "heatmap": torch.stack([sample["heatmap"] for sample in batch], dim=0),
        "class": [sample["class"] for sample in batch],
    }