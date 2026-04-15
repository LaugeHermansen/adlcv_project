from pathlib import Path
import sys
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

from src.globals import HF_CACHE_DIR, PLACES365_ROOT, PLACES365_TRIMMED_ROOT, HEATMAPS_ROOT


class HiddenObjectsBase(Dataset):
    def __init__(self, split="train", use_trimmed=None, image_size=512):
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

        self.image_size = image_size
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


class HiddenObjectsImageClassLevelFast(HiddenObjectsBase):
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

        self.bg_path_arr = self.df["bg_path"].to_numpy()
        self.class_arr = self.df["fg_class"].to_numpy()
        self.bbox_arr = self.df["bbox"].to_numpy()
        self.label_arr = self.df["label"].to_numpy()
        self.conf_arr = self.df["confidence"].to_numpy()
        self.reward_arr = self.df["image_reward_score"].to_numpy()
        self.entry_id_arr = self.df["entry_id"].to_numpy()
        self.source_arr = self.df["source"].to_numpy()

    def __len__(self):
        return len(self.unique_entry_ids)

    def __getitem__(self, idx):
        idxs = self.order[self.starts[idx]:self.ends[idx]]

        boxes = torch.from_numpy(np.vstack(self.bbox_arr[idxs])).float().mul_(self.image_size)
        labels = torch.from_numpy(self.label_arr[idxs])
        confidences = torch.from_numpy(self.conf_arr[idxs])
        reward_scores = torch.from_numpy(self.reward_arr[idxs])
        entry_ids = torch.from_numpy(self.entry_id_arr[idxs])

        classes = self.class_arr[idxs]
        sources = self.source_arr[idxs]

        bg_path = self.bg_path_arr[idxs[0]]
        image = self._load_image(bg_path)

        return {
            "image": image,
            "bg_path": str(bg_path),
            "entry_id": int(entry_ids[0]),
            "boxes": boxes,
            "labels": labels,
            "class": classes[0],
            "confidences": confidences,
            "image_reward_scores": reward_scores,
            "sources": sources,
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
        image_size=512,
        heatmap_fn=BoxGaussianHeatmap(),
        use_fast_dataset=True,
        use_saved_heatmaps=True,
    ):
        if use_fast_dataset:
            self.anno_dataset = HiddenObjectsImageClassLevelFast(split=split, image_size=image_size)
        else:
            self.anno_dataset = HiddenObjectsImageClassLevel(split=split, image_size=image_size)
        self.heatmap_fn = heatmap_fn
        self.use_saved_heatmaps = use_saved_heatmaps

    def __len__(self):
        return len(self.anno_dataset)

    def __getitem__(self, idx):
        sample = self.anno_dataset[idx]

        bg_path = Path(sample["bg_path"])
        heatmap_path = HEATMAPS_ROOT / f"{bg_path.with_suffix("")}_{sample['class']}.tiff"

        if self.use_saved_heatmaps:
            if heatmap_path.exists():
                heatmap_img = Image.open(heatmap_path)
                if heatmap_img.mode != "F":
                    raise ValueError(f"Expected heatmap image to be grayscale, but got mode {heatmap_img.mode}")
                heatmap = self.anno_dataset.transform(heatmap_img)
            else:
                warnings.warn(f"Heatmap image not found at {heatmap_path}, computing heatmap on the fly")   
                heatmap = self.heatmap_fn(sample)
                # save image for future use
                heatmap_img = Image.fromarray((heatmap.numpy()), mode="F")
                heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                heatmap_img.save(heatmap_path, compression="tiff_lzw")
        else:
            heatmap = self.heatmap_fn(sample)

        return {
            **sample,
            "heatmap": heatmap,
        }

def heatmap_collate(batch):
    return {
        "image": torch.stack([sample["image"] for sample in batch], dim=0),
        "heatmap": torch.stack([sample["heatmap"] for sample in batch], dim=0),
        "class": [sample["class"] for sample in batch],
    }

if __name__ == "__main__":
    from tqdm import tqdm
    ds = HiddenObjectsHeatmap(split="train", image_size=512, use_fast_dataset=True, use_saved_heatmaps=True)
    ds_fast = HiddenObjectsHeatmap(split="train", image_size=512, use_fast_dataset=True, use_saved_heatmaps=False)
    N = len(ds)
    num_fracs = 1
    start_fracs = np.linspace(0, 1, num_fracs + 1).round(5)
    start_frac_idx = int(sys.argv[1])-1 if len(sys.argv) > 1 else 0
    start_idx = int(N * start_fracs[int(start_frac_idx)])
    end_idx = int(N * start_fracs[int(start_frac_idx) + 1])
    print(f"Processing samples {start_idx} to {end_idx} out of {N} (start_frac={start_fracs[int(start_frac_idx)]}, end_frac={start_fracs[int(start_frac_idx) + 1]})")

    for i in tqdm(range(start_idx, end_idx)):
        sample_1 = ds[i]
        # sample_2 = ds_fast[i]
        # assert sample_1["bg_path"] == sample_2["bg_path"]
        # assert sample_1["class"] == sample_2["class"]
        # assert torch.allclose(sample_1["heatmap"], sample_2["heatmap"]), f"Heatmaps differ for sample {i} (bg_path={sample_1['bg_path']}, class={sample_1['class']})"
        # assert torch.allclose(sample_1["image"], sample_2["image"]), f"Images differ for sample {i} (bg_path={sample_1['bg_path']}, class={sample_1['class']})"                                                                                               