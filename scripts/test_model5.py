from src.hidden_objects_dataset import HiddenObjectsHeatmap
from src.models.model5 import SimplePlacementModel

ds = HiddenObjectsHeatmap(split="test")
model = SimplePlacementModel()

idx = 0
sample = ds[idx]

import torch
with torch.no_grad():
    out = model(
        sample['image'].unsqueeze(0),
        [sample['class']]
    )