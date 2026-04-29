from pathlib import Path
import json
class_heatmaps_path = Path("class_heatmaps") / "numpy_arrays"
class_heatmaps_path.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    print("Visualizing dataset priors...")
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from tqdm import tqdm

    from src.hidden_objects_dataset import HiddenObjectsHeatmap

    print("Loading dataset...")

    dataset = HiddenObjectsHeatmap(use_fast_dataset=True, use_saved_heatmaps=True)

    class_to_heatmaps: dict[str, list[torch.Tensor]] = {}

    counts = {}

    # for i in tqdm(range(len(dataset))):
    for i in tqdm(range(0, len(dataset), 1)):
        sample = dataset[i]
        heatmap = sample['heatmap']
        class_label = sample['class']
        if class_label not in class_to_heatmaps:
            class_to_heatmaps[class_label] = []
        if class_label not in counts:
            counts[class_label] = 0
        counts[class_label] += 1
        class_to_heatmaps[class_label].append(heatmap.numpy())

    mean_heatmaps = {
        class_label: np.mean(np.stack(heatmaps), axis=0) 
        for class_label, heatmaps in class_to_heatmaps.items()
        }
    
    var_heatmaps = {
        class_label: np.var(np.stack(heatmaps), axis=0) 
        for class_label, heatmaps in class_to_heatmaps.items()
        }

    for class_label, mean_heatmap in mean_heatmaps.items():
        np.save(class_heatmaps_path / f"{class_label}_mean.npy", mean_heatmap)

    for class_label, var_heatmap in var_heatmaps.items():
        np.save(class_heatmaps_path / f"{class_label}_var.npy", var_heatmap)
    
    with open(class_heatmaps_path / "counts.json", "w") as f:
        json.dump(counts, f, indent=4)
