import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_minmax(x):
    xmin = x.min()
    xmax = x.max()
    return (x - xmin) / (xmax - xmin) if xmax > xmin else np.zeros_like(x)


def prepare_image(image):
    image = to_numpy(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.max() > 1.0:
        image = image / 255.0

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    return image


def overlay_heatmap_on_image(image, heatmap, alpha=0.4, cmap="jet"):
    heatmap = normalize_minmax(to_numpy(heatmap))
    heatmap_rgb = plt.get_cmap(cmap)(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heatmap_rgb, 0.0, 1.0)


def plot_boxes_with_confidence(
    sample,
    ax=None,
    max_boxes=None,
    selection="top",
    seed=None,
    linewidth=2.0,
    use_only_positives=False,
):
    image = prepare_image(sample["image"])
    boxes = to_numpy(sample["boxes"])
    confidences = to_numpy(sample["confidences"])

    if use_only_positives:
        labels = to_numpy(sample["labels"])
        keep = labels == 1
        boxes = boxes[keep]
        confidences = confidences[keep]

    if max_boxes is not None and len(boxes) > max_boxes:
        if selection == "random":
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(boxes), size=max_boxes, replace=False)
        else:
            keep = np.argsort(confidences)[-max_boxes:]

        boxes = boxes[keep]
        confidences = confidences[keep]

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(image)
    ax.axis("off")

    if len(boxes) == 0:
        return ax

    conf = normalize_minmax(confidences)

    patches = []
    colors = []
    for box, c in zip(boxes, conf):
        x, y, w, h = box
        patches.append(Rectangle((x, y), w, h))
        colors.append((1.0 - c, c, 0.0, 1.0))

    ax.add_collection(
        PatchCollection(
            patches,
            facecolor="none",
            edgecolors=colors,
            linewidths=linewidth,
        )
    )
    return ax


def show_sample(
    sample,
    heatmap_fn,
    alpha=0.4,
    cmap="jet",
    plot_max_boxes=20,
    plot_selection="top",
    use_only_positives_for_boxes=True,
):
    image = prepare_image(sample["image"])
    heatmap = heatmap_fn(sample)
    heatmap = to_numpy(heatmap)
    overlay = overlay_heatmap_on_image(image, heatmap, alpha=alpha, cmap=cmap)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_boxes_with_confidence(
        sample,
        ax=axes[0],
        max_boxes=plot_max_boxes,
        selection=plot_selection,
        use_only_positives=use_only_positives_for_boxes,
    )

    plt.suptitle("class: " + sample['class'])
    title0 = f"Image + boxes (subset of {plot_selection} {plot_max_boxes})" if plot_max_boxes else "Image + boxes"
    axes[0].set_title(title0)

    axes[1].imshow(heatmap, cmap=cmap)
    axes[1].axis("off")
    axes[1].set_title("Heatmap")

    axes[2].imshow(overlay)
    axes[2].axis("off")
    axes[2].set_title("Overlay")

    plt.tight_layout()
    plt.show()

    return heatmap, overlay