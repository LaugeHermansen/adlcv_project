# from save_dataset_priors import class_heatmaps_path
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import math

# mean_class_to_heatmaps = {}

# for filename in class_heatmaps_path.glob("*_mean.npy"):
#     class_label = filename.stem.split("_")[0]
#     mean_heatmap = np.load(filename)
#     mean_class_to_heatmaps[class_label] = mean_heatmap

# var_class_to_heatmaps = {}

# for filename in class_heatmaps_path.glob("*_var.npy"):
#     class_label = filename.stem.split("_")[0]
#     var_heatmap = np.load(filename)
#     var_class_to_heatmaps[class_label] = var_heatmap


# nrows = math.ceil(len(mean_class_to_heatmaps)**0.5)
# ncols = math.ceil(len(mean_class_to_heatmaps) / nrows)

# fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))

# for (class_label, mean_heatmap), ax in tqdm(
#     zip(mean_class_to_heatmaps.items(), axs.flatten()), total=len(mean_class_to_heatmaps)
#     ):
#     var_heatmap = var_class_to_heatmaps[class_label]

#     heatmap_stuff = np.hstack([mean_heatmap[0], var_heatmap[0]**0.5])
#     ax.imshow(heatmap_stuff, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
#     ax.set_title(f'Heatmap for Class {class_label}')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     # plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax)

# for ax in axs.flatten():
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig('dataset_priors.png')
# plt.show()


from save_dataset_priors import class_heatmaps_path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# ------------------------------------------------------------------
# Load heatmaps
# ------------------------------------------------------------------

mean_class_to_heatmaps = {}

for filename in class_heatmaps_path.glob("*_mean.npy"):
    class_label = filename.stem.split("_")[0]
    mean_heatmap = np.load(filename)
    mean_class_to_heatmaps[class_label] = mean_heatmap

var_class_to_heatmaps = {}

for filename in class_heatmaps_path.glob("*_var.npy"):
    class_label = filename.stem.split("_")[0]
    var_heatmap = np.load(filename)
    var_class_to_heatmaps[class_label] = var_heatmap

# ------------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------------

n_classes = len(mean_class_to_heatmaps)

nrows = math.ceil(n_classes ** 0.5)
ncols = math.ceil(n_classes / nrows)

nrows = 10
ncols = 5

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(2.5*ncols, 1.6*nrows)
)

axs = np.array(axs).flatten()

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

separator_width = 8

for (class_label, mean_heatmap), ax in tqdm(
    zip(mean_class_to_heatmaps.items(), axs),
    total=n_classes
):

    var_heatmap = var_class_to_heatmaps[class_label]

    mean_img = mean_heatmap[0]
    std_img = np.sqrt(var_heatmap[0])

    # White separator between the two images
    separator = np.ones((mean_img.shape[0], separator_width))

    combined_img = np.hstack([
        mean_img,
        separator,
        std_img
    ])

    ax.imshow(
        combined_img,
        cmap="hot",
        interpolation="nearest",
        vmin=0,
        vmax=1
    )

    # Shared title for both sub-images
    ax.set_title(
        f"{class_label}",
        fontsize=10,
        pad=6
    )

    # Labels under each sub-image
    h, w = mean_img.shape

    ax.text(
        w / 2,
        h + 6,
        "mean",
        ha="center",
        va="bottom",
        fontsize=8,
        color="white"
    )

    ax.text(
        w + separator_width + w / 2,
        h + 6,
        "std",
        ha="center",
        va="bottom",
        fontsize=8,
        color="white"
    )

    ax.axis("off")

# Turn off unused axes
for ax in axs[n_classes:]:
    ax.axis("off")

# ------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------

plt.subplots_adjust(
    wspace=0.15,
    hspace=0.4
)

plt.savefig(
    "dataset_priors.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()