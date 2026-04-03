from src.globals import PLACES365_ROOT
import torchvision.datasets as datasets

root = PLACES365_ROOT
dataset = datasets.Places365(root=root, split='train-standard', small=False, download=True)
print(f"Downloaded {len(dataset)} images to {root}")
