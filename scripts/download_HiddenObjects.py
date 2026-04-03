from src.globals import HF_CACHE_DIR
from datasets import load_dataset

dataset = load_dataset("marco-schouten/hidden-objects", cache_dir=HF_CACHE_DIR, streaming=False)

first_row = dataset["train"][0]
print(first_row)
