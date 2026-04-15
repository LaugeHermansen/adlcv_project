import os
import shutil
from tqdm import tqdm

from src.hidden_objects_dataset import HiddenObjectsImageLevel
from src.globals import PLACES365_ROOT, PLACES365_TRIMMED_ROOT

train = HiddenObjectsImageLevel(split='train')
test = HiddenObjectsImageLevel(split='test')

rels = [str(p) for ds in [train, test] for p in ds.unique_bg_paths]
for rel in tqdm(rels):
    src = PLACES365_ROOT / rel
    dst = PLACES365_TRIMMED_ROOT / rel

    if not os.path.exists(src):
        raise FileNotFoundError
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
