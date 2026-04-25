from pathlib import Path

DATA_ROOT = Path('/dtu/blackhole/10/169104/data/adlcv')
#HF_CACHE_DIR = Path('/zhome/06/9/168972/.cache/huggingface')  # your own writable path
HF_CACHE_DIR = DATA_ROOT
PLACES365_ROOT = DATA_ROOT / 'Places365'
PLACES365_TRIMMED_ROOT = DATA_ROOT / 'Places365_trimmed'
HEATMAPS_ROOT = DATA_ROOT / 'Heatmaps'
