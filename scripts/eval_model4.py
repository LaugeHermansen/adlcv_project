import torch

from src.train import load_trained_heatmap_module
from src.evaluation_pipeline import mean_heatmap_evaluation_pipeline

ckpt_path = "runs/model4/version_21/checkpoints/last.ckpt"
model = load_trained_heatmap_module(ckpt_path, 'cuda')
# torch.load()
def heatmap_pred_fn(images, cats):
    model.eval()
    return model(images, cats)

mean_heatmap_evaluation_pipeline(
    model_name='model4',
    heatmap_pred_fn=heatmap_pred_fn,
    img_size=512,
    device='cuda',
)