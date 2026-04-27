from src.models.model2 import DepthFiLMHeatmapModel
from src.train import train_heatmap_experiment

model_class = DepthFiLMHeatmapModel
model_config = {
    
}

experiment_name = 'model2'

train_heatmap_experiment(
    model_class=model_class,
    model_config=model_config,
    experiment_name=experiment_name,
    max_epochs=10,
    num_workers=16,
    lr=1e-4,
    weight_decay=1e-4,
    resume_from_checkpoint=f'runs/{experiment_name}/version_0/checkpoints/last.ckpt'
)