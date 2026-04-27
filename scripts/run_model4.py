from src.models.model4 import ObjectPlacementHeatmapModel
from src.train import train_heatmap_experiment

model_class = ObjectPlacementHeatmapModel
model_config = {
    
}

experiment_name = 'model2'

train_heatmap_experiment(
    model_class=model_class,
    model_config=model_config,
    experiment_name=experiment_name,
    max_epochs=5,
    num_workers=16,
    lr=1e-3,
    weight_decay=1e-2,
    # resume_from_checkpoint=f'runs/{experiment_name}/version_0/checkpoints/last.ckpt'
)