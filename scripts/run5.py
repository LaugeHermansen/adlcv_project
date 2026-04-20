from src.models.model5 import SimplePlacementModel
from src.train import train_heatmap_experiment

model_class = SimplePlacementModel
model_config = {
    
}

experiment_name = 'model5'

train_heatmap_experiment(
    model_class=model_class,
    model_config=model_config,
    experiment_name=experiment_name,
    max_epochs=10,
    num_workers=16,
    resume_from_checkpoint=f'runs/{experiment_name}/version_3/checkpoints/last.ckpt'
)