from src.models.model4 import ObjectPlacementHeatmapModel
from src.train import train_heatmap_experiment

def main():
    model_class = ObjectPlacementHeatmapModel
    model_config = {
        
    }

    experiment_name = 'model4'

    train_heatmap_experiment(
        model_class=model_class,
        model_config=model_config,
        experiment_name=experiment_name,
        max_epochs=10,
        # num_workers=4,
        num_workers=0,
        lr=3e-4,
        weight_decay=1e-3,
        # resume_from_checkpoint=f'runs/{experiment_name}/version_21/checkpoints/last.ckpt'
    )

if __name__ == '__main__':
    main()