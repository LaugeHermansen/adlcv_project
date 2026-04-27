from pathlib import Path

import torch

from src.models.model1 import PatchFeatureFiLMDecoderHeatmapModel
from src.hidden_objects_dataset import BoxBetaModeHeatmap
from src.train import train_heatmap_experiment
from src.evaluation_pipeline import (
    evaluation_pipeline,
    get_heatmap_model_evaluation_fn,
    mean_heatmap_score_fn,
    summarize_results,
)

model_class = PatchFeatureFiLMDecoderHeatmapModel
model_config = {}

experiment_name = 'model1'

trainer, lit_model = train_heatmap_experiment(
    model_class=model_class,
    model_config=model_config,
    experiment_name=experiment_name,
    max_epochs=10,
    num_workers=4,
    use_augmentation=True,
    heatmap_fn=BoxBetaModeHeatmap(),
    use_saved_heatmaps=False,
    # resume_from_checkpoint=f'runs/{experiment_name}/version_0/checkpoints/last.ckpt'
)

# Evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
lit_model = lit_model.to(device).eval()

calculate_score_fn = get_heatmap_model_evaluation_fn(lit_model.model, device, mean_heatmap_score_fn)
output_df = evaluation_pipeline(calculate_score_fn)
summary_df = summarize_results(output_df)

print(summary_df)

save_dir = Path("evaluation_results")
save_dir.mkdir(exist_ok=True, parents=True)
output_df.to_csv(save_dir / f"{experiment_name}_raw.csv", index=False)
summary_df.to_csv(save_dir / f"{experiment_name}_summary.csv")
print(f"Results saved to {save_dir}")
