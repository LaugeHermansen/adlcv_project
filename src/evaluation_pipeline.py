from pathlib import Path

import pycocotools

from matplotlib import pyplot as plt
import math
from tqdm import tqdm
from src.globals import COCO_OOC_ROOT
import numpy as np
from typing import Dict, List, Callable, Literal, Tuple, cast
import torch
import pandas as pd
from dataclasses import asdict, dataclass

from src.coco_ooc_dataset.coco_ooc_loader import get_dataloader
from src.hidden_objects_dataset import HiddenObjectsHeatmap


@dataclass
class EvaluationDataLine:
    dataset_idx: int
    category: str
    label: int
    score: float|int
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float

BATCH_SCORE_TYPE = List[EvaluationDataLine]
BATCH_TYPE = Tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, list[int]]
BATCH_EVAL_SCORE_FN_TYPE = Callable[[torch.Tensor, list[str], torch.Tensor, torch.Tensor], list[float]]

def evaluation_pipeline(
        calculate_score_fn: BATCH_EVAL_SCORE_FN_TYPE,
        batch_size: int = 64,
        img_size: int = 512,
        dataset_step_size=100,
        max_in_context_pr_ooc=5,
        ) -> pd.DataFrame:

    """
    Run the evaluation pipeline for the COCO_OOC dataset.
    The pipeline consists of the following steps:
    1. Initialize the COCO_OOC dataloader and the HiddenObjectsHeatmap dataset to get the unique 
       classes.
    2. Iterate over the COCO_OOC dataloder and calculate the score for each batch using the provided 
       calculate_score_fn.
    3. Store the outputs in a list of EvaluationDataLine dataclasses, which contain the category, 
       label, score, box coordinates and dataset index for each sample in the batch.
    4. Convert the list of EvaluationDataLine dataclasses to a pandas DataFrame and return it.

    Args:
        - calculate_score_fn (BATCH_EVAL_SCORE_FN_TYPE): A function that takes a batch and 
            returns a list of scores - one score for each sample in the batch, 
            in the same order as the input batch. The dataset is a set of images and 
            bounding boxes, where each box is labeled as in-context (0) or out-of-context (1).

            Note that this is the function that will call the heatmap prediction model, and 
            then evaluate the likelihood of the bounding boxes based on the predicted heatmaps.
            That is, for each sample, it will use the predicted heatmap to compute a score that
            indicates how likely it is to have an object of the given class in the bounding box.
            Hence, for in-context objects, we expect the score to be higher, and for
            out-of-context objects, we expect the score to be lower.

            NOTE: This function can be implemented in various ways, depending on the choice of
            model. For this purpose, the function `get_heatmap_model_evaluation_fn` may come in
            handy, as it provides a way to create a calculate_score_fn from a heatmap prediction
            function and a batch score function. Take a look at that ;)

            The function should have the following signature:
            calculate_score_fn(
                imgs: torch.Tensor (batch of images, shape = [batch_size, channels, height, width]), 
                cats: list[str] ( list of categories for each sample in the batch), 
                boxes: torch.Tensor (batch of bounding boxes, shape = [batch_size, 4]), 
                labels: torch.Tensor (batch of labels (0=in context, 1=out of context), shape = [batch_size]), 
                dataset_indices: list[int] ( list of dataset indices for each sample in the batch)) -> list[float]

        - batch_size (int): The size of each batch to be processed. Default is 64.
        - img_size (int): The size to which the images will be resized (center-cropped) before 
        being fed to the model. Default is 512.
        - dataset_step_size (int): The step size for iterating over the dataset. Default is 100.
        - max_in_context_pr_ooc (int): The maximum number of in-context samples to include for 
        each out-of-context sample in the dataset. Default is 5.

        For the last four arguments, see the `get_dataloader` function in `coco_ooc_loader.py` for
        more details on how they affect the dataloader and the dataset.

    Returns:
        - pd.DataFrame: A DataFrame containing the evaluation results, with columns for 
            category, label, score, box coordinates and dataset index.
    
    """

    # Initialize COCO_OOC dataloader - and HiddenObjectsHeatmap dataset to get the unique classes
    print("Starting evaluation pipeline...")
    print("Initializing dataloader and dataset...")
    ho_ds = HiddenObjectsHeatmap()

    dl, coco_ds = get_dataloader(
        COCO_OOC_ROOT,
        target_classes=ho_ds.anno_dataset.unique_classes,
        batch_size=batch_size,
        dataset_step_size=dataset_step_size,
        max_in_context_pr_ooc=max_in_context_pr_ooc,
        img_size=img_size
    )

    # prepare iteration
    last_dataset_idx = -1

    outputs: BATCH_SCORE_TYPE = []


    print("Iterating over dataloader and calculating scores...")
    for batch in (bar := tqdm(dl, total=len(coco_ds))):
        batch = cast(BATCH_TYPE, batch)
        imgs, cats, boxes, labels, dataset_indices = batch

        # calculate the score of the batch - get the output lines for each sample in the batch
        scores = calculate_score_fn(imgs, cats, boxes, labels)
        outputs += [
            EvaluationDataLine(
                label=int(labels[i].item()), 
                category=cats[i],
                score=score, 
                box_x1=boxes[i][0].item(), 
                box_y1=boxes[i][1].item(), 
                box_x2=boxes[i][2].item(), 
                box_y2=boxes[i][3].item(),
                dataset_idx=dataset_indices[i]
            )
            for i, score in enumerate(scores)]

        # Sync tqdm progress with dataset index
        if dataset_indices[0] != last_dataset_idx:
            last_dataset_idx = dataset_indices[0]   
            bar.n = min(dataset_indices[0] + 1, len(coco_ds))
            bar.refresh()
    
    # convert outputs to dataframe
    output_df = pd.DataFrame([asdict(line) for line in tqdm(outputs, desc="Converting outputs to DataFrame")])

    print("Evaluation pipeline completed.")

    return output_df



def get_heatmap_pred_fn(model, device):
    def heatmap_pred_fn(imgs: torch.Tensor, cats: list[str]) -> torch.Tensor:
        # This is a heatmap prediction function that uses the provided model to predict heatmaps for the input images and categories.
        # The function should take care of any necessary preprocessing (e.g., normalization) before feeding the images to the model.
        # The function should return a batch of predicted heatmaps, with shape [batch_size, 1, img_size, img_size].
        model.eval()
        with torch.no_grad():
            imgs = imgs.to(device)
            heatmaps = model(imgs, cats)  # Assuming the model takes images and categories as input and returns heatmaps
            return heatmaps.cpu()
        
    return heatmap_pred_fn




def get_heatmap_model_evaluation_fn(
        heatmap_pred_fn: Callable[[torch.Tensor, list[str]]], 
        device: str,
        batch_score_fn: Callable[[torch.Tensor, torch.Tensor], list[float]],
    ) -> BATCH_EVAL_SCORE_FN_TYPE:

    """
    This function is a helper function that creates a calculate_score_fn for the evaluation pipeline,
    given a heatmap prediction function and a batch score function.
    
    The heatmap prediction function takes a batch of images and their corresponding categories, and
    returns a batch of predicted heatmaps. This is the function that will call the heatmap prediction 
    model. Note that the images in the COCO OOC dataset are centercropped (so they are quadratic) and 
    their size is fixed in `evaluation_pipeline`. The images are not normalized, so the heatmap 
    prediction function should take care of any necessary additional preprocessing (e.g., normalization) 
    before feeding the images to the model.
    
    The batch score function takes the predicted heatmaps and the bounding boxes, and returns a list of
    scores for each sample in the batch. See `mean_heatmap_score_fn` and `median_heatmap_score_fn` for
    examples of a batch score function.
    """

    # define the function (output) that will be used to evaluate a batch of samples
    def calculate_score_fn(
            imgs: torch.Tensor,
            cats: list[str],
            boxes: torch.Tensor,
            labels: torch.Tensor,
    ) -> list[float]:
        # unpack batch and move to device
        imgs = imgs.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        # Calculate scores
        with torch.no_grad():
            heatmaps = heatmap_pred_fn(imgs, cats)
        
        scores = batch_score_fn(heatmaps, boxes)

        return scores

    return calculate_score_fn

def mean_heatmap_score_fn(
        heatmaps: torch.Tensor, 
        boxes: torch.Tensor, 
    ) -> list[float]:
    """
    Calculate the mean heatmap score for in-context and out-of-context objects.
    That is, for each sample in the batch, we take the predicted heatmap and calculate 
    the mean value of the heatmap at area the bounding box covers.
    """
    # Get the heatmap values at the box locations
    scores = []
    assert heatmaps.shape[0] == boxes.shape[0], "Batch size of heatmaps and boxes must be the same"
    assert heatmaps.shape[1] == 1, "Heatmaps must have a single channel"
    assert len(heatmaps.shape) == 4, "Heatmaps must be 4D tensors"


    H, W = heatmaps.shape[2], heatmaps.shape[3]
    for box, heatmap_pred in zip(boxes, heatmaps.cpu().numpy()):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
        x1 = min(x1, W-1)
        y1 = min(y1, H-1)
        x2 = min(max(x2, x1+1), W)
        y2 = min(max(y2, y1+1), H)
        heatmap_mean_value = heatmap_pred[0, y1:y2, x1:x2].mean().item()
        scores.append(heatmap_mean_value)

    return scores

def median_heatmap_score_fn(
        heatmaps: torch.Tensor, 
        boxes: torch.Tensor, 
    ) -> list[float]:
    """
    Calculate the median heatmap score in bounding boxes.
    That is, for each sample in the batch, we take the predicted heatmap and calculate 
    the median value of the heatmap at area the bounding box covers.
    """
    # Get the heatmap values at the box locations
    scores = []
    assert heatmaps.shape[0] == boxes.shape[0], "Batch size of heatmaps and boxes must be the same"
    assert heatmaps.shape[1] == 1, "Heatmaps must have a single channel"
    assert len(heatmaps.shape) == 4, "Heatmaps must be 4D tensors"


    H, W = heatmaps.shape[2], heatmaps.shape[3]
    for box, heatmap_pred in zip(boxes, heatmaps.cpu().numpy()):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
        x1 = min(x1, W-1)
        y1 = min(y1, H-1)
        x2 = min(max(x2, x1+1), W)
        y2 = min(max(y2, y1+1), H)
        heatmap_median_value = np.median(heatmap_pred[0, y1:y2, x1:x2])
        scores.append(heatmap_median_value)

    return scores

def summarize_results(df: pd.DataFrame):
    """
    This function takes the output DataFrame from the evaluation pipeline and prints a 
    summary of the results.
    The summary includes the mean score, count of samples, and variance of scores for 
    each label (in-context and out-of-context).
    0 is the label for in-context objects, and 1 is the label for out-of-context objects.
    """

    groupby_result = df.groupby("label")
    mean = groupby_result["score"].mean()
    count = groupby_result["score"].count()
    var = groupby_result["score"].var()
    summary = pd.DataFrame({
        "mean_score": mean,
        "count": count,
        "var_score": var
    })
    return summary

def mean_heatmap_evaluation_pipeline(
        model_name: str, 
        heatmap_pred_fn: Callable[[torch.Tensor, list[str]], torch.Tensor], 
        img_size: int,
        device: str):
    """
    This is an example on how to use the evaluation pipeline with a dummy heatmap
    prediction function and the mean_heatmap_score_fn as the batch score function.
    
    """
    save_path = Path(f"evaluation_results") / f"{model_name}_summary.csv"
    save_path.parent.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calculate_score_fn = get_heatmap_model_evaluation_fn(heatmap_pred_fn, device, mean_heatmap_score_fn)
    output_df = evaluation_pipeline(
        calculate_score_fn, 
        batch_size=64,
        img_size=img_size,
        dataset_step_size=100,
        max_in_context_pr_ooc=5
    )
    print("Summarizing results...")
    summary_df = summarize_results(output_df)

    print("Summary of results:")
    print(summary_df)
    summary_df.to_csv(save_path)
    print(f"Summary saved to {save_path}")


# if __name__ == "__main__":
#     mean_heatmap_evaluation_pipeline()