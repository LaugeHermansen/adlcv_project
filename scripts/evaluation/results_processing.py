from src.evaluation_pipeline import EvaluationDataLine
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from typing import List

def analyse_results(results: pd.DataFrame) -> tuple[pd.DataFrame, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process the results of the evaluation pipeline and return a DataFrame with the relevant information.
    """
    
    grouped = results.groupby("label")["score"]

    cat_groups = results.groupby("category")

    for category, group in cat_groups:
        
        grouped_label = group.groupby("label")["score"]
        cat_mean = grouped_label.mean()
        cat_std = grouped_label.std()
        cat_count = grouped_label.count()

        cat_df = pd.DataFrame({
            "mean": cat_mean,
            "std": cat_std / cat_count**0.5,
            "count": cat_count,
        })
        print(f"Category: {category}")
        print(cat_df)
        print()

    mean = grouped.mean()
    std = grouped.std()
    count = grouped.count()
    sem = std / count**0.5

    summary_df = pd.DataFrame({
        "mean": mean,
        "sem": sem,
        "std": std,
        "count": count,
    })

    roc_auc_value = roc_auc_score(1-results["label"], results["score"])
    # print(f"ROC AUC: {roc_auc_value:.4f}")

    fpr, tpr, thresholds = roc_curve(1-results["label"], results["score"])

    return summary_df, roc_auc_value, fpr, tpr, thresholds
    
if __name__ == "__main__":
    results_df = pd.read_csv("evaluation_results/diffusion_results.csv")
    summary_df, roc_auc_value, fpr, tpr, thresholds = analyse_results(results_df)
    print(summary_df)
    print(f"ROC AUC: {roc_auc_value:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_value:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig("roc_curve.png")