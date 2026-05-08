"""
Evaluation utilities for multi-class and binary skin lesion classifiers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def collect_predictions(dataset, model) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect predicted probabilities and true labels from a TensorFlow dataset.

    Returns:
        `(y_prob, y_true)`. For multi-class models, `y_prob` has shape `(n, c)`.
    """
    probabilities = []
    labels = []

    for x_batch, y_batch in dataset:
        probabilities.append(model.predict(x_batch, verbose=0))
        labels.append(y_batch.numpy())

    y_prob = np.concatenate(probabilities)

    y_true_raw = np.concatenate(labels)
    if y_true_raw.ndim > 1:
        y_true = np.argmax(y_true_raw, axis=1)
    else:
        y_true = y_true_raw.astype(int).ravel()

    return y_prob, y_true


def evaluate_multiclass(dataset, model, class_names: list[str]) -> dict[str, float]:
    """
    Evaluate a seven-class model with accuracy, macro F1, top-3 accuracy, and reports.
    """
    y_prob, y_true = collect_predictions(dataset, model)
    y_pred = np.argmax(y_prob, axis=1)

    top3_hits = [
        y_true[i] in np.argsort(y_prob[i])[-3:]
        for i in range(len(y_true))
    ]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "top3_accuracy": float(np.mean(top3_hits)),
    }

    print("=== Multi-class Evaluation ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics


def select_threshold_for_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_target: float = 0.90,
) -> float:
    """
    Select an operating threshold that prioritises high recall.

    Among thresholds meeting the recall target, this chooses the one with the
    highest precision. If none meet the target, it falls back to the best F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    candidates = [
        (p, r, t)
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
        if r >= recall_target
    ]

    if candidates:
        _, _, best_threshold = max(candidates, key=lambda item: item[0])
        return float(best_threshold)

    f1_scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-8)
    return float(thresholds[int(np.argmax(f1_scores))])


def evaluate_binary(
    dataset,
    model,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Evaluate a binary malignant-vs-benign classifier.
    """
    y_prob, y_true = collect_predictions(dataset, model)
    y_prob = y_prob.ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    print("=== Binary Evaluation ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion matrix [TN FP; FN TP]:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["benign", "malignant"]))

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path | None = None,
) -> None:
    """
    Plot and optionally save a confusion matrix.
    """
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    output_path: str | Path | None = None,
) -> float:
    """
    Plot and optionally save the ROC curve for binary classification.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary Malignancy ROC Curve")
    plt.legend()
    plt.tight_layout()

    if threshold is not None:
        op_idx = int(np.argmin(np.abs(thresholds - threshold)))
        plt.scatter(fpr[op_idx], tpr[op_idx])
        plt.annotate(f"threshold={threshold:.2f}", (fpr[op_idx], tpr[op_idx]))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200)

    plt.show()
    return float(roc_auc)
