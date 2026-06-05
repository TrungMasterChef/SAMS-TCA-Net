"""Shared training and evaluation utilities."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve the torch device from a config string."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def make_dummy_dataloaders(
    num_samples: int = 128,
    seq_len: int = 256,
    num_channels: int = 3,
    num_classes: int = 4,
    batch_size: int = 16,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create dummy [B, T, C] accelerometer DataLoaders for smoke tests."""
    x = torch.randn(num_samples, seq_len, num_channels)
    y = torch.randint(0, num_classes, (num_samples,))
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)

    datasets = (
        TensorDataset(x[:train_end], y[:train_end]),
        TensorDataset(x[train_end:val_end], y[train_end:val_end]),
        TensorDataset(x[val_end:], y[val_end:]),
    )
    return (
        DataLoader(datasets[0], batch_size=batch_size, shuffle=True),
        DataLoader(datasets[1], batch_size=batch_size, shuffle=False),
        DataLoader(datasets[2], batch_size=batch_size, shuffle=False),
    )


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute a confusion matrix with rows=true labels and columns=predictions."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def f1_scores_from_confusion(confusion: np.ndarray) -> tuple[float, float]:
    """Compute macro-F1 and weighted-F1 from a confusion matrix."""
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    macro_f1 = float(f1.mean()) if f1.size else 0.0
    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() > 0 else 0.0
    return macro_f1, weighted_f1


def per_class_f1_from_confusion(confusion: np.ndarray) -> list[dict[str, float | int]]:
    """Compute precision, recall, F1, and support for every class."""
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return [
        {
            "class_id": int(class_id),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1_score": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in range(confusion.shape[0])
    ]


def matthews_corrcoef_from_confusion(confusion: np.ndarray) -> float:
    """Compute multiclass Matthews correlation coefficient from a confusion matrix."""
    total = confusion.sum()
    if total == 0:
        return 0.0
    true_counts = confusion.sum(axis=1).astype(np.float64)
    pred_counts = confusion.sum(axis=0).astype(np.float64)
    correct = np.trace(confusion).astype(np.float64)
    numerator = correct * total - np.dot(true_counts, pred_counts)
    denominator_left = total**2 - np.dot(pred_counts, pred_counts)
    denominator_right = total**2 - np.dot(true_counts, true_counts)
    denominator = np.sqrt(denominator_left * denominator_right)
    return float(numerator / denominator) if denominator > 0 else 0.0


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save per-epoch training history to CSV."""
    if not rows:
        return
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_table_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save a list of dictionaries to CSV."""
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_confusion_matrix_png(path: str | Path, confusion: np.ndarray) -> None:
    """Save a confusion matrix heatmap as a PNG image."""
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(6, min(14, confusion.shape[0] * 0.6))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(confusion.shape[1]))
    ax.set_yticks(np.arange(confusion.shape[0]))
    threshold = confusion.max() / 2.0 if confusion.size and confusion.max() > 0 else 0.0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            value = int(confusion[row, col])
            text_color = "white" if value > threshold else "black"
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_history_plots(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save train/validation loss, accuracy, and macro-F1 curves."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in rows], label="valid")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, [row["train_macro_f1"] for row in rows], label="train")
    axes[2].plot(epochs, [row["val_macro_f1"] for row in rows], label="valid")
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint and its config."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "config": config}
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
