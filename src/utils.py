"""Shared training and evaluation utilities."""

from __future__ import annotations

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


def save_checkpoint(path: str | Path, model: torch.nn.Module, config: dict[str, Any]) -> None:
    """Save a model checkpoint and its config."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
