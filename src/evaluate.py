"""Evaluate SAMS-TCA-Net classification metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import create_dataloaders
from src.models import SAMSTCANet
from src.utils import (
    accuracy_score_np,
    confusion_matrix_np,
    f1_scores_from_confusion,
    get_device,
    load_config,
    make_dummy_dataloaders,
)


def load_model(config: dict, checkpoint_path: str | Path | None, device: torch.device) -> SAMSTCANet:
    """Build a model and optionally restore checkpoint weights."""
    model_cfg = config["model"]
    model = SAMSTCANet(
        num_channels=int(model_cfg["num_channels"]),
        num_classes=int(model_cfg["num_classes"]),
        hidden_channels=int(model_cfg.get("hidden_channels", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        input_layout="btc",
    ).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


def build_test_loader(config: dict) -> torch.utils.data.DataLoader:
    """Create the test DataLoader from real or dummy data."""
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    batch_size = int(config["training"]["batch_size"])
    if (data_dir / "X.npy").exists() and (data_dir / "y.npy").exists():
        _, _, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=int(data_cfg.get("num_workers", 0)),
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            seed=int(config.get("seed", 42)),
            augment=False,
        )
        return test_loader

    print(f"Data not found in {data_dir}. Using dummy data.")
    _, _, test_loader = make_dummy_dataloaders(
        num_samples=int(data_cfg.get("dummy_samples", 128)),
        seq_len=int(data_cfg["seq_len"]),
        num_channels=int(config["model"]["num_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        batch_size=batch_size,
    )
    return test_loader


def evaluate(config_path: str, checkpoint_path: str | None = None) -> dict[str, object]:
    """Evaluate accuracy, macro-F1, weighted-F1, and confusion matrix."""
    config = load_config(config_path)
    device = get_device(str(config["training"].get("device", "auto")))
    checkpoint_path = checkpoint_path or config["training"].get("checkpoint_path")
    model = load_model(config, checkpoint_path, device)
    test_loader = build_test_loader(config)

    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y.numpy())

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    confusion = confusion_matrix_np(true, pred, int(config["model"]["num_classes"]))
    macro_f1, weighted_f1 = f1_scores_from_confusion(confusion)
    metrics = {
        "accuracy": accuracy_score_np(true, pred),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion,
    }
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"weighted_f1={metrics['weighted_f1']:.4f}")
    print("confusion_matrix=")
    print(confusion)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sams_tca.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.checkpoint)
