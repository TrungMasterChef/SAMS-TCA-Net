"""Train SAMS-TCA-Net with CrossEntropyLoss."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.data import create_dataloaders
from src.models import SAMSTCANet
from src.utils import get_device, load_config, make_dummy_dataloaders, save_checkpoint, set_seed


def build_dataloaders(config: dict) -> tuple:
    """Create real-data loaders, or dummy loaders when data files are absent."""
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    batch_size = int(config["training"]["batch_size"])

    if (data_dir / "X.npy").exists() and (data_dir / "y.npy").exists():
        return create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=int(data_cfg.get("num_workers", 0)),
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            seed=int(config.get("seed", 42)),
            augment=bool(data_cfg.get("augment", True)),
        )

    print(f"Data not found in {data_dir}. Using dummy data.")
    return make_dummy_dataloaders(
        num_samples=int(data_cfg.get("dummy_samples", 128)),
        seq_len=int(data_cfg["seq_len"]),
        num_channels=int(config["model"]["num_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        batch_size=batch_size,
    )


def train(config_path: str) -> None:
    """Run model training."""
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    device = get_device(str(config["training"].get("device", "auto")))
    train_loader, val_loader, _ = build_dataloaders(config)

    model_cfg = config["model"]
    model = SAMSTCANet(
        num_channels=int(model_cfg["num_channels"]),
        num_classes=int(model_cfg["num_classes"]),
        hidden_channels=int(model_cfg.get("hidden_channels", 64)),
        num_blocks=int(model_cfg.get("num_blocks", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        input_layout="btc",
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    best_val_loss = float("inf")
    checkpoint_path = config["training"].get("checkpoint_path", "outputs/sams_tca_net.pt")
    for epoch in range(1, int(config["training"].get("epochs", 10)) + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)

        model.eval()
        val_loss = 0.0
        val_items = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                loss = criterion(model(x), y)
                val_loss += loss.item() * x.size(0)
                val_items += x.size(0)

        train_loss = total_loss / max(1, total_items)
        val_loss = val_loss / max(1, val_items)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_path, model, config)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sams_tca.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
