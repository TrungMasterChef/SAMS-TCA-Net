"""Train SAMS-TCA-Net with CrossEntropyLoss."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import create_dataloaders
from src.models import build_model
from src.utils import (
    accuracy_score_np,
    confusion_matrix_np,
    count_parameters,
    f1_scores_from_confusion,
    get_device,
    load_config,
    matthews_corrcoef_from_confusion,
    make_dummy_dataloaders,
    per_class_f1_from_confusion,
    save_checkpoint,
    save_confusion_matrix_png,
    save_history_csv,
    save_history_plots,
    save_table_csv,
    set_seed,
)


def build_dataloaders(config: dict) -> tuple:
    """Create real-data loaders, or dummy loaders when data files are absent."""
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    batch_size = int(config["training"]["batch_size"])

    input_file = str(data_cfg.get("input_file", "X.npy"))
    label_file = str(data_cfg.get("label_file", "y.npy"))
    if (data_dir / input_file).exists() and (data_dir / label_file).exists():
        return create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=int(data_cfg.get("num_workers", 0)),
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            seed=int(config.get("seed", 42)),
            augment=bool(data_cfg.get("augment", True)),
            input_file=input_file,
            label_file=label_file,
            input_layout=str(data_cfg.get("input_layout", "ntc")),
            normalization=str(data_cfg.get("normalization", "train")),
            window_length=data_cfg.get("window_length"),
            crop_mode=str(data_cfg.get("crop_mode", "none")),
            temporal_stride=int(data_cfg.get("temporal_stride", 1)),
            transform=str(data_cfg.get("transform", "raw")),
            eval_num_crops=int(data_cfg.get("eval_num_crops", 1)),
        )

    print(f"Data not found in {data_dir}. Using dummy data.")
    return make_dummy_dataloaders(
        num_samples=int(data_cfg.get("dummy_samples", 128)),
        seq_len=int(data_cfg["seq_len"]),
        num_channels=int(config["model"]["num_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        batch_size=batch_size,
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    """Evaluate loss, accuracy, macro-F1, and weighted-F1 on a loader."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if x.ndim == 4:
                batch_size, num_crops, seq_len, channels = x.shape
                logits = model(x.reshape(batch_size * num_crops, seq_len, channels))
                logits = logits.reshape(batch_size, num_crops, -1).mean(dim=1)
            else:
                logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

    true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    confusion = confusion_matrix_np(true, pred, num_classes)
    macro_f1, weighted_f1 = f1_scores_from_confusion(confusion)
    return {
        "loss": total_loss / max(1, total_items),
        "accuracy": accuracy_score_np(true, pred),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": matthews_corrcoef_from_confusion(confusion),
        "confusion_matrix": confusion,
    }


def train(config_path: str) -> list[dict[str, float | int]]:
    """Run model training."""
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    device = get_device(str(config["training"].get("device", "auto")))
    train_loader, val_loader, _ = build_dataloaders(config)

    model = build_model(config["model"]).to(device)
    parameter_count = count_parameters(model)
    print(f"parameter_count={parameter_count}")

    training_cfg = config["training"]
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training_cfg.get("scheduler_factor", 0.5)),
            patience=int(training_cfg.get("scheduler_patience", 4)),
            min_lr=float(training_cfg.get("min_learning_rate", 1e-5)),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training_cfg.get("epochs", 10)),
            eta_min=float(training_cfg.get("min_learning_rate", 1e-5)),
        )
    best_checkpoint_path = training_cfg.get(
        "best_checkpoint_path",
        training_cfg.get("checkpoint_path", "outputs/best_checkpoint.pt"),
    )
    last_checkpoint_path = training_cfg.get("last_checkpoint_path", "outputs/last_checkpoint.pt")
    history_path = training_cfg.get("history_path", "outputs/history.csv")
    history_plot_path = training_cfg.get("history_plot_path", "outputs/history.png")
    val_confusion_npy_path = training_cfg.get("val_confusion_matrix_npy_path", "outputs/val_confusion_matrix.npy")
    val_confusion_png_path = training_cfg.get("val_confusion_matrix_png_path", "outputs/val_confusion_matrix.png")
    val_f1_scores_path = training_cfg.get("val_f1_scores_path", "outputs/val_f1_scores.csv")
    training_log_path = Path(training_cfg.get("training_log_path", "outputs/training_log.txt"))
    training_log_path.parent.mkdir(parents=True, exist_ok=True)
    training_log_path.write_text("", encoding="utf-8")
    patience = int(training_cfg.get("early_stopping_patience", 0))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    total_epochs = int(config["training"].get("epochs", 10))
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_items = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0))
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_loss = total_loss / max(1, total_items)
        train_true_np = np.concatenate(train_true) if train_true else np.array([], dtype=np.int64)
        train_pred_np = np.concatenate(train_pred) if train_pred else np.array([], dtype=np.int64)
        train_confusion = confusion_matrix_np(train_true_np, train_pred_np, int(config["model"]["num_classes"]))
        train_macro_f1, train_weighted_f1 = f1_scores_from_confusion(train_confusion)
        train_acc = accuracy_score_np(train_true_np, train_pred_np)
        val_metrics = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            int(config["model"]["num_classes"]),
        )
        val_loss = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_weighted_f1 = float(val_metrics["weighted_f1"])
        val_mcc = float(val_metrics["mcc"])
        val_confusion = val_metrics["confusion_matrix"]
        if not isinstance(val_confusion, np.ndarray):
            raise TypeError("Validation confusion matrix must be a NumPy array")
        improved = val_macro_f1 > best_val_macro_f1 + min_delta
        if improved:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "train_weighted_f1": train_weighted_f1,
            "train_mcc": matthews_corrcoef_from_confusion(train_confusion),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "val_mcc": val_mcc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "parameter_count": parameter_count,
            "is_best": int(improved),
        }
        history.append(row)
        save_history_csv(history_path, history)
        save_history_plots(history_plot_path, history)
        checkpoint_val_metrics = {
            "loss": val_loss,
            "accuracy": val_accuracy,
            "macro_f1": val_macro_f1,
            "weighted_f1": val_weighted_f1,
            "mcc": val_mcc,
        }
        save_checkpoint(
            last_checkpoint_path,
            model,
            config,
            extra={
                "epoch": epoch,
                "parameter_count": parameter_count,
                "val_metrics": checkpoint_val_metrics,
                "is_best": improved,
            },
        )
        if improved:
            Path(val_confusion_npy_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(val_confusion_npy_path, val_confusion)
            save_confusion_matrix_png(val_confusion_png_path, val_confusion)
            save_table_csv(val_f1_scores_path, per_class_f1_from_confusion(val_confusion))
            save_checkpoint(
                best_checkpoint_path,
                model,
                config,
                extra={
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                    "val_metrics": checkpoint_val_metrics,
                    "is_best": True,
                },
            )

        if scheduler is not None:
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_macro_f1)
            else:
                scheduler.step()

        best_marker = " *" if improved else ""
        log_line = (
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_accuracy:.4f}{best_marker} | "
            f"val_f1: {val_macro_f1:.4f} | "
            f"val_mcc: {val_mcc:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
            f"{row['epoch_seconds']:.1f}s"
        )
        print(log_line)
        with training_log_path.open("a", encoding="utf-8") as f:
            f.write(log_line + "\n")
        if patience > 0 and epochs_without_improvement >= patience:
            stop_line = f"early_stopping epoch={epoch} best_val_macro_f1={best_val_macro_f1:.4f}"
            print(stop_line)
            with training_log_path.open("a", encoding="utf-8") as f:
                f.write(stop_line + "\n")
            break
    return history


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sams_tca.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
