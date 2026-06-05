"""Train and evaluate SAMS-TCA-Net plus baseline model configs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate
from src.train import train
from src.utils import load_config


def default_config_paths() -> list[Path]:
    """Return the default model comparison config list."""
    return [
        Path("configs/sams_tca.yaml"),
        *sorted(Path("configs/baselines").glob("*.yaml")),
    ]


def run_models(config_paths: list[Path], output_csv: str | Path) -> None:
    """Train/evaluate each config and save a summary CSV."""
    rows: list[dict[str, object]] = []
    for config_path in config_paths:
        config = load_config(config_path)
        model_name = str(config["model"].get("name", config_path.stem))
        print(f"Running model config: {config_path}")
        train(str(config_path))
        checkpoint_path = config["training"].get("best_checkpoint_path", config["training"].get("checkpoint_path"))
        metrics = evaluate(str(config_path), checkpoint_path)
        rows.append(
            {
                "config": str(config_path),
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "parameter_count": metrics["parameter_count"],
                "checkpoint_path": metrics["checkpoint_path"],
                "history_path": config["training"].get("history_path"),
                "history_plot_path": config["training"].get("history_plot_path"),
                "val_confusion_matrix_png_path": config["training"].get("val_confusion_matrix_png_path"),
                "val_f1_scores_path": config["training"].get("val_f1_scores_path"),
                "test_confusion_matrix_png_path": config["training"].get("confusion_matrix_png_path"),
                "test_f1_scores_path": config["training"].get("f1_scores_path"),
            }
        )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved model comparison results to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", default="outputs/model_results.csv")
    parser.add_argument("configs", nargs="*", help="Optional config paths. Defaults to SAMS-TCA and baselines.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = [Path(path) for path in args.configs] if args.configs else default_config_paths()
    run_models(paths, args.output_csv)
