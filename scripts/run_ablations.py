"""Train and evaluate all ablation configs in a directory."""

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


def run_ablations(config_dir: str | Path, output_csv: str | Path) -> None:
    """Train and evaluate every YAML config in an ablation directory."""
    config_paths = sorted(Path(config_dir).glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No ablation configs found in {config_dir}")

    rows: list[dict[str, object]] = []
    flag_names: list[str] = []
    for config_path in config_paths:
        variant = config_path.stem
        print(f"Running ablation: {variant}")
        train(str(config_path))

        config = load_config(config_path)
        checkpoint_path = config["training"].get("checkpoint_path")
        metrics = evaluate(str(config_path), checkpoint_path)
        model_cfg = config["model"]
        ablation_flags = {key: value for key, value in model_cfg.items() if isinstance(value, bool)}
        for key in ablation_flags:
            if key not in flag_names:
                flag_names.append(key)
        rows.append(
            {
                "variant": variant,
                "config": str(config_path),
                "checkpoint_path": checkpoint_path,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "roc_auc_macro": metrics["roc_auc_macro"],
                "roc_auc_micro": metrics["roc_auc_micro"],
                "roc_auc_weighted": metrics["roc_auc_weighted"],
                "parameter_count": metrics["parameter_count"],
                **ablation_flags,
            }
        )

    fieldnames = [key for key in rows[0] if key not in flag_names] + flag_names
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ablation results to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs/ablations")
    parser.add_argument("--output-csv", default="outputs/ablation_results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablations(args.config_dir, args.output_csv)
