"""Plot training/validation learning curves for every trained model.

Single-split version (no k-fold): each model has one ``history.csv`` under its
output directory, so we plot one clean curve per model instead of mean +/- std
shadows. Produces a combined 2x2 comparison, one figure per metric, and a
summary CSV of the best-epoch metrics.

Usage:
    python scripts/plot_learning_curves.py
    python scripts/plot_learning_curves.py --outputs-dir outputs --dest outputs/learning_curves
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Modern, publication-grade plot style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.5,
        "axes.edgecolor": "#333333",
        "axes.axisbelow": True,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.fontsize": 9,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": True,
        "legend.shadow": True,
        "grid.color": "#d0d0d0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
    }
)

# Display name -> output directory (relative to --outputs-dir). The proposed
# model is listed first and drawn with emphasis.
MODELS: dict[str, str] = {
    "MSCA-G": "msca_net",
    "InceptionTime": "baselines/inception_time_baseline",
    "FCN": "baselines/fcn_1d",
    "1D-CNN": "baselines/simple_cnn_1d",
    "TCN": "baselines/tcn_1d",
    "MambaSL": "baselines/mamba_sl",
    "PatchTST": "baselines/patchtst",
    "TSMixer": "baselines/tsmixer",
    "Transformer": "baselines/transformer",
}

COLORS: dict[str, str] = {
    "MSCA-G": "#db2777",        # pink (proposed, emphasised)
    "InceptionTime": "#ea580c", # orange
    "FCN": "#9333ea",           # purple
    "1D-CNN": "#16a34a",        # green
    "TCN": "#2563eb",           # blue
    "MambaSL": "#0891b2",       # cyan
    "PatchTST": "#ca8a04",      # gold
    "TSMixer": "#dc2626",       # red
    "Transformer": "#64748b",   # slate
}

# History-CSV column name -> human-readable metric title.
METRICS: list[tuple[str, str]] = [
    ("train_loss", "Training Loss"),
    ("val_loss", "Validation Loss"),
    ("train_accuracy", "Training Accuracy"),
    ("val_accuracy", "Validation Accuracy"),
]


def load_history(history_path: Path) -> dict[str, np.ndarray] | None:
    """Load a ``history.csv`` into column arrays, or None if missing/empty."""
    if not history_path.exists():
        return None
    with history_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        try:
            columns[key] = np.array([float(row[key]) for row in rows], dtype=float)
        except (ValueError, TypeError):
            continue
    return columns


def _line_style(model_name: str) -> dict:
    """Emphasise the proposed model with a thicker line."""
    if model_name == "MSCA-G":
        return {"linewidth": 3.2, "alpha": 1.0, "zorder": 5}
    return {"linewidth": 2.0, "alpha": 0.9, "zorder": 3}


def _finalise_axis(ax: plt.Axes, metric: str, title: str) -> None:
    ax.set_xlabel("Epoch", fontweight="bold", fontsize=11)
    ax.set_ylabel(title, fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=8)
    legend = ax.legend(loc="best", framealpha=0.95, fancybox=True, shadow=True, fontsize=8.5, edgecolor="#cccccc")
    legend.get_frame().set_linewidth(1.2)
    ax.grid(True, which="major", color="#d0d0d0", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.grid(True, which="minor", color="#e5e5e5", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#333333")
    if "accuracy" in metric.lower():
        ax.set_ylim([0, 1.05])
    else:
        ax.set_ylim(bottom=0)
    ax.set_facecolor("#f8f9fa")


def plot_combined(data: dict[str, dict[str, np.ndarray]], dest: Path, dpi: int) -> None:
    """Combined 2x2 figure: train/val loss and train/val accuracy."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle("Learning Curves Across Models", fontsize=14, fontweight="bold", y=0.997)
    for (metric, title), ax in zip(METRICS, axes.ravel()):
        for model_name, columns in data.items():
            if metric not in columns:
                continue
            epochs = columns.get("epoch", np.arange(1, len(columns[metric]) + 1))
            ax.plot(epochs, columns[metric], color=COLORS.get(model_name, "#333333"), label=model_name, **_line_style(model_name))
        _finalise_axis(ax, metric, title)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = dest / f"learning_curves_comparison.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"saved {path}")
    plt.close(fig)


def plot_individual(data: dict[str, dict[str, np.ndarray]], dest: Path, dpi: int) -> None:
    """One figure per metric."""
    for metric, title in METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("white")
        for model_name, columns in data.items():
            if metric not in columns:
                continue
            epochs = columns.get("epoch", np.arange(1, len(columns[metric]) + 1))
            ax.plot(epochs, columns[metric], color=COLORS.get(model_name, "#333333"), label=model_name, **_line_style(model_name))
        _finalise_axis(ax, metric, title)
        fig.tight_layout()
        name = metric.replace("_", "-")
        for ext in ("png", "pdf"):
            fig.savefig(dest / f"learning_curve_{name}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"saved learning_curve_{name}.(png|pdf)")
        plt.close(fig)


def write_summary(data: dict[str, dict[str, np.ndarray]], dest: Path) -> None:
    """Write best-epoch metrics (by val_macro_f1, else val_accuracy) per model."""
    rows: list[dict[str, object]] = []
    for model_name, columns in data.items():
        selector = columns.get("val_macro_f1", columns.get("val_accuracy"))
        if selector is None:
            continue
        best = int(np.argmax(selector))
        rows.append(
            {
                "model": model_name,
                "best_epoch": int(columns.get("epoch", np.arange(1, len(selector) + 1))[best]),
                "train_loss": round(float(columns["train_loss"][best]), 4) if "train_loss" in columns else "",
                "val_loss": round(float(columns["val_loss"][best]), 4) if "val_loss" in columns else "",
                "train_accuracy": round(float(columns["train_accuracy"][best]), 4) if "train_accuracy" in columns else "",
                "val_accuracy": round(float(columns["val_accuracy"][best]), 4) if "val_accuracy" in columns else "",
            }
        )
    if not rows:
        return
    path = dest / "learning_curves_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--dest", default="outputs/learning_curves")
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict[str, np.ndarray]] = {}
    for model_name, rel_dir in MODELS.items():
        columns = load_history(outputs_dir / rel_dir / "history.csv")
        if columns is None:
            print(f"skip {model_name}: no history at {outputs_dir / rel_dir / 'history.csv'}")
            continue
        data[model_name] = columns

    if not data:
        raise SystemExit("No history.csv files found. Train models first (scripts/run_models.py).")

    print(f"plotting {len(data)} models: {', '.join(data)}")
    plot_combined(data, dest, args.dpi)
    plot_individual(data, dest, args.dpi)
    write_summary(data, dest)
    print("done.")


if __name__ == "__main__":
    main()
