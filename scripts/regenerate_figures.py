"""Re-render saved confusion matrices and training histories with the current
publication-quality plotting style.

This reads existing artifacts (``*.npy`` confusion matrices and ``history.csv``)
and rewrites their PNGs without retraining or re-evaluating any model. ROC and
t-SNE plots are not regenerated here because they require stored class
probabilities, which are produced by ``src.evaluate`` rather than persisted.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import save_confusion_matrix_png, save_history_plots


def _read_history(path: Path) -> list[dict[str, float | int]]:
    """Load a ``history.csv`` file into typed per-epoch rows."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, float | int]] = []
        for raw in reader:
            row: dict[str, float | int] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                number = float(value)
                row[key] = int(number) if key in {"epoch", "parameter_count", "is_best"} else number
            rows.append(row)
    return rows


def regenerate(outputs_dir: str | Path) -> list[Path]:
    """Re-render every confusion matrix and history plot under ``outputs_dir``."""
    outputs_path = Path(outputs_dir)
    rendered: list[Path] = []

    for npy_path in sorted(outputs_path.rglob("*confusion_matrix.npy")):
        confusion = np.load(npy_path)
        png_path = npy_path.with_suffix(".png")
        split = "Validation " if npy_path.name.startswith("val_") else "Test "
        save_confusion_matrix_png(png_path, confusion, title=f"{split}Confusion Matrix")
        rendered.append(png_path)
        print(f"rendered {png_path}")

    for history_path in sorted(outputs_path.rglob("history.csv")):
        rows = _read_history(history_path)
        if not rows:
            continue
        png_path = history_path.with_name("history.png")
        save_history_plots(png_path, rows)
        rendered.append(png_path)
        print(f"rendered {png_path}")

    return rendered


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = regenerate(args.outputs_dir)
    print(f"Regenerated {len(paths)} figure(s).")
