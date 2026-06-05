"""Copy the MSCA-Net evaluation figures into paper/figures/ for the manuscript."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "confusion_matrix.png": "msca_confusion_matrix.png",
    "roc_curve.png": "msca_roc_curve.png",
    "tsne.png": "msca_tsne.png",
    "history.png": "msca_history.png",
}


def export(model_dir: str = "outputs/msca_net", dest: str = "paper/figures") -> list[Path]:
    """Copy figures from ``model_dir`` into ``dest`` with paper-friendly names."""
    src_dir = ROOT / model_dir
    dest_dir = ROOT / dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src_name, dest_name in SOURCES.items():
        src = src_dir / src_name
        if not src.exists():
            print(f"missing {src} (run evaluation first)")
            continue
        target = dest_dir / dest_name
        shutil.copyfile(src, target)
        copied.append(target)
        print(f"copied {src} -> {target}")
    return copied


if __name__ == "__main__":
    args = sys.argv[1:]
    export(*args)
