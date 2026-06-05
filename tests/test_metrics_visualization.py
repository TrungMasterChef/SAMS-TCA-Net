import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import compute_multiclass_auc, save_roc_curve_png, save_tsne_png


def test_auc_roc_and_tsne_artifacts(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.6, 0.2],
            [0.2, 0.1, 0.7],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float32,
    )
    metrics = compute_multiclass_auc(y_true, probabilities, num_classes=3)
    assert metrics["roc_auc_macro"] is not None
    assert metrics["roc_auc_micro"] is not None
    assert metrics["roc_auc_weighted"] is not None

    roc_path = tmp_path / "roc.png"
    tsne_path = tmp_path / "tsne.png"
    save_roc_curve_png(roc_path, y_true, probabilities, num_classes=3)
    save_tsne_png(tsne_path, probabilities, y_true, max_points=20)
    assert roc_path.exists()
    assert tsne_path.exists()
