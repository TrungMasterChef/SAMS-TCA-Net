import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import SAMSTCANet


def test_forward_btc_shape() -> None:
    model = SAMSTCANet(num_channels=3, num_classes=5, hidden_channels=16, num_blocks=2)
    x = torch.randn(4, 128, 3)
    logits = model(x)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_forward_bct_shape() -> None:
    model = SAMSTCANet(
        num_channels=3,
        num_classes=5,
        hidden_channels=16,
        num_blocks=2,
        input_layout="bct",
    )
    x = torch.randn(4, 3, 128)
    logits = model(x)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()
